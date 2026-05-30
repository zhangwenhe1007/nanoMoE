import argparse
import inspect
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tiktoken
from tqdm import tqdm

from config import MODEL_PRESETS, apply_model_preset, architecture_run_name, build_model_config
from dataloader import TextDataset
from model import GPT
from schedulers import WarmupCosineScheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="mixed_train.txt")
    parser.add_argument("--model-preset", type=str, default="dense_360m", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--tokenizer-name", type=str, default="gpt2", choices=["gpt2", "cl100k_base", "o200k_base"])
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=8)
    # TODO(nanoDSV4-scaling): add explicit presets such as tiny/base/medium that
    # keep active params, total params, batch size, sequence length, and token
    # budget in sync. Good English needs enough model capacity and many clean
    # tokens; the architecture alone will not rescue an under-trained model.
    parser.add_argument("--attention-mode", type=str, default="gqa", choices=["mha", "gqa"])
    parser.add_argument("--attention-impl", type=str, default="gqa", choices=["mha", "gqa", "mla"])
    parser.add_argument("--encoding-mode", type=str, default="rope", choices=["learned", "sinusoidal", "rope"])
    parser.add_argument("--mlp-impl", type=str, default="dense", choices=["dense", "deepseek_moe"])
    parser.add_argument("--num-experts", type=int, default=0)
    parser.add_argument("--num-shared-experts", type=int, default=0)
    parser.add_argument("--num-routed-experts", type=int, default=0)
    parser.add_argument("--routed-experts-per-token", type=int, default=0)
    parser.add_argument("--moe-layer-frequency", type=int, default=0)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--router-z-loss-weight", type=float, default=0.0)
    parser.add_argument("--global-balance-loss-weight", type=float, default=0.0)
    parser.add_argument("--sequence-balance-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--sparse-attention-impl",
        type=str,
        default="none",
        choices=["none", "sliding", "csa", "hca", "alternating_csa_hca"],
    )
    parser.add_argument("--use-mhc", action="store_true")
    parser.add_argument("--use-attention-residual", action="store_true")
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=12_000_000_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--sample-interval", type=int, default=3000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--relative-lr", action="store_true")
    args = parser.parse_args()
    args = apply_model_preset(args)
    return args


def estimate_parameter_count(config):
    vocab_size = config["vocab_size"]
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"] or n_heads
    d_head = d_model // n_heads
    hidden_dim = int((8 / 3) * d_model)
    hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)

    embedding_params = vocab_size * d_model
    attn_params = (
        d_model * d_model
        + d_model * n_kv_heads * d_head
        + d_model * n_kv_heads * d_head
        + d_model * d_model
    )
    mlp_params = d_model * hidden_dim * 3
    norm_params = 2 * d_model
    final_norm_params = d_model
    return embedding_params + n_layers * (attn_params + mlp_params + norm_params) + final_norm_params


def build_optimizer(model, args):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and not name.endswith("token_embedding.weight"):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer_kwargs = {
        "lr": args.base_lr,
        "betas": (args.beta1, args.beta2),
    }
    if "fused" in inspect.signature(torch.optim.AdamW).parameters and torch.cuda.is_available():
        optimizer_kwargs["fused"] = True

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **optimizer_kwargs,
    )


def setup_distributed():
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ddp, rank, local_rank, world_size, device


def infinite_loader(loader, sampler=None, start_epoch=0):
    epoch = start_epoch
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=50):
    model.eval()
    losses = []

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        # TODO(nanoDSV4-eval): log dense validation loss separately from router
        # losses and attention compression stats. Reasoning improvements should
        # be tracked with small exact-answer evals, not samples alone.
        output = model(x, return_output=True)
        logits = output.logits
        B, T, C = logits.shape

        loss = F.cross_entropy(
            logits.view(B * T, C),
            y.view(B * T),
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def autocast_dtype(args):
    if args.dtype == "bf16":
        return torch.bfloat16
    if args.dtype == "fp16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    ddp, rank, local_rank, world_size, device = setup_distributed()
    is_main_process = rank == 0
    autocast_device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_dtype = autocast_dtype(args)

    tokens_per_step = args.batch_size * args.grad_accum_steps * world_size * args.block_size
    if args.target_tokens and args.max_steps is None:
        args.max_steps = math.ceil(args.target_tokens / tokens_per_step)
    if args.warmup_steps is None:
        args.warmup_steps = min(2000, max(100, args.max_steps // 100))

    checkpoint_root = "checkpoints"

    enc = tiktoken.get_encoding(args.tokenizer_name)
    model_config = build_model_config(args, enc.n_vocab)
    run_name = architecture_run_name(model_config)
    checkpoint_dir = os.path.join(checkpoint_root, run_name)
    if is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

    train_config = {
        "data_path": args.data_path,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "global_batch_size": args.batch_size * world_size,
        "effective_global_batch_size": args.batch_size * args.grad_accum_steps * world_size,
        "tokens_per_step": tokens_per_step,
        "target_tokens": args.target_tokens,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "dtype": args.dtype,
        "compile": args.compile,
        "relative_lr": args.relative_lr,
        "world_size": world_size,
        # TODO(nanoDSV4-data): store dataset mixture metadata, token count,
        # dedup/filtering settings, and eval suite version here. Coherence and
        # basic factuality will mostly come from cleaner, larger, better-mixed
        # data plus enough optimization steps.
    }

    train_dataset = TextDataset(args.data_path, args.block_size, split="train", tokenizer_name=args.tokenizer_name)
    val_dataset = TextDataset(args.data_path, args.block_size, split="val", tokenizer_name=args.tokenizer_name)

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if ddp else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = GPT(
        vocab_size=model_config["vocab_size"],
        block_size=model_config["block_size"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        n_kv_heads=model_config["n_kv_heads"],
        mode=model_config["mode"],
        pos_encoding=model_config["pos_encoding"],
        model_config=model_config,
    ).to(device)

    if is_main_process:
        param_count = estimate_parameter_count(model_config)
        planned_tokens = args.max_steps * tokens_per_step
        print(f"model preset: {args.model_preset}")
        print(f"estimated parameters: {param_count / 1e6:.1f}M")
        print(f"tokens/optimizer-step: {tokens_per_step:,}")
        print(f"planned tokens: {planned_tokens:,}")
        print(f"max steps: {args.max_steps:,}, warmup steps: {args.warmup_steps:,}")

    checkpoint = None
    start_step = 0
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        start_step = checkpoint.get("step", -1) + 1
        if is_main_process:
            print(f"resuming from {args.resume_from} at step {start_step}")

    if args.compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = build_optimizer(model, args)
    scheduler_max_steps = args.max_steps
    if args.relative_lr:
        scheduler_max_steps = args.max_steps - start_step
        if scheduler_max_steps <= 0:
            raise ValueError("--max-steps must be greater than the checkpoint step when using --relative-lr")

    scheduler = WarmupCosineScheduler(
        optimizer,
        args.base_lr,
        args.min_lr,
        args.warmup_steps,
        scheduler_max_steps,
    )

    if checkpoint is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def raw_model():
        base_model = model.module if ddp else model
        return base_model._orig_mod if hasattr(base_model, "_orig_mod") else base_model

    def save_checkpoint(path, step):
        torch.save({
            "step": step,
            "model_config": model_config,
            "train_config": train_config,
            "model_state_dict": raw_model().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)

    pbar = tqdm(total=args.max_steps, initial=start_step) if is_main_process else None
    step = start_step - 1
    start_epoch = start_step // max(1, len(train_loader))
    data_iter = infinite_loader(train_loader, train_sampler, start_epoch)

    try:
        for local_step in range(args.max_steps - start_step):
            x, y = next(data_iter)
            step = start_step + local_step

            scheduler_step = local_step if args.relative_lr else step
            lr = scheduler.step(scheduler_step)

            if step > 0 and step % args.eval_interval == 0:
                if ddp:
                    dist.barrier()

                if is_main_process:
                    raw_model().eval()
                    val_loss = estimate_loss(raw_model(), val_loader, device, max_batches=args.eval_batches)
                    print(f"\nstep {step}: val loss {val_loss:.4f}")
                    raw_model().train()

                if ddp:
                    dist.barrier()

            if step > 0 and step % args.sample_interval == 0:
                if ddp:
                    dist.barrier()

                if is_main_process:
                    raw_model().eval()
                    sample_prompts = [
                        "Once upon a time",
                        "In simple terms, machine learning is",
                        "The history of the Roman Empire",
                        "Photosynthesis is the process by which",
                    ]

                    for prompt in sample_prompts:
                        ids = enc.encode(prompt)
                        idx = torch.tensor([ids], dtype=torch.long, device=device)
                        out = raw_model().generate(idx, max_new_tokens=30, eos_token_id=enc.eot_token)
                        print("\n--- sample ---")
                        print(enc.decode(out[0].tolist()))

                    raw_model().train()

                if ddp:
                    dist.barrier()

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_lm_loss = 0.0

            for micro_step in range(args.grad_accum_steps):
                if micro_step > 0:
                    x, y = next(data_iter)

                x = x.to(device)
                y = y.to(device)

                if ddp:
                    model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1

                with torch.autocast(
                    device_type=autocast_device_type,
                    dtype=amp_dtype,
                    enabled=autocast_device_type == "cuda" and args.dtype != "fp32",
                ):
                    # TODO(nanoDSV4-loss): total loss should be:
                    # LM loss + router z-loss + global/sequence balance losses +
                    # optional aux-loss-free bias updates. Log each term separately.
                    output = model(x, return_output=True)
                    logits = output.logits
                    B, T, C = logits.shape
                    lm_loss = F.cross_entropy(
                        logits.view(B * T, C),
                        y.view(B * T),
                    )
                    aux_loss = output.aux_loss
                    loss = lm_loss if aux_loss is None else lm_loss + aux_loss
                    loss = loss / args.grad_accum_steps

                loss.backward()
                total_loss += loss.item()
                total_lm_loss += lm_loss.item() / args.grad_accum_steps

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if is_main_process:
                pbar.set_description(f"loss {total_loss:.4f} lm {total_lm_loss:.4f} lr {lr:.2e}")
                pbar.update(1)

            if is_main_process and step > 0 and step % args.save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
                save_checkpoint(checkpoint_path, step)

        if is_main_process:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            save_checkpoint(checkpoint_path, step)

    finally:
        if is_main_process and pbar is not None:
            pbar.close()
        if ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
