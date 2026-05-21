import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tiktoken
from tqdm import tqdm

from dataloader import TextDataset
from model import GPT
from schedulers import WarmupCosineScheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="mixed_train.txt")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=8)
    # TODO(MoE): add architecture flags here: --mlp-mode dense|switch|mixtral,
    # --num-experts, --top-k, --capacity-factor, --moe-layer-frequency,
    # --shared-experts, --aux-loss-weight, and --router-z-loss-weight.
    parser.add_argument("--attention-mode", type=str, default="gqa", choices=["mha", "gqa"])
    parser.add_argument("--encoding-mode", type=str, default="learned", choices=["learned", "sinusoidal", "rope"])
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--relative-lr", action="store_true")
    return parser.parse_args()


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


def get_run_name(attention_mode, n_heads, n_kv_heads, encoding_mode):
    # TODO(MoE): include MoE settings in the run name so dense and sparse
    # checkpoints do not collide.
    if attention_mode == "gqa":
        return f"gqa_q{n_heads}_kv{n_kv_heads}_{encoding_mode}"
    return f"{attention_mode}_{encoding_mode}"


def build_model_config(args, vocab_size):
    # TODO(MoE): save all MoE architecture knobs in checkpoints. Inference and
    # SFT should reconstruct the exact router/expert layout from this dict.
    return {
        "vocab_size": vocab_size,
        "block_size": args.block_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_kv_heads": args.n_kv_heads,
        "mode": args.attention_mode,
        "pos_encoding": args.encoding_mode,
    }


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

        # TODO(MoE): handle model outputs that include logits plus router aux
        # losses/statistics once sparse blocks are implemented.
        logits = model(x)
        B, T, C = logits.shape

        loss = F.cross_entropy(
            logits.view(B * T, C),
            y.view(B * T),
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    ddp, rank, local_rank, world_size, device = setup_distributed()
    is_main_process = rank == 0
    autocast_device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    checkpoint_root = "checkpoints"
    run_name = get_run_name(args.attention_mode, args.n_heads, args.n_kv_heads, args.encoding_mode)
    checkpoint_dir = os.path.join(checkpoint_root, run_name)
    if is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    model_config = build_model_config(args, enc.n_vocab)
    train_config = {
        "data_path": args.data_path,
        "batch_size": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "base_lr": args.base_lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "relative_lr": args.relative_lr,
        "world_size": world_size,
    }

    train_dataset = TextDataset(args.data_path, args.block_size, split="train")
    val_dataset = TextDataset(args.data_path, args.block_size, split="val")

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
    ).to(device)

    checkpoint = None
    start_step = 0
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        start_step = checkpoint.get("step", -1) + 1
        if is_main_process:
            print(f"resuming from {args.resume_from} at step {start_step}")

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
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
        return model.module if ddp else model

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

    try:
        for local_step, (x, y) in enumerate(infinite_loader(train_loader, train_sampler, start_epoch)):
            step = start_step + local_step

            if step >= args.max_steps:
                break

            scheduler_step = local_step if args.relative_lr else step
            lr = scheduler.step(scheduler_step)

            if step > 0 and step % 3000 == 0:
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
                    val_loss = estimate_loss(raw_model(), val_loader, device)
                    print(f"\nstep {step}: val loss {val_loss:.4f}")

                if ddp:
                    dist.barrier()

            x = x.to(device)
            y = y.to(device)

            with torch.autocast(
                device_type=autocast_device_type,
                dtype=torch.bfloat16,
                enabled=autocast_device_type == "cuda",
            ):
                # TODO(MoE): total loss should become LM loss + weighted router
                # aux losses. Log both pieces separately so router fixes do not
                # hide language-model regressions.
                logits = model(x)
                B, T, C = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, C),
                    y.view(B * T),
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if is_main_process:
                pbar.set_description(f"loss {loss.item():.4f} lr {lr:.2e}")
                pbar.update(1)

            if is_main_process and step > 0 and step % 10000 == 0:
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
