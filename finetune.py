import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import tiktoken
from tqdm import tqdm

from model import GPT
from schedulers import WarmupCosineScheduler


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


def find_response_start(text):
    markers = ["### Response:\n", "Answer: "]
    starts = []

    for marker in markers:
        pos = text.find(marker)
        if pos != -1:
            starts.append(pos + len(marker))

    if not starts:
        return 0

    return min(starts)


class InstructionDataset(Dataset):
    def __init__(self, path, block_size, enc, split="train", train_frac=0.9, data_format="auto"):
        self.block_size = block_size
        self.enc = enc
        self.data_format = infer_data_format(path, data_format)

        if self.data_format == "jsonl":
            examples = read_jsonl_examples(path)
        elif self.data_format == "text":
            examples = read_text_examples(path)
        else:
            raise ValueError("data_format must be 'auto', 'text', or 'jsonl'")

        n = int(train_frac * len(examples))

        if split == "train":
            self.examples = examples[:n]
        elif split == "val":
            self.examples = examples[n:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if isinstance(example, dict):
            prompt_text = example["prompt"]
            response_text = example["response"] + "<|endoftext|>"
            text = prompt_text + response_text
        else:
            text = example
            response_start = find_response_start(text)
            prompt_text = text[:response_start]

        ids = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        prompt_ids = self.enc.encode(prompt_text, allowed_special={"<|endoftext|>"})

        ids = ids[: self.block_size + 1]
        x = ids[:-1]
        y = ids[1:]

        loss_mask = [0.0] * len(y)
        for i in range(len(y)):
            token_position_in_text = i + 1
            if token_position_in_text >= len(prompt_ids):
                loss_mask[i] = 1.0

        pad_id = self.enc.eot_token
        pad_len = self.block_size - len(x)

        if pad_len > 0:
            x = x + [pad_id] * pad_len
            y = y + [pad_id] * pad_len
            loss_mask = loss_mask + [0.0] * pad_len

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float),
        )


def infer_data_format(path, data_format):
    if data_format != "auto":
        return data_format
    if path.endswith(".jsonl"):
        return "jsonl"
    return "text"


def read_text_examples(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    raw_examples = text.split("<|endoftext|>")
    examples = []
    for ex in raw_examples:
        ex = ex.strip()
        if ex:
            examples.append(ex + "<|endoftext|>")
    return examples


def read_jsonl_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            prompt = record.get("prompt")
            response = record.get("response")

            if prompt is None or response is None:
                text = record.get("text")
                if text:
                    examples.append(text)
                continue

            if response.endswith("<|endoftext|>"):
                response = response[: -len("<|endoftext|>")]

            examples.append({"prompt": prompt, "response": response})
    return examples


def show_debug_batch(dataset, enc, count=2):
    for i in range(min(count, len(dataset))):
        x, y, loss_mask = dataset[i]
        active = loss_mask.bool()
        prompt = enc.decode(x[~active].tolist())
        target = enc.decode(y[active].tolist())
        print(f"\n--- debug example {i} ---")
        print("prompt/context tokens:")
        print(prompt)
        print("supervised target tokens:")
        print(target)
        print(f"active loss tokens: {int(loss_mask.sum().item())}")


def masked_cross_entropy(logits, targets, loss_mask):
    # TODO(MoE): keep response-only LM loss masked, but decide whether router
    # balancing losses should see prompt tokens, response tokens, or all tokens.
    B, T, C = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, C),
        targets.view(B * T),
        reduction="none",
    )
    loss = loss.view(B, T)
    loss = loss * loss_mask
    return loss.sum() / loss_mask.sum().clamp_min(1.0)


@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=50):
    model.eval()
    losses = []

    for i, (x, y, loss_mask) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        loss_mask = loss_mask.to(device)

        # TODO(MoE): handle logits plus router aux losses/statistics during SFT.
        logits = model(x)
        loss = masked_cross_entropy(logits, y, loss_mask)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def load_pretrained_model(checkpoint_path, fallback_config, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_config = checkpoint["model_config"]
        state_dict = checkpoint["model_state_dict"]
    else:
        model_config = fallback_config
        state_dict = checkpoint

    # TODO(MoE): pass any saved MoE config through to GPT so SFT can continue a
    # sparse checkpoint without silently falling back to dense SwiGLU blocks.
    model = GPT(
        vocab_size=model_config["vocab_size"],
        block_size=model_config["block_size"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        n_kv_heads=model_config["n_kv_heads"],
        mode=model_config["mode"],
        pos_encoding=model_config["pos_encoding"],
    )
    model.load_state_dict(state_dict)
    return model, model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="instruction_train.txt")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-format", type=str, default="auto", choices=["auto", "text", "jsonl"])
    parser.add_argument("--output-dir", type=str, default="checkpoints/finetune")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--debug-batch", action="store_true")
    args = parser.parse_args()

    ddp, rank, local_rank, world_size, device = setup_distributed()
    is_main_process = rank == 0
    autocast_device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if ddp:
        dist.barrier()

    enc = tiktoken.get_encoding("gpt2")
    fallback_config = {
        "vocab_size": enc.n_vocab,
        "block_size": 512,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "n_kv_heads": 2,
        "mode": "gqa",
        "pos_encoding": "rope",
        # TODO(MoE): add dense fallback defaults for new MoE config fields so old
        # nanoGPT checkpoints remain loadable during migration experiments.
    }

    start_step = 0
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = torch.load(args.resume_from, map_location=device)
        model_config = resume_checkpoint["model_config"]
        # TODO(MoE): reconstruct router/expert modules from checkpoint config.
        model = GPT(
            vocab_size=model_config["vocab_size"],
            block_size=model_config["block_size"],
            d_model=model_config["d_model"],
            n_heads=model_config["n_heads"],
            n_layers=model_config["n_layers"],
            n_kv_heads=model_config["n_kv_heads"],
            mode=model_config["mode"],
            pos_encoding=model_config["pos_encoding"],
        )
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        start_step = resume_checkpoint.get("step", -1) + 1
        if is_main_process:
            print(f"resuming finetune from {args.resume_from} at step {start_step}")
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --resume-from is set")
        model, model_config = load_pretrained_model(args.checkpoint, fallback_config, device)

    model = model.to(device)

    train_dataset = InstructionDataset(
        args.data_path,
        model_config["block_size"],
        enc,
        split="train",
        data_format=args.data_format,
    )
    val_dataset = InstructionDataset(
        args.data_path,
        model_config["block_size"],
        enc,
        split="val",
        data_format=args.data_format,
    )

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

    if args.debug_batch and is_main_process:
        show_debug_batch(train_dataset, enc)
    if ddp:
        dist.barrier()

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
    scheduler = WarmupCosineScheduler(
        optimizer,
        args.base_lr,
        args.min_lr,
        args.warmup_steps,
        args.max_steps,
    )

    if args.resume_from:
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])

    train_config = dict(vars(args))
    train_config["world_size"] = world_size
    train_config["global_batch_size"] = args.batch_size * world_size

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
        for local_step, (x, y, loss_mask) in enumerate(infinite_loader(train_loader, train_sampler, start_epoch)):
            step = start_step + local_step

            if step >= args.max_steps:
                break

            lr = scheduler.step(step)

            if step > 0 and step % args.eval_interval == 0:
                if ddp:
                    dist.barrier()

                if is_main_process:
                    val_loss = estimate_loss(raw_model(), val_loader, device)
                    print(f"\nstep {step}: val loss {val_loss:.4f}")

                    raw_model().eval()
                    prompts = [
                        "### Instruction:\nExplain machine learning in one sentence.\n\n### Response:\n",
                        (
                            "### Instruction:\nAnswer the question in one short sentence.\n\n"
                            "### Input:\nWhat is the capital of France?\n\n"
                            "### Response:\n"
                        ),
                    ]
                    for prompt in prompts:
                        ids = enc.encode(prompt)
                        idx = torch.tensor([ids], dtype=torch.long, device=device)
                        out = raw_model().generate(
                            idx,
                            max_new_tokens=60,
                            temperature=0.7,
                            top_k=50,
                            eos_token_id=enc.eot_token,
                        )
                        print("\n--- sample ---")
                        print(enc.decode(out[0].tolist()))
                    raw_model().train()

                if ddp:
                    dist.barrier()

            x = x.to(device)
            y = y.to(device)
            loss_mask = loss_mask.to(device)

            with torch.autocast(
                device_type=autocast_device_type,
                dtype=torch.bfloat16,
                enabled=autocast_device_type == "cuda",
            ):
                # TODO(MoE): add router aux loss to masked LM loss and log router
                # diagnostics during instruction tuning.
                logits = model(x)
                loss = masked_cross_entropy(logits, y, loss_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if is_main_process:
                pbar.set_description(f"loss {loss.item():.4f} lr {lr:.2e}")
                pbar.update(1)

            if is_main_process and step > 0 and step % args.save_interval == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
                save_checkpoint(checkpoint_path, step)

        if is_main_process:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
            save_checkpoint(checkpoint_path, step)

    finally:
        if is_main_process and pbar is not None:
            pbar.close()
        if ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
