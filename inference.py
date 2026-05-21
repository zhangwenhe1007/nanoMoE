import torch
import tiktoken
import argparse
from model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--max-new-tokens", type=int, default=100)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--top-p", type=float, default=None)
parser.add_argument("--multiline", action="store_true")
parser.add_argument("--end-marker", type=str, default="END")

args = parser.parse_args()

checkpoint_path = args.checkpoint

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

checkpoint = torch.load(checkpoint_path, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model_config = checkpoint["model_config"]
    state = checkpoint["model_state_dict"]
else:
    model_config = {
        "vocab_size": vocab_size,
        "block_size": 512,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "n_kv_heads": 2,
        "mode": "gqa",
        "pos_encoding": "rope",
        # TODO(MoE): include dense fallback defaults for MoE config fields so
        # legacy dense checkpoints and new sparse checkpoints can share inference.
    }
    state = checkpoint

# TODO(MoE): reconstruct the exact MoE layout from model_config. Generation should
# ignore training-only aux losses while still exposing optional router stats for
# debugging expert specialization.
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

model.load_state_dict(state)

model = model.to(device)
model.eval()


def read_prompt(multiline, end_marker):
    if not multiline:
        return input("\nPrompt> ")

    print(f"\nPrompt> paste your prompt. End with a line containing only {end_marker}.")
    lines = []
    while True:
        line = input()
        if line == end_marker:
            break
        lines.append(line)
    return "\n".join(lines)


while True:
    prompt = read_prompt(args.multiline, args.end_marker)

    if prompt.lower() in {"exit", "quit", "q"}:
        break

    ids = enc.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=enc.eot_token,
    )

    text = enc.decode(out[0].tolist())
    print("\n" + text)
