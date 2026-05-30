import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompts-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--base-lr", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args()


def main():
    parse_args()
    raise NotImplementedError(
        "TODO(nanoDSV4-GRPO): implement reasoning post-training: load a base/SFT "
        "checkpoint, sample multiple completions per prompt, score with rule-based "
        "verifiers, compute group-relative advantages, apply clipped policy loss "
        "with KL to the reference model, and save accepted reasoning traces for "
        "later distillation."
    )


if __name__ == "__main__":
    main()
