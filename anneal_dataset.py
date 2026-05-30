import argparse
import os
import random
import sys

from datasets import load_dataset


END = "<|endoftext|>"


def clean(text):
    return " ".join(str(text or "").split())


def load_stream(dataset_name, *, config=None, split="train"):
    if config is None:
        return iter(load_dataset(dataset_name, split=split, streaming=True))
    return iter(load_dataset(dataset_name, config, split=split, streaming=True))


def make_source(name, weight, dataset_name, formatter, *, config=None, split="train"):
    try:
        iterator = load_stream(dataset_name, config=config, split=split)
    except Exception as exc:
        print(f"skipping {name}: failed to load {dataset_name}: {exc}")
        return None

    return {
        "name": name,
        "weight": weight,
        "dataset_name": dataset_name,
        "config": config,
        "split": split,
        "iterator": iterator,
        "formatter": formatter,
        "resets": 0,
    }


def reset_source(source):
    source["iterator"] = load_stream(
        source["dataset_name"],
        config=source["config"],
        split=source["split"],
    )
    source["resets"] += 1


def format_text(ex):
    text = ex.get("text") or ex.get("content") or ex.get("document") or ""
    text = clean(text)
    if not text:
        return None
    return text + END


def format_fineweb_edu(ex, min_score=None):
    if min_score is not None:
        score = ex.get("score", ex.get("educational_score", ex.get("int_score")))
        if score is not None and float(score) < min_score:
            return None
    return format_text(ex)


def format_code(ex):
    text = ex.get("code") or ex.get("content") or ex.get("text") or ""
    text = clean(text)
    if not text:
        return None
    return text + END


def next_formatted(source):
    while True:
        try:
            ex = next(source["iterator"])
        except StopIteration:
            reset_source(source)
            ex = next(source["iterator"])

        formatted = source["formatter"](ex)
        if formatted:
            return formatted


def normalize_sources(sources):
    total = sum(source["weight"] for source in sources)
    for source in sources:
        source["weight"] = source["weight"] / total


def build_sources(args):
    sources = []

    # TODO(nanoDSV4-anneal): make annealing data progressively higher quality:
    # more educational/factual/code/math late in training, less noisy web text,
    # and explicit eval sets for basic facts, sentence coherence, and arithmetic.
    fineweb_formatter = lambda ex: format_fineweb_edu(ex, args.min_fineweb_score)
    sources.append(make_source(
        "fineweb_edu",
        0.55,
        "HuggingFaceFW/fineweb-edu",
        fineweb_formatter,
        config=args.fineweb_config,
    ))

    sources.append(make_source(
        "cosmopedia",
        0.25,
        args.cosmopedia_dataset,
        format_text,
        config=args.cosmopedia_config,
    ))

    if args.code_source != "none":
        if args.code_source == "codeparrot":
            sources.append(make_source(
                "codeparrot_python",
                0.10,
                "codeparrot/github-code-clean",
                format_code,
            ))
        elif args.code_source == "starcoderdata":
            sources.append(make_source(
                "starcoderdata",
                0.10,
                "bigcode/starcoderdata",
                format_code,
            ))
        elif args.code_source == "stack_v2":
            sources.append(make_source(
                "stack_v2_python",
                0.10,
                "bigcode/the-stack-v2-dedup",
                format_code,
                config=args.stack_v2_config,
            ))

    sources.append(make_source(
        "openwebmath",
        0.10,
        "open-web-math/open-web-math",
        format_text,
    ))

    sources = [source for source in sources if source is not None]
    if not sources:
        raise RuntimeError("no annealing sources loaded")

    normalize_sources(sources)
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="anneal_train.txt")
    parser.add_argument("--num-examples", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fineweb-config", type=str, default="sample-10BT")
    parser.add_argument("--min-fineweb-score", type=float, default=None)
    parser.add_argument("--cosmopedia-dataset", type=str, default="HuggingFaceTB/cosmopedia-v2")
    parser.add_argument("--cosmopedia-config", type=str, default=None)
    parser.add_argument("--code-source", type=str, default="codeparrot", choices=["codeparrot", "starcoderdata", "stack_v2", "none"])
    parser.add_argument("--stack-v2-config", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    sources = build_sources(args)

    names = [source["name"] for source in sources]
    weights = [source["weight"] for source in sources]
    by_name = {source["name"]: source for source in sources}
    counts = {name: 0 for name in names}
    resets = {name: 0 for name in names}

    print("annealing sources:")
    for source in sources:
        print(f"  {source['name']}: weight={source['weight']:.4f}")

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_examples):
            name = random.choices(names, weights=weights, k=1)[0]
            source = by_name[name]
            text = next_formatted(source)

            f.write(text)
            f.write("\n")

            counts[name] += 1
            resets[name] = source["resets"]

            if (i + 1) % 10_000 == 0:
                print(f"wrote {i + 1} examples: {counts}; resets: {resets}")

    print(f"done: wrote {args.num_examples} examples to {args.output}")
    print(counts)
    print(f"resets: {resets}")
    sys.stdout.flush()
    sys.stderr.flush()

    # Avoid occasional HF streaming/PyArrow finalization crashes after writing.
    os._exit(0)


if __name__ == "__main__":
    main()
