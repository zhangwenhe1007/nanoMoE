import argparse
import json
import os
import random
import re
import sys
from array import array
from collections import defaultdict

import tiktoken
from datasets import load_dataset


WORD_RE = re.compile(r"[A-Za-z0-9]+")
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    return SPACE_RE.sub(" ", str(text or "")).strip()


def words(text):
    return WORD_RE.findall(text)


def bad_text(text, min_words, max_chars):
    if len(text) > max_chars:
        return True

    ws = words(text)
    if len(ws) < min_words:
        return True

    unique_ratio = len(set(w.lower() for w in ws)) / max(1, len(ws))
    if unique_ratio < 0.22:
        return True

    lowered = text.lower()
    banned = [
        "lorem ipsum",
        "javascript is disabled",
        "enable cookies",
        "all rights reserved",
        "privacy policy",
        "terms of service",
    ]
    return any(marker in lowered for marker in banned)


def load_stream(dataset_name, *, config=None, split="train"):
    if config is None:
        return iter(load_dataset(dataset_name, split=split, streaming=True))
    return iter(load_dataset(dataset_name, config, split=split, streaming=True))


def make_source(name, weight, dataset_name, text_fn, *, config=None, split="train"):
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
        "text_fn": text_fn,
        "resets": 0,
    }


def reset_source(source):
    source["iterator"] = load_stream(
        source["dataset_name"],
        config=source["config"],
        split=source["split"],
    )
    source["resets"] += 1


def text_field(*keys):
    def read(ex):
        for key in keys:
            value = ex.get(key)
            if value:
                return value
        return ""

    return read


def openwebmath_text(ex):
    return ex.get("text", "")


def tinystory_text(ex):
    return ex.get("text", "")


def wikipedia_text(ex):
    return ex.get("text", "")


def code_text(ex):
    return ex.get("code") or ex.get("content") or ex.get("text") or ""


def build_sources(args):
    # This mixture is intentionally conservative: high-educational-score web
    # text for language and facts, DCLM for broad clean web coverage, synthetic
    # textbook-style data for compact concept exposure, math/code for reasoning,
    # and a small TinyStories slice for short-form fluency.
    specs = [
        ("fineweb_edu", 0.45, "HuggingFaceFW/fineweb-edu", text_field("text"), "sample-10BT", "train"),
        ("dclm_baseline", 0.20, "mlfoundations/dclm-baseline-1.0", text_field("text"), None, "train"),
        ("cosmopedia_v2", 0.15, "HuggingFaceTB/smollm-corpus", text_field("text"), "cosmopedia-v2", "train"),
        ("openwebmath", 0.10, "open-web-math/open-web-math", openwebmath_text, None, "train"),
        ("wikipedia", 0.05, "wikimedia/wikipedia", wikipedia_text, "20231101.en", "train"),
        ("tinystories", 0.03, "roneneldan/TinyStories", tinystory_text, None, "train"),
        ("codeparrot_python", 0.02, "codeparrot/github-code-clean", code_text, None, "train"),
    ]

    sources = []
    for name, weight, dataset_name, text_fn, config, split in specs:
        source = make_source(name, weight, dataset_name, text_fn, config=config, split=split)
        if source is not None:
            sources.append(source)

    if not sources:
        raise RuntimeError("no dataset sources loaded")

    total = sum(source["weight"] for source in sources)
    for source in sources:
        source["weight"] /= total
    return sources


def next_clean_text(source, args, max_attempts=1000):
    for _ in range(max_attempts):
        try:
            ex = next(source["iterator"])
        except StopIteration:
            reset_source(source)
            ex = next(source["iterator"])

        text = clean_text(source["text_fn"](ex))
        if bad_text(text, args.min_words, args.max_chars):
            continue
        return text

    raise RuntimeError(f"source {source['name']} produced no clean text after {max_attempts} attempts")


def write_manifest(output_dir, metadata):
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def open_shard(output_dir, shard_idx):
    name = f"shard_{shard_idx:06d}.bin"
    path = os.path.join(output_dir, name)
    return name, open(path, "wb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="clean_12b_gpt2")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2", choices=["gpt2", "cl100k_base", "o200k_base"])
    parser.add_argument("--target-tokens", type=int, default=12_000_000_000)
    parser.add_argument("--shard-tokens", type=int, default=250_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--progress-interval", type=int, default=100_000_000)
    args = parser.parse_args()

    random.seed(args.seed)
    enc = tiktoken.get_encoding(args.tokenizer_name)
    dtype = "uint16" if enc.n_vocab <= 65535 else "uint32"
    array_typecode = "H" if dtype == "uint16" else "I"
    end_token = enc.eot_token

    sources = build_sources(args)
    names = [source["name"] for source in sources]
    weights = [source["weight"] for source in sources]
    by_name = {source["name"]: source for source in sources}
    counts = defaultdict(int)
    token_counts = defaultdict(int)
    resets = defaultdict(int)

    os.makedirs(args.output_dir, exist_ok=True)

    total_tokens = 0
    shard_idx = 0
    shard_tokens = 0
    shards = []
    shard_name, shard_file = open_shard(args.output_dir, shard_idx)
    next_progress = args.progress_interval
    metadata = {
        "format": "sharded_flat_token_bin",
        "dtype": dtype,
        "tokenizer": args.tokenizer_name,
        "target_tokens": args.target_tokens,
        "shard_tokens": args.shard_tokens,
        "seed": args.seed,
        "actual_tokens": 0,
        "shards": [],
        "sources": [
            {
                "name": source["name"],
                "weight": source["weight"],
                "dataset_name": source["dataset_name"],
                "config": source["config"],
                "split": source["split"],
            }
            for source in sources
        ],
    }
    write_manifest(args.output_dir, metadata)

    try:
        while total_tokens < args.target_tokens:
            name = random.choices(names, weights=weights, k=1)[0]
            source = by_name[name]
            text = next_clean_text(source, args)
            ids = enc.encode(text) + [end_token]
            arr = array(array_typecode, ids)

            if shard_tokens > 0 and shard_tokens + len(arr) > args.shard_tokens:
                shard_file.close()
                shards.append({"file": shard_name, "tokens": shard_tokens})
                metadata["actual_tokens"] = total_tokens
                metadata["shards"] = shards
                write_manifest(args.output_dir, metadata)
                shard_idx += 1
                shard_tokens = 0
                shard_name, shard_file = open_shard(args.output_dir, shard_idx)

            arr.tofile(shard_file)

            n_tokens = len(arr)
            total_tokens += n_tokens
            shard_tokens += n_tokens
            counts[name] += 1
            token_counts[name] += n_tokens
            resets[name] = source["resets"]

            if args.progress_interval > 0 and total_tokens >= next_progress:
                print(f"tokens: {total_tokens:,}")
                print(f"documents: {dict(counts)}")
                print(f"source_tokens: {dict(token_counts)}")
                print(f"resets: {dict(resets)}")
                sys.stdout.flush()
                next_progress += args.progress_interval
    finally:
        shard_file.close()

    if shard_tokens > 0:
        shards.append({"file": shard_name, "tokens": shard_tokens})

    metadata.update({
        "actual_tokens": total_tokens,
        "shards": shards,
        "documents": dict(counts),
        "source_tokens": dict(token_counts),
        "resets": dict(resets),
    })
    write_manifest(args.output_dir, metadata)
    print(f"done: wrote {total_tokens:,} tokens across {len(shards)} shards to {args.output_dir}")


if __name__ == "__main__":
    main()
