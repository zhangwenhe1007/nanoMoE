import argparse
import random

from datasets import load_dataset


END = "<|endoftext|>"


def clean(text):
    return " ".join(str(text).split())


def load_stream(dataset_name, *, config=None, split="train"):
    if config is None:
        return iter(load_dataset(dataset_name, split=split, streaming=True))
    return iter(load_dataset(dataset_name, config, split=split, streaming=True))


def make_source(name, weight, dataset_name, formatter, *, config=None, split="train"):
    return {
        "name": name,
        "weight": weight,
        "dataset_name": dataset_name,
        "config": config,
        "split": split,
        "iterator": load_stream(dataset_name, config=config, split=split),
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


def format_plain_text(ex):
    text = ex.get("text") or ex.get("content") or ""
    text = clean(text)
    if not text:
        return None
    return text + END


def format_tinystory(ex):
    text = clean(ex["text"])
    if not text:
        return None
    return text + END


def format_gsm8k(ex):
    return (
        "Question: " + clean(ex["question"]) + "\n"
        "Answer: " + clean(ex["answer"]) + END
    )


def format_trivia(ex):
    ans = ex["answer"]
    if isinstance(ans, dict):
        answer = ans.get("value") or ans.get("normalized_value")
        if answer is None and ans.get("aliases"):
            answer = ans["aliases"][0]
    else:
        answer = str(ans)

    if not answer:
        return None

    return (
        "Question: " + clean(ex["question"]) + "\n"
        "Answer: " + clean(answer) + END
    )


def format_openbookqa(ex):
    choices = ex.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    answer_key = ex.get("answerKey")

    answer = None
    for label, text in zip(labels, texts):
        if label == answer_key:
            answer = text
            break

    if answer is None:
        answer = answer_key

    return (
        "Question: " + clean(ex["question_stem"]) + "\n"
        "Answer: " + clean(answer) + END
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="medium_train.txt")
    parser.add_argument("--num-examples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cosmopedia-config", type=str, default="web_samples_v2")
    args = parser.parse_args()

    random.seed(args.seed)

    # TODO(nanoDSV4-data): move this mixture into a versioned manifest with
    # source weights, filters, license notes, and held-out eval contamination
    # checks. If the model is incoherent, first improve this mixture and train
    # longer before blaming the architecture.
    sources = [
        make_source("fineweb_edu", 0.60, "HuggingFaceFW/fineweb-edu", format_plain_text, config="sample-10BT"),
        make_source("cosmopedia", 0.20, "HuggingFaceTB/cosmopedia", format_plain_text, config=args.cosmopedia_config),
        make_source("openwebmath", 0.10, "open-web-math/open-web-math", format_plain_text),
        make_source("tinystories", 0.05, "roneneldan/TinyStories", format_tinystory),
        make_source("gsm8k", 0.025, "openai/gsm8k", format_gsm8k, config="main"),
        make_source("triviaqa", 0.015, "mandarjoshi/trivia_qa", format_trivia, config="rc.nocontext"),
        make_source("openbookqa", 0.01, "allenai/openbookqa", format_openbookqa, config="main"),
    ]

    names = [source["name"] for source in sources]
    weights = [source["weight"] for source in sources]
    by_name = {source["name"]: source for source in sources}
    counts = {name: 0 for name in names}
    resets = {name: 0 for name in names}

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


if __name__ == "__main__":
    main()
