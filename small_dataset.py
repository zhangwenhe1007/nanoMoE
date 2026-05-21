from datasets import load_dataset, concatenate_datasets
import random

END = "<|endoftext|>"

tiny = load_dataset("roneneldan/TinyStories", split="train")
gsm8k = load_dataset("openai/gsm8k", "main", split="train")
trivia = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")

def format_tinystory(ex):
    return {"text": ex["text"] + END}

def format_gsm8k(ex):
    return {
        "text": (
            "Question: " + ex["question"] + "\n"
            "Answer: " + ex["answer"] + END
        )
    }

def format_trivia(ex):
    ans = ex["answer"]

    if isinstance(ans, dict):
        answer = ans.get("value") or ans.get("normalized_value") or ans["aliases"][0]
    else:
        answer = str(ans)

    return {
        "text": (
            "Question: " + ex["question"] + "\n"
            "Answer: " + answer + END
        )
    }

tiny = tiny.shuffle(seed=1337).select(range(80_000)).map(format_tinystory, remove_columns=tiny.column_names)
gsm8k = gsm8k.shuffle(seed=1337).select(range(len(gsm8k))).map(format_gsm8k, remove_columns=gsm8k.column_names)
trivia = trivia.shuffle(seed=1337).select(range(10_000)).map(format_trivia, remove_columns=trivia.column_names)

mixed = concatenate_datasets([tiny, gsm8k, trivia])
mixed = mixed.shuffle(seed=1337)

with open("./mixed_train.txt", "w", encoding="utf-8") as f:
    for ex in mixed:
        f.write(ex["text"])
        f.write("\n")