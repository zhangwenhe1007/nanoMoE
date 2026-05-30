import tiktoken


def get_tokenizer(tokenizer_name="gpt2"):
    return tiktoken.get_encoding(tokenizer_name)


def encode(text, special_tokens={"<|endoftext|>"}, tokenizer_name="gpt2"):
    enc = get_tokenizer(tokenizer_name)
    return enc.encode(text, allowed_special=special_tokens)
