import tiktoken

def encode(text, special_tokens={"<|endoftext|>"}):
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(text, allowed_special=special_tokens)