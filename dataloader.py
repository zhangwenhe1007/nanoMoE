import os

import torch
from torch.utils.data import Dataset

from tokenizer import encode


class TextDataset(Dataset):
    def __init__(self, path, block_size, split="train", train_frac=0.9, stride=None, cache_tokens=True):
        self.block_size = block_size
        self.stride = block_size if stride is None else stride

        cache_path = path + ".tokens.pt"

        if cache_tokens and os.path.exists(cache_path):
            print(f"loading token cache from {cache_path}")
            data = torch.load(cache_path, map_location="cpu")
        else:
            print(f"tokenizing {path}")
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            ids = encode(text)
            data = torch.tensor(ids, dtype=torch.long)

            if cache_tokens:
                print(f"saving token cache to {cache_path}")
                torch.save(data, cache_path)

        n = int(train_frac * len(data))

        if split == "train":
            self.data = data[:n]
        elif split == "val":
            self.data = data[n:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return max(0, (len(self.data) - self.block_size - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.data[start : start + self.block_size + 1]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y
