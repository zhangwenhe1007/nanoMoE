import os
import json
from bisect import bisect_right

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import encode


class TextDataset(Dataset):
    def __init__(
        self,
        path,
        block_size,
        split="train",
        train_frac=0.9,
        stride=None,
        cache_tokens=True,
        tokenizer_name="gpt2",
    ):
        self.block_size = block_size
        self.stride = block_size if stride is None else stride
        self.sharded = False

        # TODO(nanoDSV4-data): cache metadata next to token tensors: tokenizer
        # name, document count, token count, source mixture, dedup/filter version,
        # and train/val split seed. Better coherence and basic factuality will
        # depend more on data quality and token budget than on MLA/MoE alone.
        if os.path.isdir(path):
            self._load_sharded_bin(path, split, train_frac)
            return

        if path.endswith(".bin"):
            metadata_path = path + ".json"
            dtype = "uint16"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                dtype = metadata.get("dtype", dtype)

            data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
            n = int(train_frac * len(data))

            if split == "train":
                self.data = data[:n]
            elif split == "val":
                self.data = data[n:]
            else:
                raise ValueError("split must be 'train' or 'val'")
            return

        cache_path = path + f".{tokenizer_name}.tokens.pt"

        if cache_tokens and os.path.exists(cache_path):
            print(f"loading token cache from {cache_path}")
            data = torch.load(cache_path, map_location="cpu")
        else:
            print(f"tokenizing {path}")
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            ids = encode(text, tokenizer_name=tokenizer_name)
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

    def _load_sharded_bin(self, path, split, train_frac):
        manifest_path = os.path.join(path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"missing sharded dataset manifest: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        dtype = np.dtype(manifest.get("dtype", "uint16"))
        total_tokens = int(manifest["actual_tokens"])
        split_token = int(train_frac * total_tokens)

        if split == "train":
            split_start, split_end = 0, split_token
        elif split == "val":
            split_start, split_end = split_token, total_tokens
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.sharded = True
        self.segments = []
        self.cumulative_examples = []
        example_count = 0
        token_cursor = 0

        for shard in manifest["shards"]:
            shard_tokens = int(shard["tokens"])
            shard_start = token_cursor
            shard_end = token_cursor + shard_tokens
            token_cursor = shard_end

            seg_start = max(split_start, shard_start)
            seg_end = min(split_end, shard_end)
            if seg_end - seg_start <= self.block_size:
                continue

            shard_path = os.path.join(path, shard["file"])
            data = np.memmap(shard_path, dtype=dtype, mode="r")
            local_start = seg_start - shard_start
            local_end = seg_end - shard_start
            seg_len = local_end - local_start
            n_examples = max(0, (seg_len - self.block_size - 1) // self.stride + 1)
            if n_examples == 0:
                continue

            self.segments.append({
                "data": data,
                "start": local_start,
                "examples": n_examples,
            })
            example_count += n_examples
            self.cumulative_examples.append(example_count)

    def __len__(self):
        if self.sharded:
            return self.cumulative_examples[-1] if self.cumulative_examples else 0
        return max(0, (len(self.data) - self.block_size - 1) // self.stride + 1)

    def __getitem__(self, idx):
        if self.sharded:
            segment_idx = bisect_right(self.cumulative_examples, idx)
            previous = 0 if segment_idx == 0 else self.cumulative_examples[segment_idx - 1]
            segment = self.segments[segment_idx]
            local_idx = idx - previous
            start = segment["start"] + local_idx * self.stride
            chunk = segment["data"][start : start + self.block_size + 1]
        else:
            start = idx * self.stride
            chunk = self.data[start : start + self.block_size + 1]

        if not isinstance(chunk, torch.Tensor):
            chunk = torch.from_numpy(np.asarray(chunk, dtype=np.int64))

        x = chunk[:-1]
        y = chunk[1:]

        return x, y
