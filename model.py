from dataclasses import dataclass
import torch
import torch.nn as nn
import math
from config import normalize_model_config
from transformer import Transformer, RMSNorm


@dataclass
class GPTOutput:
    logits: torch.Tensor
    aux_loss: torch.Tensor | None = None
    stats: dict | None = None


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        d_model,
        n_heads,
        n_layers,
        n_kv_heads=None,
        mode="mha",
        pos_encoding="rope",
        model_config=None,
    ):
        super().__init__()

        config = normalize_model_config(model_config or {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "mode": mode,
            "attention_impl": mode,
            "pos_encoding": pos_encoding,
        })
        self.config = config
        self.block_size = config["block_size"]
        self.tokenizer_vocab_size = config.get("tokenizer_vocab_size") or config["vocab_size"]
        block_size = config["block_size"]
        vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        n_kv_heads = config["n_kv_heads"]
        mode = config["mode"]
        pos_encoding = config["pos_encoding"]
        self.last_aux_loss = None
        self.last_stats = {}

        # TODO(nanoDSV4-MLA): model_config should become the single source of
        # truth for MLA dims, RoPE/NoPE split, latent KV cache size, and whether
        # inference uses absorbed KV projections.
        #
        # TODO(nanoDSV4-MoE): wire DeepSeekMoE config into every block: shared
        # experts, fine-grained routed experts, top-k routing, router noise,
        # global/sequence balance losses, z-loss, and aux-loss-free balancing.
        #
        # TODO(nanoDSV4-sparse-attn): route long-context layers through dense,
        # sliding-window, CSA, HCA, or alternating CSA/HCA attention according to
        # config instead of hard-coding one attention class.
        #
        # TODO(nanoDSV4-posttrain): preserve architecture/post-training metadata
        # in checkpoints so SFT, reasoning distillation, and GRPO can resume the
        # exact same model family.
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        use_rope = False

        if (pos_encoding == "sinusoidal"):
            self.positional_embedding = SinusoidalPositionalEncoding(block_size, d_model)
        elif (pos_encoding == "learned"):
            self.positional_embedding = nn.Embedding(block_size, d_model)
        elif (pos_encoding == "rope"):
            self.positional_embedding = None
            use_rope = True
        else:
            raise ValueError(f"unknown positional encoding: {pos_encoding}")

        self.transformer = Transformer(
            d_model,
            n_heads,
            n_layers,
            n_kv_heads,
            mode,
            use_rope,
            config=config,
        )
        #self.ln_f = nn.LayerNorm(d_model)
        self.ln_f = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        #weight tying
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, idx, targets=None, return_output=False):
        """
        idx:     (B, T)
        targets: (B, T), optional
        """

        B, T = idx.shape

        assert T <= self.block_size

        tok_emb = self.token_embedding(idx) # (B, T, d_model)

        if (self.positional_embedding is not None):
            pos = torch.arange(T, device=idx.device)
            pos_emb = self.positional_embedding(pos)  # (T, d_model)

            x = tok_emb + pos_emb   # (B, T, d_model)
        else:
            x = tok_emb

        # TODO(nanoDSV4-output): once MLA/MoE/sparse attention are live, return a
        # GPTOutput by default from training paths and keep generation on logits.
        # Include lm_loss, aux_loss, router stats, attention compression stats,
        # active parameters, and KV-cache bytes in the structured output.
        x = self.transformer(x)
        self.last_aux_loss = self.transformer.last_aux_loss
        self.last_stats = self.transformer.last_stats
        x = self.ln_f(x)

        logits = self.lm_head(x)
        if return_output:
            return GPTOutput(logits=logits, aux_loss=self.last_aux_loss, stats=self.last_stats)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50, top_p=None, eos_token_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)    # (B, T, vocab_size)
            if isinstance(logits, GPTOutput):
                logits = logits.logits
            logits = logits[:, -1, :]  # (B, vocab_size)
            if self.tokenizer_vocab_size < logits.shape[-1]:
                logits[:, self.tokenizer_vocab_size:] = -float("inf")

            if top_k is not None:
                top_k = min(top_k, logits.shape[-1])
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_value, torch.full_like(logits, -float("inf")), logits)

            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                remove = cumulative_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False

                sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat([idx, next_idx], dim=1)    # (B, T+1)

            if eos_token_id is not None and torch.all(next_idx == eos_token_id):
                break
        return idx

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, block_size, d_model):
        super().__init__()

        pe = torch.zeros(block_size, d_model)

        pos = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe)

    def forward(self, T):
        return self.pe[:T]
