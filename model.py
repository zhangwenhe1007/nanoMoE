import torch
import torch.nn as nn
import math
from transformer import Transformer, RMSNorm


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_heads, n_layers, n_kv_heads=None, mode="mha", pos_encoding="rope"):
        super().__init__()

        self.block_size = block_size
        # TODO(MoE): add model-level MoE config here, e.g. mlp_mode, num_experts,
        # top_k, capacity_factor, aux_loss_weight, router_z_loss_weight,
        # shared_experts, and moe_layer_frequency.
        
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

        # TODO(MoE): pass the MoE config into Transformer so individual blocks can
        # choose dense SwiGLU vs sparse expert MLPs without changing attention code.
        self.transformer = Transformer(d_model, n_heads, n_layers, n_kv_heads, mode, use_rope)
        #self.ln_f = nn.LayerNorm(d_model)
        self.ln_f = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        #weight tying
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, idx, targets=None):
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

        mask = torch.tril(torch.ones(T, T, device=idx.device))
        mask = mask.view(1, 1, T, T)

        # TODO(MoE): when MoE blocks are implemented, return or store router aux
        # losses/statistics from the transformer in addition to logits.
        x = self.transformer(x, mask)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50, top_p=None, eos_token_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits = self(idx_cond)    # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)

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
