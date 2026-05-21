import torch
import torch.nn as nn
import numpy as np


#Attention helpers
def attention(q, k, v, mask=None):
    """
    q: (B, H, T, d_head)
    k: (B, H, T, d_head)
    v: (B, H, T, d_head)
    """

    scores = (q @ k.transpose(-2, -1))/q.shape[-1] ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    attn_weights = nn.functional.softmax(scores, dim=-1)
    return attn_weights @ v

def combine_heads(q, k, v, mask=None):
    out = attention(q, k, v, mask)   # (B, H, T, d_h)
    B, H, T, d_h = out.shape

    out = out.transpose(1,2).contiguous()
    out = out.reshape(B, T, H * d_h)
    return out

#RoPE
def apply_rope(x):
    # x: (B, H, T, d_head)
    B, H, T, d_head = x.shape

    assert d_head % 2 == 0
    n_pairs = d_head//2
    
    x = x.view(B, H, T, n_pairs, 2)

    x_even = x[..., 0]
    x_odd = x[..., 1]

    pos = torch.arange(T, device=x.device)
    i = torch.arange(n_pairs, device=x.device)

    theta = 10000 ** (-2 * i / d_head)
    angles = pos[:, None] * theta[None, :]

    cos = torch.cos(angles)[None, None, :, :]
    sin = torch.sin(angles)[None, None, :, :]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)

    return x_rot.view(B, H, T, d_head)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.scale * x_norm


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, n_kv_heads=None, mode="mha", use_rope=True):
        super().__init__()

        # TODO(MoE): thread MoE options through this constructor and pass layer_id
        # into each block so we can start with MoE every other layer, then compare
        # dense-only, all-MoE, and mixed dense/MoE stacks.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, use_rope, n_kv_heads, mode) for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        # TODO(MoE): aggregate per-layer aux loss, z-loss, router entropy, expert
        # token counts, and overflow/drop stats here for training logs.
        for block in self.blocks:
            x = block(x, mask)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_rope, n_kv_heads=None, mode="mha"):
        super().__init__()
        #layernorm, mha, residual add, layernorm, mlp, residual add. x6
        #self.ln1 = nn.LayerNorm(d_model)
        self.ln1 = RMSNorm(d_model)

        if (mode == "mha"):
            self.attn = MultiHeadAttention(d_model, n_heads, use_rope)
        elif (mode == "gqa" and n_kv_heads is not None):
            self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, use_rope)
        else:
            raise ValueError(f"unknown attention mode: {mode}")

        #self.ln2 = nn.LayerNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        # TODO(MoE): replace this dense SwiGLU with a selectable MLP implementation:
        # dense SwiGLU, Switch-style top-1 MoE, Mixtral-style top-2 MoE, and later
        # shared-plus-routed experts inspired by DeepSeekMoE.
        self.mlp = SwiGLU(d_model)
        
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # TODO(MoE): keep this class as the single expert implementation so dense
        # MLP and sparse experts use identical feed-forward math.
        hidden_dim = int((8/3) * d_model)
        hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)

        self.gate_proj = nn.Linear(d_model, hidden_dim)
        self.up_proj = nn.Linear(d_model, hidden_dim)
        self.silu = nn.SiLU()
        self.down_proj = nn.Linear(hidden_dim, d_model)
    
    def forward(self, x):
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))


# TODO(MoE): add a small Router module:
# - project token states from d_model -> num_experts
# - compute router logits/probabilities in fp32
# - support noisy routing during training
# - return top_k expert ids, routing weights, and router diagnostics.
#
# TODO(MoE): add SwitchMoE / Top1MoE first:
# - flatten (B, T, C) tokens
# - send each token to its top-1 expert
# - implement capacity_factor and either token dropping or residual fallback
# - compute load-balancing loss and router z-loss.
#
# TODO(MoE): add MixtralMoE / Top2MoE second:
# - route each token to two experts
# - combine expert outputs with normalized router weights
# - log expert utilization so collapse is obvious during small runs.
#
# TODO(MoE): add frontier-style variants after the basic MoE works:
# - dropless grouped dispatch
# - shared expert plus routed experts
# - fine-grained experts with more smaller experts selected per token
# - expert-parallel dispatch across GPUs.

class GeLU_MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(4 * d_model, d_model)
    
    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))
    

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, use_rope):
        super().__init__()

        assert d_model % n_q_heads == 0
        assert n_q_heads % n_kv_heads == 0
        self.d_head = d_model // n_q_heads
        self.group_size = n_q_heads // n_kv_heads
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, n_q_heads * self.d_head)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        B, T, _ = x.shape

        q = q.view(B, T, self.n_q_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q)
            k = apply_rope(k)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        out = combine_heads(q, k, v, mask)
        return self.out_proj(out)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_rope):
        super().__init__()
        self.n_heads = n_heads
        self.use_rope = use_rope
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        B, T, d_model = q.shape

        H = self.n_heads
        
        assert d_model % H == 0
        d_head = d_model // H

        q = q.view(B, T, H, d_head).transpose(1, 2)
        k = k.view(B, T, H, d_head).transpose(1, 2)
        v = v.view(B, T, H, d_head).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q)
            k = apply_rope(k)

        out = combine_heads(q, k, v, mask)
        return self.out_proj(out)

