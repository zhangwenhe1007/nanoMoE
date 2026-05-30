import torch
import torch.nn as nn


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
    def __init__(self, d_model, n_heads, n_layers, n_kv_heads=None, mode="mha", use_rope=True, config=None):
        super().__init__()
        self.config = config or {}
        self.last_aux_loss = None
        self.last_stats = {}

        # TODO(nanoDSV4-depth): assign each layer an attention/MLP role from the
        # config. Target ladder:
        # 1. dense/GQA baseline
        # 2. MLA in all attention layers
        # 3. MLA + DeepSeekMoE in selected MLP layers
        # 4. MLA + DeepSeekMoE + alternating dense/sliding/CSA/HCA attention
        # 5. optional mHC or attention residuals for deeper stacks.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                use_rope,
                n_kv_heads,
                mode,
                layer_id=layer_id,
                config=self.config,
            )
            for layer_id in range(n_layers)
        ])

    def forward(self, x, mask=None):
        aux_losses = []
        layer_stats = []
        for block in self.blocks:
            x = block(x, mask)

            if block.last_aux_loss is not None:
                aux_losses.append(block.last_aux_loss)
            if block.last_stats:
                layer_stats.append(block.last_stats)

        self.last_aux_loss = None
        if aux_losses:
            self.last_aux_loss = sum(aux_losses)
        self.last_stats = {"layers": layer_stats} if layer_stats else {}
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, use_rope, n_kv_heads=None, mode="mha", layer_id=0, config=None):
        super().__init__()
        self.layer_id = layer_id
        self.config = config or {}
        self.last_aux_loss = None
        self.last_stats = {}

        #layernorm, mha, residual add, layernorm, mlp, residual add. x6
        #self.ln1 = nn.LayerNorm(d_model)
        self.ln1 = RMSNorm(d_model)

        self.attn = build_attention(d_model, n_heads, n_kv_heads, use_rope, self.config)

        #self.ln2 = nn.LayerNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.mlp = build_mlp(d_model, layer_id, self.config)
        
    
    def forward(self, x, mask=None):
        self.last_aux_loss = None
        self.last_stats = {}

        x = x + self.attn(self.ln1(x), mask)
        mlp_out = self.mlp(self.ln2(x))
        if isinstance(mlp_out, tuple):
            mlp_out, aux_loss, stats = mlp_out
            self.last_aux_loss = aux_loss
            self.last_stats = stats or {}

        # TODO(nanoDSV4-mHC): when use_mhc=True, replace this simple residual
        # stream with multi-head hyper-connections that mix several residual
        # streams per layer. Keep use_attention_residual as a separate option so
        # Kimi-style attention residuals can be compared against mHC directly.
        x = x + mlp_out
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # TODO(nanoDSV4-MoE): keep this class as the expert implementation used
        # by dense MLP, shared experts, and routed experts so experiments compare
        # routing/topology rather than different feed-forward math.
        hidden_dim = int((8/3) * d_model)
        hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)

        self.gate_proj = nn.Linear(d_model, hidden_dim)
        self.up_proj = nn.Linear(d_model, hidden_dim)
        self.silu = nn.SiLU()
        self.down_proj = nn.Linear(hidden_dim, d_model)
    
    def forward(self, x):
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))


def build_attention(d_model, n_heads, n_kv_heads, use_rope, config):
    attention_impl = config.get("attention_impl", config.get("mode", "mha"))
    sparse_impl = config.get("sparse_attention_impl", "none")

    if sparse_impl in {"sliding", "csa", "hca", "alternating_csa_hca"}:
        raise NotImplementedError(
            "TODO(nanoDSV4-sparse-attn): implement sliding-window attention, "
            "compressed block selection for CSA, pooled global context for HCA, "
            "and layer schedules for alternating CSA/HCA."
        )
    if attention_impl == "mha":
        return MultiHeadAttention(d_model, n_heads, use_rope)
    if attention_impl == "gqa" and n_kv_heads is not None:
        return GroupedQueryAttention(d_model, n_heads, n_kv_heads, use_rope)
    if attention_impl == "mla":
        raise NotImplementedError(
            "TODO(nanoDSV4-MLA): implement MultiHeadLatentAttention with "
            "q/kv low-rank projections, RoPE/NoPE split, latent KV cache, "
            "and absorbed inference projections."
        )
    raise ValueError(f"unknown attention implementation: {attention_impl}")


def build_mlp(d_model, layer_id, config):
    mlp_impl = config.get("mlp_impl", "dense")
    frequency = config.get("moe_layer_frequency", 0)

    if mlp_impl == "dense":
        return SwiGLU(d_model)
    use_moe_layer = frequency <= 0 or (layer_id + 1) % frequency == 0
    if not use_moe_layer:
        return SwiGLU(d_model)
    if mlp_impl == "deepseek_moe":
        raise NotImplementedError(
            "TODO(nanoDSV4-MoE): implement DeepSeekMoE with shared experts, "
            "fine-grained routed experts, top-k routing, capacity/drop policy, "
            "global and sequence-level balance losses, router z-loss, optional "
            "aux-loss-free balancing, and per-layer utilization stats."
        )
    raise ValueError(f"unknown MLP implementation: {mlp_impl}")


# TODO(nanoDSV4-Router): add a Router module:
# - project token states from d_model -> num_routed_experts in fp32
# - support top-k routing with temperature and optional training noise
# - return expert ids, normalized route weights, router logits, and diagnostics
# - compute z-loss, global balance loss, sequence balance loss, and dropped-token
#   stats without hiding LM loss regressions.
#
# TODO(nanoDSV4-DeepSeekMoE): add DeepSeekMoE:
# - always run num_shared_experts shared SwiGLU experts
# - dispatch tokens to fine-grained routed SwiGLU experts
# - combine shared and routed outputs with stable scaling
# - start with simple per-expert loops, then add grouped/dropless dispatch
# - log expert specialization by dataset source, token position, and prompt type.
#
# TODO(nanoDSV4-MLA): add MultiHeadLatentAttention:
# - implement q down/up projection and compressed kv latent projection
# - split query/key heads into RoPE and NoPE dimensions
# - expose KV cache byte counts versus MHA/GQA
# - add an inference path with absorbed projections after correctness tests pass.
#
# TODO(nanoDSV4-CSA-HCA): add long-context attention variants:
# - local dense/sliding attention for nearby tokens
# - CSA: compressed block summaries plus top-k block retrieval
# - HCA: heavier pooling/register compression for global context
# - mHC: multi-head hyper-connections as a depth/optimization option
# - attention residuals: separate Kimi-style comparison flag.

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
