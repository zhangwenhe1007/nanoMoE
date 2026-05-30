DEFAULT_MODEL_CONFIG = {
    "architecture_target": "nano_dsv4",
    "tokenizer_name": "gpt2",
    "vocab_size": None,
    "block_size": 512,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 8,
    "n_kv_heads": 2,
    "mode": "gqa",
    "pos_encoding": "rope",
    # Stage 1: MLA. Current values keep the existing dense GQA path active.
    "attention_impl": "gqa",
    "mla_latent_dim": 0,
    "mla_q_lora_rank": 0,
    "mla_kv_lora_rank": 0,
    "mla_rope_head_dim": 0,
    "mla_nope_head_dim": 0,
    "mla_absorb_kv": False,
    # Stage 2: DeepSeekMoE. Current values keep dense SwiGLU blocks active.
    "mlp_impl": "dense",
    "num_experts": 0,
    "num_shared_experts": 0,
    "num_routed_experts": 0,
    "routed_experts_per_token": 0,
    "moe_layer_frequency": 0,
    "expert_hidden_mult": 8 / 3,
    "capacity_factor": 1.0,
    "drop_overflow_tokens": False,
    "router_noise_std": 0.0,
    "router_temperature": 1.0,
    "router_z_loss_weight": 0.0,
    "global_balance_loss_weight": 0.0,
    "sequence_balance_loss_weight": 0.0,
    "aux_loss_free_balance": False,
    # Stage 3: sparse attention, then CSA/HCA/mHC.
    "sparse_attention_impl": "none",
    "sliding_window": 0,
    "compressed_block_size": 0,
    "csa_top_blocks": 0,
    "hca_pool_factor": 0,
    "hca_num_registers": 0,
    "use_mhc": False,
    "use_attention_residual": False,
    # Stage 4: training/post-training diagnostics.
    "return_aux_loss": False,
    "return_router_stats": False,
}


MODEL_PRESETS = {
    "custom": {},
    "dense_115m": {
        "block_size": 512,
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "n_kv_heads": 4,
        "attention_impl": "gqa",
        "mode": "gqa",
        "pos_encoding": "rope",
        "mlp_impl": "dense",
    },
    "dense_230m": {
        "block_size": 1024,
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 16,
        "n_kv_heads": 4,
        "attention_impl": "gqa",
        "mode": "gqa",
        "pos_encoding": "rope",
        "mlp_impl": "dense",
    },
    "dense_360m": {
        "block_size": 1024,
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 28,
        "n_kv_heads": 4,
        "attention_impl": "gqa",
        "mode": "gqa",
        "pos_encoding": "rope",
        "mlp_impl": "dense",
    },
}


def normalize_model_config(config):
    normalized = dict(DEFAULT_MODEL_CONFIG)
    normalized.update(config)

    # Backward compatibility with existing checkpoints and CLI names.
    normalized["attention_impl"] = normalized.get("attention_impl") or normalized.get("mode", "gqa")
    normalized["mode"] = normalized.get("mode") or normalized["attention_impl"]
    normalized["pos_encoding"] = normalized.get("pos_encoding", "rope")
    return normalized


def apply_model_preset(args):
    preset = MODEL_PRESETS[args.model_preset]
    for key, value in preset.items():
        arg_name = key
        if key == "mode":
            arg_name = "attention_mode"
        elif key == "pos_encoding":
            arg_name = "encoding_mode"

        current_value = getattr(args, arg_name)
        parser_default = DEFAULT_MODEL_CONFIG.get(key)
        if key == "mode":
            parser_default = "gqa"
        elif key == "pos_encoding":
            parser_default = "rope"

        if current_value == parser_default:
            setattr(args, arg_name, value)
    return args


def build_model_config(args, vocab_size):
    config = dict(DEFAULT_MODEL_CONFIG)
    config.update({
        "vocab_size": vocab_size,
        "tokenizer_name": args.tokenizer_name,
        "block_size": args.block_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_kv_heads": args.n_kv_heads,
        "mode": args.attention_mode,
        "attention_impl": args.attention_impl,
        "pos_encoding": args.encoding_mode,
        "mlp_impl": args.mlp_impl,
        "num_experts": args.num_experts,
        "num_shared_experts": args.num_shared_experts,
        "num_routed_experts": args.num_routed_experts,
        "routed_experts_per_token": args.routed_experts_per_token,
        "moe_layer_frequency": args.moe_layer_frequency,
        "capacity_factor": args.capacity_factor,
        "router_z_loss_weight": args.router_z_loss_weight,
        "global_balance_loss_weight": args.global_balance_loss_weight,
        "sequence_balance_loss_weight": args.sequence_balance_loss_weight,
        "sparse_attention_impl": args.sparse_attention_impl,
        "use_mhc": args.use_mhc,
        "use_attention_residual": args.use_attention_residual,
    })
    return normalize_model_config(config)


def architecture_run_name(config):
    config = normalize_model_config(config)
    parts = [
        config["architecture_target"],
        f"attn-{config['attention_impl']}",
        f"mlp-{config['mlp_impl']}",
        f"d{config['d_model']}",
        f"l{config['n_layers']}",
    ]

    if config["attention_impl"] == "gqa":
        parts.append(f"q{config['n_heads']}_kv{config['n_kv_heads']}")
    if config["mlp_impl"] != "dense":
        parts.append(
            f"e{config['num_experts']}_s{config['num_shared_experts']}"
            f"_k{config['routed_experts_per_token']}"
        )
    if config["sparse_attention_impl"] != "none":
        parts.append(f"sparse-{config['sparse_attention_impl']}")
    if config["use_mhc"]:
        parts.append("mhc")
    if config["use_attention_residual"]:
        parts.append("attnres")
    parts.append(config["pos_encoding"])
    return "_".join(str(part) for part in parts)
