# nanoMoE

Educational MoE language-model project forked from the completed `nanogpt`
learning stack.

This folder now contains the runnable dense baseline from `nanogpt`: a compact
LLaMA-style decoder with RoPE, RMSNorm, SwiGLU, GQA/MQA support, GPT-2 BPE
tokenization, DDP training, annealing, SFT, and interactive inference.

The next implementation target is to replace the dense `SwiGLU` feed-forward
path with sparse MoE variants while keeping attention and training infrastructure
stable.

## MoE Implementation Plan

1. Start with dense `SwiGLU` unchanged as the baseline.
2. Add a Switch-style top-1 MoE MLP.
3. Add router aux loss, router z-loss, capacity factor, and utilization logging.
4. Upgrade to Mixtral-style top-2 routing.
5. Add dropless grouped dispatch.
6. Add shared-plus-routed experts and fine-grained experts.
7. Eventually explore expert parallelism across GPUs.

The code contains `TODO(MoE)` comments in the core touchpoints:

- `transformer.py`: expert MLP, router, dispatch, and per-layer stats
- `model.py`: model-level MoE config and output plumbing
- `train.py`: CLI/config/checkpoint/loss integration
- `finetune.py`: sparse checkpoint loading and SFT aux-loss behavior
- `inference.py`: sparse checkpoint reconstruction and optional router stats
