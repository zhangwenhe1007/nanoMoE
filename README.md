# nanoDSV4

Educational nano DeepSeek-V4-style language-model project forked from a
completed nanoLLaMA/nanoGPT learning stack.

The current runnable model is still the dense baseline: a compact decoder with
RMSNorm, RoPE/learned/sinusoidal positions, SwiGLU, MHA/GQA, GPT-2 BPE
tokenization, DDP pretraining, annealing, SFT, and interactive inference.

The goal is to grow this into a small but inspectable DeepSeek-style system:
MLA, DeepSeekMoE, sparse long-context attention, CSA/HCA/mHC options, and
reasoning post-training. This is an educational target, not a claim that a nano
model can reproduce DeepSeek-V4 or R1 performance.

## Quality Target

The final model should be noticeably better than the earlier nanoLLaMA:

1. produce full English sentences consistently
2. answer basic factual questions that appear often in training data
3. show simple reasoning on arithmetic, short QA, and instruction-following
4. remain debuggable: router stats, active parameters, KV-cache size, and evals

Architecture helps efficiency, but coherence mostly comes from enough clean
tokens, enough active parameters, stable optimization, and post-training data.

## Implementation Ladder

1. **Dense Baseline Hardening**
   - Keep the current dense GQA/SwiGLU path as the reference.
   - Add parameter counts, active parameter counts, tokens/sec, loss curves, and
     small deterministic evals.
   - Train until the baseline forms coherent sentences before adding complexity.

2. **MLA**
   - Implement Multi-head Latent Attention as a third attention implementation.
   - Add RoPE/NoPE split, compressed latent KV cache, cache byte accounting, and
     an inference path with absorbed projections after correctness tests pass.

3. **DeepSeekMoE**
   - Add shared experts plus fine-grained routed experts.
   - Add top-k routing, router temperature/noise, capacity/drop policy, z-loss,
     global balance loss, sequence balance loss, and optional aux-loss-free
     balancing.
   - Log expert utilization and specialization by layer, token position, and
     dataset source.

4. **Sparse Attention**
   - Start with sliding-window attention.
   - Add compressed block summaries and top-k block retrieval.
   - Add long-context diagnostics before optimizing kernels.

5. **CSA/HCA/mHC**
   - CSA: compressed sparse attention over selected blocks.
   - HCA: heavier compressed global context path.
   - mHC: optional multi-head hyper-connections for depth/stability.
   - Keep attention residuals as a separate comparison flag.

6. **Reasoning Post-Training**
   - SFT on clean instruction and reasoning traces.
   - GRPO on rule-checkable tasks: arithmetic, exact-answer QA, GSM8K-style
     final answer extraction, and small code/unit-test tasks.
   - Distill successful reasoning samples back into SFT data.

## Training Guidance

For a nano model, better performance is more likely to come from this order:

1. cleaner and larger pretraining data
2. enough training tokens, not just more parameters
3. a slightly bigger dense or active model
4. stable SFT data with masked response loss
5. small verifiable reasoning RL
6. MLA/MoE/CSA/HCA efficiency work

DeepSeek-style MoE can increase total capacity at fixed active compute, but at
small scale it can also slow training and starve experts of data. Do not assume
sparse equals better until the dense baseline is strong and the router is stable.

## Code Map

- `config.py`: architecture config fields and dense fallback defaults
- `model.py`: GPT wrapper, checkpoint-compatible config, structured output hook
- `transformer.py`: attention/MLP builders and TODOs for MLA, DeepSeekMoE,
  sparse attention, CSA/HCA, mHC, and attention residuals
- `train.py`: pretraining CLI, checkpoint config, future architecture losses
- `finetune.py`: response-masked SFT and future architecture loss handling
- `inference.py`: checkpoint reconstruction and future debug stats
- `*_dataset.py`: pretraining, annealing, and SFT data builders

## Immediate Next Milestone

Train the stronger dense baseline before adding MLA/MoE:

```bash
pip install -r requirements.txt
# Install the cluster's recommended PyTorch/CUDA build separately.
```

```bash
python clean_pretrain_dataset.py \
  --output-dir clean_12b_gpt2 \
  --target-tokens 12000000000 \
  --shard-tokens 250000000 \
  --tokenizer-name gpt2

python train.py \
  --data-path clean_12b_gpt2 \
  --model-preset dense_360m \
  --tokenizer-name gpt2 \
  --target-tokens 12000000000 \
  --batch-size 8 \
  --grad-accum-steps 64 \
  --dtype bf16 \
  --compile
```

The `dense_360m` preset is approximately 362M parameters with GPT-2 BPE:

- `d_model=1024`
- `n_layers=28`
- `n_heads=16`
- `n_kv_heads=4`
- `block_size=1024`
- `GQA + RoPE + RMSNorm + SwiGLU`

Using `--tokenizer-name cl100k_base` is supported, but it increases the
embedding table and moves this preset closer to roughly 410M parameters. Use it
as an experiment, not as a free upgrade.

For `cl100k_base`, build a separate corpus because token ids no longer fit in
`uint16` and the shards will be `uint32`:

```bash
python clean_pretrain_dataset.py \
  --output-dir clean_12b_cl100k \
  --target-tokens 12000000000 \
  --shard-tokens 250000000 \
  --tokenizer-name cl100k_base

python train.py \
  --data-path clean_12b_cl100k \
  --model-preset dense_360m \
  --tokenizer-name cl100k_base \
  --target-tokens 12000000000 \
  --batch-size 32 \
  --grad-accum-steps 8 \
  --dtype bf16 \
  --compile \
  --num-workers 8
```

The dataset builder writes many shard files plus `manifest.json`. At
`250M` tokens per shard, `cl100k_base` uses roughly 1 GB per shard, so 12B tokens
is about 48 shards. GPT-2 BPE shards are roughly half that size because they use
`uint16`.

For a 2x B200 run, start with:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --data-path clean_12b_cl100k \
  --model-preset dense_360m \
  --tokenizer-name cl100k_base \
  --target-tokens 12000000000 \
  --batch-size 32 \
  --grad-accum-steps 8 \
  --dtype bf16 \
  --compile \
  --num-workers 8
```

If memory allows, prefer larger per-GPU batches and lower accumulation, such as
`--batch-size 64 --grad-accum-steps 4`. The tokens per optimizer step stay the
same, but fewer microsteps usually improves GPU utilization. Sharded pretraining
data is not shuffled by default because the corpus builder already mixes
documents; this keeps memory-mapped reads mostly sequential.

After this baseline forms coherent English, compare:

1. dense GQA/SwiGLU
2. MLA/SwiGLU
3. MLA/DeepSeekMoE
4. MLA/DeepSeekMoE plus sparse attention
