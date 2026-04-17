# Qwen3-4B DFlash + LittleBit Pipeline

This guide wires together the full pipeline for:

1. Preparing PerfectBlend data
2. Regenerating it with the target model
3. Applying LittleBit QAT to `z-lab/Qwen3-4B-DFlash-b16`
4. Evaluating acceptance length on SD-style benchmarks
5. Comparing against `AngelSlim/Qwen3-4B_eagle3`

## 1. Prepare PerfectBlend

```bash
python scripts/prepare_data.py --dataset perfectblend
```

This writes the raw training file to `cache/dataset/perfectblend_train.jsonl`.

## 2. Regenerate with the Qwen3-4B Target

SpecForge’s official data-preparation guidance recommends regenerating responses with the target model to better match its output distribution and improve acceptance length:

- SpecForge docs: https://docs.sglang.ai/SpecForge/basic_usage/data_preparation.html

Example:

```bash
python scripts/regenerate_train_data.py \
    --model Qwen/Qwen3-4B \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/perfectblend_train.jsonl \
    --output-file-path ./cache/dataset/perfectblend_qwen3_4b_regen.jsonl
```

## 3. Train LittleBit-DFlash

The main training entrypoint is:

```bash
python scripts/train_littlebit_dflash.py --help
```

Remote 4xH100-style example:

```bash
bash examples/run_qwen3_4b_dflash_littlebit_qat.sh 4 0.5
```

Important defaults:

- Student init: pretrained DFlash checkpoint
- Quant init: LittleBit DualSVD
- QAT objective: final-logit KL + intermediate hidden-state MSE
- KD/MSE reduction: kept aligned with the LittleBit trainer style
- Primary comparison metric: acceptance length

## 4. Run Acceptance Smoke

```bash
bash examples/run_qwen3_4b_dflash_acceptance_smoke.sh
```

This runs a small HF-reference acceptance evaluation over `gsm8k`, `math500`, and `mtbench`.

## 5. Compare Eagle3 vs DFlash vs LittleBit-DFlash

```bash
bash examples/run_qwen3_4b_sd_compare.sh /path/to/littlebit/checkpoint
```

The comparison script writes:

- `benchmark_comparison.json`
- `benchmark_comparison.md`

in the requested output directory.

## Notes

- `eagle3` throughput comes from the existing SGLang benchmark runner.
- `dflash` and `littlebit-dflash` throughput come from a HF-reference evaluator and should not be treated as final kernel numbers.
- Until a dedicated LittleBit serving kernel is added, acceptance length is the metric to prioritize.
