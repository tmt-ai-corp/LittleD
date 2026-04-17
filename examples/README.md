# Run SpecForge Examples

This folder contains the examples of running SpecForge on different models. The scripts can be invoked by the following command:

```bash
bash examples/<script-name>.sh [NUM_GPUS] [TP_SIZE]
```

We use the ShareGPT dataset for all the examples for now, but you can replace it with more robust datasets such as perfectblend, magpie-qwen2.5-pro-1m-v0.1, etc.

Additional DFlash + LittleBit examples:

- `run_qwen3_4b_dflash_littlebit_qat.sh`: 4xH100-style QAT training entrypoint for `z-lab/Qwen3-4B-DFlash-b16`
- `run_qwen3_4b_dflash_acceptance_smoke.sh`: small acceptance-length smoke run on `gsm8k/math500/mtbench`
- `run_qwen3_4b_sd_compare.sh`: compare `AngelSlim/Qwen3-4B_eagle3` vs `z-lab/Qwen3-4B-DFlash-b16` vs a LittleBit-DFlash checkpoint
