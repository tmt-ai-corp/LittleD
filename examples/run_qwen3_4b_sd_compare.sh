#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

LITTLEBIT_PATH=${1:-"$ROOT_DIR/outputs/qwen3-4b-dflash-littlebit-0.5bit/epoch_0_step_500"}
OUTPUT_DIR=${2:-"$ROOT_DIR/results/qwen3_4b_sd_compare"}

python "$ROOT_DIR/scripts/compare_sd_benchmarks.py" \
    --target-model-path Qwen/Qwen3-4B \
    --eagle3-draft-model-path AngelSlim/Qwen3-4B_eagle3 \
    --dflash-draft-model-path z-lab/Qwen3-4B-DFlash-b16 \
    --littlebit-draft-model-path "$LITTLEBIT_PATH" \
    --benchmark-list mtbench:20 gsm8k:20 math500:20 \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --torch-dtype bfloat16 \
    --trust-remote-code
