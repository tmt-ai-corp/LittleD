#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

DRAFT_MODEL_PATH=${1:-z-lab/Qwen3-4B-DFlash-b16}
DRAFT_TYPE=${2:-dflash}
OUTPUT_PATH=${3:-"$ROOT_DIR/results/qwen3_4b_dflash_smoke.json"}

python "$ROOT_DIR/scripts/eval_dflash_acceptance.py" \
    --target-model-path Qwen/Qwen3-4B \
    --draft-model-path "$DRAFT_MODEL_PATH" \
    --draft-type "$DRAFT_TYPE" \
    --benchmark-list gsm8k:2 math500:2 mtbench:2 \
    --max-input-tokens 1024 \
    --max-new-tokens 256 \
    --output-path "$OUTPUT_PATH" \
    --device cuda \
    --torch-dtype bfloat16 \
    --trust-remote-code
