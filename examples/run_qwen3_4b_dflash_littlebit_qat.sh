#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-32}

NUM_GPUS=${1:-4}
EFF_BIT=${2:-0.5}
TRAIN_DATA_PATH=${3:-"$ROOT_DIR/cache/dataset/perfectblend_qwen3_4b_regen.jsonl"}
OUTPUT_DIR=${4:-"$ROOT_DIR/outputs/qwen3-4b-dflash-littlebit-${EFF_BIT}bit"}

torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$ROOT_DIR/scripts/train_littlebit_dflash.py" \
    --target-model-path Qwen/Qwen3-4B \
    --draft-model-path z-lab/Qwen3-4B-DFlash-b16 \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 3 \
    --batch-size 2 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.03 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --target-model-backend hf \
    --attention-backend flex_attention \
    --num-anchors 512 \
    --log-interval 20 \
    --save-interval 500 \
    --eval-interval 1000 \
    --eff-bit "$EFF_BIT" \
    --quant-func STEBinary \
    --kv-factor 1.0 \
    --l2l-loss-scale 10.0 \
    --kd-loss-scale 1.0 \
    --report-to wandb \
    --wandb-project specforge-qwen3-4b-dflash-littlebit \
    --wandb-name "qwen3-4b-dflash-littlebit-${EFF_BIT}bit" \
    --trust-remote-code
