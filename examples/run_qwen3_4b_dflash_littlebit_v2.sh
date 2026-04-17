#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-32}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

NUM_GPUS=${1:-4}
EFF_BIT=${2:-0.5}
TRAIN_DATA_PATH=${3:-"$ROOT_DIR/cache/dataset/perfectblend_qwen3_4b_regen.jsonl"}
OUTPUT_DIR=${4:-"$ROOT_DIR/outputs/qwen3-4b-dflash-littlebit-v2-${EFF_BIT}bit"}
NUM_ANCHORS=${5:-512}
MAX_LENGTH=${6:-3072}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen3-4B}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-z-lab/Qwen3-4B-DFlash-b16}
TARGET_BACKEND=${TARGET_BACKEND:-sglang}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flex_attention}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.35}
LOGIT_CHUNK_SIZE=${LOGIT_CHUNK_SIZE:-512}
BATCH_SIZE=${BATCH_SIZE:-2}
REPORT_TO=${REPORT_TO:-wandb}

EXTRA_ARGS=()
if [[ "${DEBUG_STEP_TIMING:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--debug-step-timing --debug-step-interval "${DEBUG_STEP_INTERVAL:-1}")
fi

torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$ROOT_DIR/scripts/train_littlebit_dflash_v2.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-path "$DRAFT_MODEL_PATH" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 3 \
    --batch-size "$BATCH_SIZE" \
    --learning-rate 2e-5 \
    --warmup-ratio 0.03 \
    --max-grad-norm 1.0 \
    --max-length "$MAX_LENGTH" \
    --chat-template qwen \
    --target-model-backend "$TARGET_BACKEND" \
    --attention-backend "$ATTENTION_BACKEND" \
    --num-anchors "$NUM_ANCHORS" \
    --fixed-num-anchors \
    --loss-decay-gamma 7.0 \
    --logit-chunk-size "$LOGIT_CHUNK_SIZE" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    --log-interval 5 \
    --save-interval 500 \
    --eval-interval 1000 \
    --eff-bit "$EFF_BIT" \
    --quant-func STEBinary \
    --kv-factor 1.0 \
    --report-to "$REPORT_TO" \
    --wandb-project specforge-qwen3-4b-dflash-littlebit \
    --wandb-name "qwen3-4b-dflash-littlebit-v2-${EFF_BIT}bit" \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
