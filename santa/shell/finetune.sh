#!/bin/bash
set -e
# Usage: bash finetune.sh
#    or: MODEL_NAME=<name from model_configs.json> bash finetune.sh
# Finetunes a pretrained SANTA checkpoint on t2xml's retrieval benchmark
# (data/<MODEL_NAME>/finetune.*.jsonl, built from retrieval/<finetune_source>/
# corpus+queries+qrels).
#
# PRETRAIN_CHECKPOINT defaults to shell/best-dev-pretrain.sh's output -- run
# that first so .../checkpoints/best_dev exists (mirrors SANTA_v2's own
# finetune-code.sh, whose Pretrain_checkpoint likewise points at
# pretrain${RUN_TAG}/checkpoints/best_dev). Otherwise point PRETRAIN_CHECKPOINT
# at a specific runs/pretrain*/checkpoints/checkpoint-N dir yourself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

# Default to a single GPU: with >1 GPU visible and no torchrun/distributed
# launcher, the HF Trainer here auto-enters a distributed code path that
# hangs forever waiting for a peer process that never joins. Override this
# yourself (e.g. "0,1") if you're launching under torchrun for real multi-GPU.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODEL_NAME=${MODEL_NAME:-}
export DATA_DIR=${DATA_DIR:-${SANTA_DIR}/data${MODEL_NAME:+/${MODEL_NAME}}}
export RUN_TAG=${RUN_TAG:-${MODEL_NAME:+/${MODEL_NAME}}}
if [[ -z "${RUN_TAG}" ]]; then
    echo "WARNING: RUN_TAG is not set -- writing to the untagged default path (runs/finetune), not a tagged experiment folder." >&2
fi
export PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-${SANTA_DIR}/runs/pretrain${RUN_TAG}/checkpoints/best_dev}
export OUTPUT=${SANTA_DIR}/runs/finetune${RUN_TAG}

# max_steps is intentionally left unset (HF TrainingArguments default -1) so
# --num_train_epochs alone governs training length -- see pretrain.sh for why
# the previous explicit steps-per-epoch formula broke (computed 0 steps) for
# most of t2xml's per-folder train splits.

cd "${SANTA_DIR}"
"${PYTHON}" train_santa.py \
    --output_dir ${OUTPUT}/checkpoints \
    --model_name_or_path ${PRETRAIN_CHECKPOINT} \
    --do_train \
    --overwrite_output_dir \
    --save_steps 1000 \
    --save_total_limit 3 \
    --train_path ${DATA_DIR}/finetune.train.jsonl \
    --eval_path ${DATA_DIR}/finetune.dev.jsonl \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --bf16 True \
    --train_n_passages 1 \
    --learning_rate 2e-5 \
    --q_max_len 50 \
    --p_max_len 256 \
    --l_max_len 64 \
    --num_train_epochs 12 \
    --use_generate False \
    --logging_dir ${OUTPUT}/logs
