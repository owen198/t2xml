#!/bin/bash
set -e
# Usage: bash pretrain.sh
#    or: MODEL_NAME=<name from model_configs.json> bash pretrain.sh
# Pretrains SANTA (joint SDA contrastive + MEP generative loss) on t2xml's
# S1000D data. Run prepare_data.py first to produce data/<MODEL_NAME>/
# pretrain.*.jsonl (or data/pretrain.*.jsonl if MODEL_NAME is unset).

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
export MODEL=${MODEL:-Salesforce/codet5-base}
export RUN_TAG=${RUN_TAG:-${MODEL_NAME:+/${MODEL_NAME}}}
if [[ -z "${RUN_TAG}" ]]; then
    echo "WARNING: RUN_TAG is not set -- writing to the untagged default path (runs/pretrain), not a tagged experiment folder." >&2
fi
export OUTPUT=${SANTA_DIR}/runs/pretrain${RUN_TAG}

# max_steps is intentionally left unset (HF TrainingArguments default -1) so
# --num_train_epochs alone governs training length. An explicit steps-per-
# epoch formula (round(train_size/effective_batch_size)) used to compute this
# instead, but that breaks down -- computing 0 steps -- for any per-folder
# train split smaller than one effective batch (16 * 8 = 128 examples), which
# is most of t2xml's per-folder splits (see santa/model_configs.json).

cd "${SANTA_DIR}"
"${PYTHON}" train_santa.py \
    --output_dir ${OUTPUT}/checkpoints \
    --model_name_or_path ${MODEL} \
    --do_train \
    --save_steps 1000 \
    --save_total_limit 3 \
    --train_path ${DATA_DIR}/pretrain.train.jsonl \
    --eval_path ${DATA_DIR}/pretrain.dev.jsonl \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --bf16 True \
    --train_n_passages 1 \
    --learning_rate 5e-5 \
    --q_max_len 50 \
    --p_max_len 256 \
    --l_max_len 64 \
    --num_train_epochs 10 \
    --use_generate True \
    --logging_dir ${OUTPUT}/logs
