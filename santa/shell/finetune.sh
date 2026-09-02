#!/bin/bash
set -e
# Usage: bash finetune.sh
# Finetunes a pretrained SANTA checkpoint on t2xml's retrieval benchmark
# (data/finetune.*.jsonl, built from retrieval/corpus+queries+qrels).
#
# PRETRAIN_CHECKPOINT defaults to shell/best-dev-pretrain.sh's output -- run
# that first so .../checkpoints/best_dev exists (mirrors SANTA_v2's own
# finetune-code.sh, whose Pretrain_checkpoint likewise points at
# pretrain${RUN_TAG}/checkpoints/best_dev). Otherwise point PRETRAIN_CHECKPOINT
# at a specific runs/pretrain*/checkpoints/checkpoint-N dir yourself.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

export DATA_DIR=${SANTA_DIR}/data
export RUN_TAG=${RUN_TAG:-}
export PRETRAIN_CHECKPOINT=${PRETRAIN_CHECKPOINT:-${SANTA_DIR}/runs/pretrain${RUN_TAG}/checkpoints/best_dev}
export OUTPUT=${SANTA_DIR}/runs/finetune${RUN_TAG}

export N=$(wc -l < ${DATA_DIR}/finetune.train.jsonl)
export MAX_STEPS=$(${PYTHON} -c "import math; n=${N}; micro=math.ceil(n/16); spe=micro//8; print(math.ceil(12*spe))")

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
    --max_steps ${MAX_STEPS} \
    --use_generate False \
    --logging_dir ${OUTPUT}/logs
