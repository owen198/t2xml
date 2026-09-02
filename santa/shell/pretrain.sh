#!/bin/bash
set -e
# Usage: bash pretrain.sh
# Pretrains SANTA (joint SDA contrastive + MEP generative loss) on t2xml's
# S1000D data. Run prepare_data.py first to produce data/pretrain.*.jsonl.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

export DATA_DIR=${SANTA_DIR}/data
export MODEL=${MODEL:-Salesforce/codet5-base}
export RUN_TAG=${RUN_TAG:-}
if [[ -z "${RUN_TAG}" ]]; then
    echo "WARNING: RUN_TAG is not set -- writing to the untagged default path (runs/pretrain), not a tagged experiment folder." >&2
fi
export OUTPUT=${SANTA_DIR}/runs/pretrain${RUN_TAG}

# See SANTA's own shell/pretrain-code.sh for why max_steps is computed
# explicitly rather than trusting num_train_epochs-based auto-derivation.
export N=$(wc -l < ${DATA_DIR}/pretrain.train.jsonl)
export MAX_STEPS=$(${PYTHON} -c "import math; n=${N}; micro=math.ceil(n/16); spe=micro//8; print(math.ceil(10*spe))")

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
    --max_steps ${MAX_STEPS} \
    --use_generate True \
    --logging_dir ${OUTPUT}/logs
