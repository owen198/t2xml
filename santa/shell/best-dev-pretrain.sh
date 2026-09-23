#!/bin/bash
set -e
# Usage: bash best-dev-pretrain.sh
#    or: MODEL_NAME=<name from model_configs.json> bash best-dev-pretrain.sh
# Evaluates every checkpoint from pretrain.sh's run (checkpoint-N subdirs, plus
# the final save at the run root) against <pretrain_source>/sda_pairs.dev.jsonl
# (the pretraining task's own self-referential SDA dev pairs, not the Stage 3
# retrieval/*.dev.jsonl benchmark), then copies whichever scored highest into
# checkpoints/best_dev -- mirrors best-dev-finetune.sh, but for pretrain.sh's run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${SANTA_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

export MODEL_NAME=${MODEL_NAME:-}
export RUN_TAG=${RUN_TAG:-${MODEL_NAME:+/${MODEL_NAME}}}
if [[ -z "${RUN_TAG}" ]]; then
    echo "WARNING: RUN_TAG is not set -- writing to the untagged default path (results/), not a tagged experiment folder." >&2
fi
# PRETRAIN_SOURCE = the folder slug before "_to_" in MODEL_NAME, e.g.
# BIKE_7_to_FOSSIG -> BIKE_7. No folder slug contains the substring "_to_",
# so this round-trips cleanly.
export PRETRAIN_SOURCE=${PRETRAIN_SOURCE:-${MODEL_NAME%%_to_*}}
export MODEL_PATH=${MODEL_PATH:-${SANTA_DIR}/runs/pretrain${RUN_TAG}/checkpoints}
export RESULTS_PATH=${RESULTS_PATH:-${REPO_ROOT}/results${RUN_TAG}/best_dev_pretrain.json}

cd "${SANTA_DIR}/best_dev"
"${PYTHON}" evaluate_xml_pretrain.py \
    --model_path ${MODEL_PATH} \
    --eval_path ${REPO_ROOT}/pretrain${PRETRAIN_SOURCE:+/${PRETRAIN_SOURCE}}/sda_pairs.dev.jsonl \
    --per_device_eval_batch_size 64 \
    --q_max_len 50 \
    --p_max_len ${P_MAX_LEN:-256} \
    --topk 100 \
    --results_path ${RESULTS_PATH}
