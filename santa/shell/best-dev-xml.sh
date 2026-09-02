#!/bin/bash
set -e
# Usage: bash best-dev-xml.sh
# Evaluates every checkpoint from finetune.sh's run (checkpoint-N subdirs, plus
# the final save at the run root) against retrieval/*.dev.jsonl, then copies
# whichever scored highest into checkpoints/best_dev -- SANTA's own
# code_best_dev/product_best_dev pattern, adapted for t2xml's retrieval format.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${SANTA_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

export RUN_TAG=${RUN_TAG:-}
export MODEL_PATH=${MODEL_PATH:-${SANTA_DIR}/runs/finetune${RUN_TAG}/checkpoints}
export RESULTS_PATH=${RESULTS_PATH:-${REPO_ROOT}/results${RUN_TAG}/best_dev_finetune.json}

cd "${SANTA_DIR}/best_dev"
"${PYTHON}" evaluate_xml_finetune.py \
    --model_path ${MODEL_PATH} \
    --corpus_path ${REPO_ROOT}/retrieval/corpus.dev.jsonl \
    --query_path ${REPO_ROOT}/retrieval/queries.dev.jsonl \
    --qrels_path ${REPO_ROOT}/retrieval/qrels.dev.tsv \
    --per_device_eval_batch_size 64 \
    --q_max_len 50 \
    --p_max_len 256 \
    --topk 100 \
    --results_path ${RESULTS_PATH}
