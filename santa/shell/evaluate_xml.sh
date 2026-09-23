#!/bin/bash
set -e
# Usage: bash evaluate_xml.sh
# Computes MRR@100 from a TREC-format retrieval run against t2xml's
# retrieval/qrels.test.tsv. Does not run retrieval itself -- point
# TREC_PATH at wherever your retriever wrote its run file first.

export SPLIT=test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${SANTA_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

export MODEL_NAME=${MODEL_NAME:-}
export RUN_TAG=${RUN_TAG:-${MODEL_NAME:+/${MODEL_NAME}}}
if [[ -z "${RUN_TAG}" ]]; then
    echo "WARNING: RUN_TAG is not set -- writing to the untagged default path (results/), not a tagged experiment folder." >&2
fi
# FINETUNE_SOURCE = the folder slug after "_to_" in MODEL_NAME, e.g.
# BIKE_7_to_FOSSIG -> FOSSIG. No folder slug contains the substring "_to_",
# so this round-trips cleanly.
export FINETUNE_SOURCE=${FINETUNE_SOURCE:-${MODEL_NAME##*_to_}}
export RETRIEVAL_ROOT=${RETRIEVAL_ROOT:-${REPO_ROOT}/retrieval}
export QRELS_PATH=${RETRIEVAL_ROOT}${FINETUNE_SOURCE:+/${FINETUNE_SOURCE}}/qrels.${SPLIT}.tsv
export TREC_PATH=${TREC_PATH:-${SANTA_DIR}/runs/retrieve${RUN_TAG}/${SPLIT}_inference.trec}
export RESULTS_PATH=${RESULTS_PATH:-${REPO_ROOT}/results${RUN_TAG}/eval_${SPLIT}.json}

cd "${SANTA_DIR}/evaluate_xml"
"${PYTHON}" evaluate_xml.py \
    --trec_save_path ${TREC_PATH} \
    --qrels_path ${QRELS_PATH} \
    --results_path ${RESULTS_PATH}
