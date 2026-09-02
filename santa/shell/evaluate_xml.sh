#!/bin/bash
set -e
# Usage: bash evaluate_xml.sh
# Computes MRR@100 from a TREC-format retrieval run against t2xml's
# retrieval/qrels.test.tsv. Does not run retrieval itself -- point
# TREC_PATH at wherever your retriever wrote its run file first.
#
# The test split is hardcoded, matching index-xml.sh and SANTA's own
# evaluate_code.sh (which scores ${language}_inference.trec from the test dir
# and takes no split argument).
export SPLIT=test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${SANTA_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

export RUN_TAG=${RUN_TAG:-}
if [[ -z "${RUN_TAG}" ]]; then
    echo "WARNING: RUN_TAG is not set -- writing to the untagged default path (results/), not a tagged experiment folder." >&2
fi
export QRELS_PATH=${REPO_ROOT}/retrieval/qrels.${SPLIT}.tsv
export TREC_PATH=${TREC_PATH:-${SANTA_DIR}/runs/retrieve${RUN_TAG}/${SPLIT}_inference.trec}
export RESULTS_PATH=${RESULTS_PATH:-${REPO_ROOT}/results${RUN_TAG}/eval_${SPLIT}.json}

cd "${SANTA_DIR}/evaluate_xml"
"${PYTHON}" evaluate_xml.py \
    --trec_save_path ${TREC_PATH} \
    --qrels_path ${QRELS_PATH} \
    --results_path ${RESULTS_PATH}
