#!/bin/bash
set -e
# Usage: bash index-xml.sh
# Encodes t2xml's retrieval/{corpus,queries}.test.jsonl with a finetuned
# SANTA checkpoint, searches a FAISS IndexFlatIP (see evaluate_xml/index_xml.py),
# and writes a TREC-format run file for evaluate_xml.sh to score. Run
# best-dev-xml.sh first so MODEL_PATH's default (.../checkpoints/best_dev)
# exists -- otherwise point MODEL_PATH at a specific checkpoint yourself.
#
# The test split is hardcoded, matching SANTA's own index-code.sh (which points
# at ${language}_test_doc.jsonl / ${language}_test_query.jsonl and takes no
# split argument). dev is used only by best-dev-xml.sh for checkpoint
# selection, so scoring it here would report on the split the model was
# selected with.
export SPLIT=test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${SANTA_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

export RUN_TAG=${RUN_TAG:-}
if [[ -z "${RUN_TAG}" ]]; then
    echo "WARNING: RUN_TAG is not set -- writing to the untagged default path (runs/retrieve), not a tagged experiment folder." >&2
fi
export MODEL_PATH=${MODEL_PATH:-${SANTA_DIR}/runs/finetune${RUN_TAG}/checkpoints/best_dev}
export TREC_PATH=${TREC_PATH:-${SANTA_DIR}/runs/retrieve${RUN_TAG}/${SPLIT}_inference.trec}

cd "${SANTA_DIR}/evaluate_xml"
"${PYTHON}" index_xml.py \
    --model_name_or_path ${MODEL_PATH} \
    --corpus_path ${REPO_ROOT}/retrieval/corpus.${SPLIT}.jsonl \
    --query_path ${REPO_ROOT}/retrieval/queries.${SPLIT}.jsonl \
    --trec_save_path ${TREC_PATH} \
    --per_device_eval_batch_size 64 \
    --q_max_len 50 \
    --p_max_len 256 \
    --topk 100
