#!/bin/bash
set -e
# Usage: bash index-xml.sh
# Encodes t2xml's retrieval/{corpus,queries}.test.jsonl with a finetuned
# SANTA checkpoint, searches a FAISS IndexFlatIP (see evaluate_xml/index_xml.py),
# and writes a TREC-format run file for evaluate_xml.sh to score. Run
# best-dev-finetune.sh first so MODEL_PATH's default (.../checkpoints/best_dev)
# exists -- otherwise point MODEL_PATH at a specific checkpoint yourself.
#
# Also dumps the corpus/query embeddings this step already computes to
# results${RUN_TAG}/embeddings.npz (gitignored, local only). Keep it after
# deleting runs/**/checkpoints to save space: point
# diagnostics/viz_embeddings.py's --model at that .npz to re-plot without the
# checkpoint on disk. Set EMB_SAVE_PATH="" to skip.

export SPLIT=test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT="$(dirname "${SANTA_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

export MODEL_NAME=${MODEL_NAME:-}
export RUN_TAG=${RUN_TAG:-${MODEL_NAME:+/${MODEL_NAME}}}
if [[ -z "${RUN_TAG}" ]]; then
    echo "WARNING: RUN_TAG is not set -- writing to the untagged default path (runs/retrieve), not a tagged experiment folder." >&2
fi
# FINETUNE_SOURCE = the folder slug after "_to_" in MODEL_NAME, e.g.
# BIKE_7_to_FOSSIG -> FOSSIG. No folder slug contains the substring "_to_",
# so this round-trips cleanly.
export FINETUNE_SOURCE=${FINETUNE_SOURCE:-${MODEL_NAME##*_to_}}
export MODEL_PATH=${MODEL_PATH:-${SANTA_DIR}/runs/finetune${RUN_TAG}/checkpoints/best_dev}
export TREC_PATH=${TREC_PATH:-${SANTA_DIR}/runs/retrieve${RUN_TAG}/${SPLIT}_inference.trec}
export EMB_SAVE_PATH=${EMB_SAVE_PATH-${REPO_ROOT}/results${RUN_TAG}/embeddings.npz}
export RETRIEVAL_ROOT=${RETRIEVAL_ROOT:-${REPO_ROOT}/retrieval}

SAVE_EMB_ARGS=()
if [[ -n "${EMB_SAVE_PATH}" ]]; then
    SAVE_EMB_ARGS=(--save_embeddings "${EMB_SAVE_PATH}")
fi

cd "${SANTA_DIR}/evaluate_xml"
"${PYTHON}" index_xml.py \
    --model_name_or_path ${MODEL_PATH} \
    --corpus_path ${RETRIEVAL_ROOT}${FINETUNE_SOURCE:+/${FINETUNE_SOURCE}}/corpus.${SPLIT}.jsonl \
    --query_path ${RETRIEVAL_ROOT}${FINETUNE_SOURCE:+/${FINETUNE_SOURCE}}/queries.${SPLIT}.jsonl \
    --trec_save_path ${TREC_PATH} \
    --per_device_eval_batch_size 64 \
    --q_max_len 50 \
    --p_max_len ${P_MAX_LEN:-256} \
    --topk 100 \
    "${SAVE_EMB_ARGS[@]}"
