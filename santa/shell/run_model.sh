#!/bin/bash
set -e
# Usage: MODEL_NAME=<name from model_configs.json> bash run_model.sh
# Runs one full model end-to-end: prepare_data -> pretrain -> best-dev-pretrain
# -> finetune -> best-dev-finetune -> index -> evaluate. Assumes Stages 1-3
# (preprocess.py / build_xml_entity.py / build_retrieval_dataset.py) have
# already been run for both the model's pretrain_source and finetune_source
# folders under pretrain/<slug>/ and retrieval/<slug>/.
#
# To run all models in model_configs.json:
#   for m in $(python3 -c "import json; print('\n'.join(x['name'] for x in json.load(open('santa/model_configs.json'))['models']))"); do
#       MODEL_NAME=$m bash santa/shell/run_model.sh
#   done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANTA_DIR="$(dirname "${SCRIPT_DIR}")"
PYTHON="${SANTA_DIR}/.venv/bin/python"

: "${MODEL_NAME:?set MODEL_NAME to a name in santa/model_configs.json}"
export MODEL_NAME

"${PYTHON}" "${SANTA_DIR}/prepare_data.py" --models "${MODEL_NAME}"
bash "${SCRIPT_DIR}/pretrain.sh"
bash "${SCRIPT_DIR}/best-dev-pretrain.sh"
bash "${SCRIPT_DIR}/finetune.sh"
bash "${SCRIPT_DIR}/best-dev-finetune.sh"
bash "${SCRIPT_DIR}/index-xml.sh"
bash "${SCRIPT_DIR}/evaluate_xml.sh"
