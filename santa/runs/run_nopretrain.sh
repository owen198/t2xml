#!/bin/bash
# Usage: bash runs/run_nopretrain.sh <TAG> <GPU> <P_MAX_LEN> "<extra train args>" [SEED]
# Finetune the raw base model (no SDA/MEP pretraining) on TARGET (default S1000D_spec_sample) and evaluate on its test split.
# Optional env: TARGET, DATA_PAIR (a pair name in DATA_ROOT whose finetune.* files belong to TARGET), DATA_ROOT (default data_cnt).
TAG=$1; GPU=$2; export P_MAX_LEN=$3; EXTRA=$4; SEED=${5:-42}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$SANTA")"
# RESULTS_ROOT: where results/<TAG>/ is written (default results/; set e.g. results/02_pooling_fix).
RESULTS_ROOT=${RESULTS_ROOT:-$REPO/results}
cd "$SANTA"
L=runs/${TAG}_logs; mkdir -p $L; TARGET=${TARGET:-S1000D_spec_sample}; name=${DATA_PAIR:-BIKE_7_to_S1000D_spec_sample}; R=$RESULTS_ROOT/$TAG/$TARGET; mkdir -p $R
export EXTRA_TRAIN_ARGS="--gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $SEED $EXTRA"
export MODEL_NAME=x_to_$TARGET RUN_TAG=/$TAG/$TARGET CUDA_VISIBLE_DEVICES=$GPU DATA_DIR=$SANTA/${DATA_ROOT:-data_cnt}/$name
export PRETRAIN_CHECKPOINT=${BASE_MODEL:-Salesforce/codet5-base}
bash shell/finetune.sh                                                       > $L/finetune.log 2>&1 || { echo FAIL finetune >> $L/progress.log; exit 1; }
RESULTS_PATH=$R/best_dev_finetune.json bash shell/best-dev-finetune.sh       > $L/bdf.log 2>&1 || { echo FAIL bdf >> $L/progress.log; exit 1; }
EMB_SAVE_PATH="$R/embeddings.npz" bash shell/index-xml.sh                                                      > $L/index.log 2>&1 || { echo FAIL index >> $L/progress.log; exit 1; }
RESULTS_PATH=$R/eval_test.json bash shell/evaluate_xml.sh                    > $L/eval.log 2>&1 || { echo FAIL eval >> $L/progress.log; exit 1; }
echo "DONE $TAG" >> $L/progress.log
