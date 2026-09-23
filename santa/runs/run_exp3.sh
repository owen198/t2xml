#!/bin/bash
# Usage: bash runs/run_exp3.sh <TAG> <P_MAX_LEN> <GPU> [SEED]
# Original XML, dot-product, bs16, accumulation 1 (= ga1 config), but with data tokenized by
# codet5-base's own tokenizer (data_cnt/, safety cap 1024). S1000D_spec_sample target only.
TAG=$1; export P_MAX_LEN=$2; GPU=$3; SEED=${4:-42}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$SANTA")"
# RESULTS_ROOT: where results/<TAG>/ is written (default results/; set e.g. results/02_pooling_fix).
RESULTS_ROOT=${RESULTS_ROOT:-$REPO/results}
cd "$SANTA"
L=runs/exp3_logs; mkdir -p $L
export EXTRA_TRAIN_ARGS="--gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $SEED"
for src in BIKE_7 FOSSIG s1kd-tools-doc; do
  name=${src}_to_S1000D_spec_sample; n=${TAG}_$name
  export MODEL_NAME=$name RUN_TAG=/${TAG}/$name CUDA_VISIBLE_DEVICES=$GPU DATA_DIR=$SANTA/data_cnt/$name
  echo "[$(date +%T)] START $n" >> $L/progress.log
  bash shell/pretrain.sh                                                   > $L/$n.pretrain.log 2>&1 || { echo "FAIL pretrain $n" >> $L/progress.log; continue; }
  RESULTS_PATH=$RESULTS_ROOT/${TAG}/$name/best_dev_pretrain.json bash shell/best-dev-pretrain.sh > $L/$n.bdp.log 2>&1 || { echo "FAIL bdp $n" >> $L/progress.log; continue; }
  bash shell/finetune.sh                                                   > $L/$n.finetune.log 2>&1 || { echo "FAIL finetune $n" >> $L/progress.log; continue; }
  RESULTS_PATH=$RESULTS_ROOT/${TAG}/$name/best_dev_finetune.json bash shell/best-dev-finetune.sh > $L/$n.bdf.log 2>&1 || { echo "FAIL bdf $n" >> $L/progress.log; continue; }
  EMB_SAVE_PATH="$RESULTS_ROOT/$TAG/$name/embeddings.npz" bash shell/index-xml.sh                                                  > $L/$n.index.log 2>&1 || { echo "FAIL index $n" >> $L/progress.log; continue; }
  RESULTS_PATH=$RESULTS_ROOT/${TAG}/$name/eval_test.json bash shell/evaluate_xml.sh > $L/$n.eval.log 2>&1 || { echo "FAIL eval $n" >> $L/progress.log; continue; }
  echo "[$(date +%T)] DONE  $n" >> $L/progress.log
done
echo "[$(date +%T)] ALL-DONE $TAG" >> $L/progress.log
