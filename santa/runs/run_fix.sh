#!/bin/bash
# Usage: bash runs/run_fix.sh <TAG> <GPU> <P_MAX_LEN> "<extra train args>" [SEED]
# CodeT5, original XML, S1000D_spec_sample target only (3 pretrain sources), data from data_cnt/ (codet5 tokenizer, cap 1024).
# Base config = ga1 (bs16, accumulation 1, lr 5e-5 / 2e-5, 10 / 12 epochs); extra args select pooling / normalize / temperature.
TAG=$1; GPU=$2; export P_MAX_LEN=$3; EXTRA=$4; SEED=${5:-42}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$SANTA")"
# RESULTS_ROOT: where results/<TAG>/ is written (default results/; set e.g. results/02_pooling_fix).
RESULTS_ROOT=${RESULTS_ROOT:-$REPO/results}
cd "$SANTA"
L=runs/${TAG}_logs; mkdir -p $L; R=$RESULTS_ROOT/$TAG
export EXTRA_TRAIN_ARGS="--gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $SEED $EXTRA"
for src in BIKE_7 FOSSIG s1kd-tools-doc; do
  name=${src}_to_S1000D_spec_sample; n=${TAG}_$name; mkdir -p $R/$name
  export MODEL_NAME=$name RUN_TAG=/$TAG/$name CUDA_VISIBLE_DEVICES=$GPU DATA_DIR=$SANTA/data_cnt/$name
  echo "[$(date +%T)] START $n" >> $L/progress.log
  bash shell/pretrain.sh                                                        > $L/$n.pretrain.log 2>&1 || { echo "FAIL pretrain $n" >> $L/progress.log; continue; }
  RESULTS_PATH=$R/$name/best_dev_pretrain.json bash shell/best-dev-pretrain.sh  > $L/$n.bdp.log 2>&1 || { echo "FAIL bdp $n" >> $L/progress.log; continue; }
  bash shell/finetune.sh                                                        > $L/$n.finetune.log 2>&1 || { echo "FAIL finetune $n" >> $L/progress.log; continue; }
  RESULTS_PATH=$R/$name/best_dev_finetune.json bash shell/best-dev-finetune.sh  > $L/$n.bdf.log 2>&1 || { echo "FAIL bdf $n" >> $L/progress.log; continue; }
  EMB_SAVE_PATH="$R/$name/embeddings.npz" bash shell/index-xml.sh                                                       > $L/$n.index.log 2>&1 || { echo "FAIL index $n" >> $L/progress.log; continue; }
  RESULTS_PATH=$R/$name/eval_test.json bash shell/evaluate_xml.sh               > $L/$n.eval.log 2>&1 || { echo "FAIL eval $n" >> $L/progress.log; continue; }
  echo "[$(date +%T)] DONE  $n" >> $L/progress.log
done
echo "[$(date +%T)] ALL-DONE $TAG" >> $L/progress.log
