#!/bin/bash
# Usage: bash runs/run_samedomain.sh <TAG> <GPU> <SEED> <ab|ba> [DATA_SUFFIX]
# DATA_SUFFIX (optional, default ""): use data_samedomain<SUFFIX>/ instead of data_samedomain/, e.g. "_alt".
# Runs both directions of the same-domain swap (pretrain on one S1000D half, finetune on the other) plus
# their matched finetune-only controls (same finetune half, no pretraining). Config: mean pooling +
# cosine tau 0.05, p_max_len 1024 -- the best config found so far.
TAG=$1; GPU=$2; SEED=$3; DSUF=${5:-}
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$SANTA")"
# RESULTS_ROOT: where results/<TAG>/ is written (default results/; set e.g. results/02_pooling_fix).
RESULTS_ROOT=${RESULTS_ROOT:-$REPO/results}
cd "$SANTA"
L=runs/${TAG}_logs; mkdir -p $L; R=$RESULTS_ROOT/$TAG
export P_MAX_LEN=1024
COMMON="--pooling mean --normalize True --temperature 0.05 --gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none --seed $SEED"

run_one() {
  arm=$1; data=$2; do_pretrain=$3
  n=${TAG}_$arm; mkdir -p $R/$arm
  export MODEL_NAME=S1000D_spec_sample_to_S1000D_spec_sample RUN_TAG=/$TAG/$arm CUDA_VISIBLE_DEVICES=$GPU DATA_DIR=$SANTA/data_samedomain${DSUF}/$data
  echo "[$(date +%T)] START $n" >> $L/progress.log
  if [ "$do_pretrain" = "yes" ]; then
    EXTRA_TRAIN_ARGS="$COMMON" bash shell/pretrain.sh > $L/$n.pretrain.log 2>&1 || { echo "FAIL pretrain $n" >> $L/progress.log; return; }
    RESULTS_PATH=$R/$arm/best_dev_pretrain.json bash shell/best-dev-pretrain.sh > $L/$n.bdp.log 2>&1 || { echo "FAIL bdp $n" >> $L/progress.log; return; }
  else
    export PRETRAIN_CHECKPOINT=Salesforce/codet5-base
  fi
  EXTRA_TRAIN_ARGS="$COMMON" bash shell/finetune.sh > $L/$n.finetune.log 2>&1 || { echo "FAIL finetune $n" >> $L/progress.log; return; }
  RESULTS_PATH=$R/$arm/best_dev_finetune.json bash shell/best-dev-finetune.sh > $L/$n.bdf.log 2>&1 || { echo "FAIL bdf $n" >> $L/progress.log; return; }
  EMB_SAVE_PATH="$R/$arm/embeddings.npz" bash shell/index-xml.sh > $L/$n.index.log 2>&1 || { echo "FAIL index $n" >> $L/progress.log; return; }
  RESULTS_PATH=$R/$arm/eval_test.json bash shell/evaluate_xml.sh > $L/$n.eval.log 2>&1 || { echo "FAIL eval $n" >> $L/progress.log; return; }
  echo "[$(date +%T)] DONE  $n" >> $L/progress.log
}

case $4 in
  ab) run_one "pretrainA_finetuneB" "pretrainA_finetuneB" yes
      run_one "finetuneB_only" "finetuneB_only" no ;;
  ba) run_one "pretrainB_finetuneA" "pretrainB_finetuneA" yes
      run_one "finetuneA_only" "finetuneA_only" no ;;
esac
echo "[$(date +%T)] ALL-DONE $TAG $4" >> $L/progress.log
