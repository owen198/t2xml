#!/bin/bash
SANTA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(dirname "$SANTA")"
# RESULTS_ROOT: where results/<TAG>/ is written (default results/; set e.g. results/02_pooling_fix).
RESULTS_ROOT=${RESULTS_ROOT:-$REPO/results}
cd "$SANTA"
TAG=ga1
L=runs/${TAG}_logs
export EXTRA_TRAIN_ARGS="--gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none"
for name in $(python3 -c "import json;print('\n'.join(m['name'] for m in json.load(open('model_configs.json'))['models']))"); do
  export MODEL_NAME=$name RUN_TAG=/$TAG/$name CUDA_VISIBLE_DEVICES=0
  echo "[$(date +%T)] START $name" >> $L/progress.log
  bash shell/pretrain.sh                                                         > $L/$name.pretrain.log 2>&1 || { echo "FAIL pretrain $name" >> $L/progress.log; continue; }
  RESULTS_PATH=$L/$name.best_dev_pretrain.json bash shell/best-dev-pretrain.sh   > $L/$name.bdp.log 2>&1 || { echo "FAIL bdp $name" >> $L/progress.log; continue; }
  bash shell/finetune.sh                                                         > $L/$name.finetune.log 2>&1 || { echo "FAIL finetune $name" >> $L/progress.log; continue; }
  RESULTS_PATH=$L/$name.best_dev_finetune.json bash shell/best-dev-finetune.sh   > $L/$name.bdf.log 2>&1 || { echo "FAIL bdf $name" >> $L/progress.log; continue; }
  bash shell/index-xml.sh                                                        > $L/$name.index.log 2>&1 || { echo "FAIL index $name" >> $L/progress.log; continue; }
  RESULTS_PATH=$L/$name.eval_test.json bash shell/evaluate_xml.sh                > $L/$name.eval.log 2>&1 || { echo "FAIL eval $name" >> $L/progress.log; continue; }
  echo "[$(date +%T)] DONE  $name" >> $L/progress.log
done
echo "[$(date +%T)] ALL-DONE" >> $L/progress.log
