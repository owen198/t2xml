#!/bin/bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Script to reproduct results

DIMS="2"
MODEL='poincare'

while true; do
  case "$1" in
    -d | --dim ) DIMS=$2; shift; shift ;;
    -m | --model ) MODEL=$2; shift; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

USAGE="usage: ./train-nouns.sh -d <dim> -m <model>
  -d: dimensions to use
  -m: model to use (can be lorentz or poincare)
  Example: ./train-nouns.sh -m lorentz -d 10
"

case "$MODEL" in
  "lorentz" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "poincare" ) EXTRA_ARGS=("-lr" "1.0");;
  * ) echo "$USAGE"; exit 1;;
esac

python embed.py \
  -checkpoint xsd_5.pth \
  -dset xsd/xsd_closure.csv \
  -epochs 300 \
  -negs 50 \
  -burnin 20 \
  -dampening 0.75 \
  -ndproc 4 \
  -eval_each 1 \
  -fresh \
  -sparse \
  -burnin_multiplier 0.01 \
  -neg_multiplier 0.1 \
  -lr_type constant \
  -lr 0.3 \
  -train_threads 2 \
  -batchsize 10 \
  -manifold poincare \
  -dim 5 \
  "${EXTRA_ARGS[@]}"

python embed.py \
       -dim 1200 \
       -lr 0.3 \
       -epochs 300 \
       -negs 50 \
       -burnin 20 \
       -ndproc 1 \
       -model distance \
       -manifold euclidean \
       -dset xsd/all_xsd_direct.csv \
       -checkpoint xsd_models/all_euclidean/eall__direct_xsd_1200.pth \
       -batchsize 10 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -gpu -1 \
       -train_threads 1

  python embed.py \
  -dim 1200 \
  -lr 0.005 \
  -epochs 300 \
  -negs 50 \
  -burnin 20 \
  -ndproc 1 \
  -model distance \
  -manifold poincare \
  -dset xsd/all_xsd_direct.csv \
  -checkpoint xsd_models/all_poincare/pall_direct_xsd_1200.pth \
  -batchsize 10 \
  -eval_each 1 \
  -fresh \
  -gpu -1 \
  -sparse \
  -train_threads 1


python viz_embeddings.py \
  --checkpoint cap10_all_xsd_2.pth \
  --style wn \
  --fit-to-disk \
  --edges xsd/all_xsd_closure_parent_cap10.csv \
  --edge-format names \
  --edge-direction child-parent \
  --depth-from reduced \
  --color-by fixed \
  --node-color black \
  --node-size-by depth \
  --min-node-size 5 \
  --max-node-size 50 \
  --edge-color "#4f6ea8" \
  --edge-alpha 0.35 \
  --edge-width 0.5 \
  --edge-color-by single \
  --annotate 20 \
  --label-mode top-degree \
  --label-fontsize 12 \
  --out cap10_all_xsd_2.png