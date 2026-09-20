# Results (ga1: gradient accumulation 1)

## Pre-training source dev evaluation

Each pretrain checkpoint on its own source's SDA dev split.

| Pre-training | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| MRR (ga1) | 0.4444 | 0.5208 | 0.7500 | 0.3189 |
| MRR (accumulation 8) | 0.6111 | 0.5208 | 0.7500 | 0.2323 |

## Fine-tuning test evaluation

Test MRR of each run's `eval_test.json`. 
| Fine-tuning \ Pre-training | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| bike | — | 0.2833 | 0.4083 | 0.6389 |
| s1kd-tools-doc | 0.4444 | — | 0.6111 | 0.6111 |
| fossig | 0.7500 | 0.7500 | — | 0.7500 |
| s1000d-spec | 0.2275 | 0.2049 | 0.1755 | — |

## Per-pair comparison with the accumulation-8 run

| Pair (pretrain -> finetune) | Finetune dev MRR | Test MRR (ga1) | Test MRR (acc 8) | Δ |
| --- | --- | --- | --- | --- |
| BIKE_7 -> FOSSIG | 0.7500 | 0.7500 | 0.7500 | 0.0000 |
| BIKE_7 -> S1000D_spec_sample | 0.2588 | 0.2275 | 0.1760 | +0.0516 |
| BIKE_7 -> s1kd-tools-doc | 0.4792 | 0.4444 | 0.6111 | -0.1667 |
| FOSSIG -> BIKE_7 | 0.6111 | 0.4083 | 0.4083 | 0.0000 |
| FOSSIG -> S1000D_spec_sample | 0.2315 | 0.1755 | 0.1835 | -0.0080 |
| FOSSIG -> s1kd-tools-doc | 0.5208 | 0.6111 | 0.7500 | -0.1389 |
| S1000D_spec_sample -> BIKE_7 | 1.0000 | 0.6389 | 0.4083 | +0.2306 |
| S1000D_spec_sample -> FOSSIG | 0.7500 | 0.7500 | 0.7500 | 0.0000 |
| S1000D_spec_sample -> s1kd-tools-doc | 0.6875 | 0.6111 | 0.6111 | 0.0000 |
| s1kd-tools-doc -> BIKE_7 | 0.6111 | 0.2833 | 0.4083 | -0.1250 |
| s1kd-tools-doc -> FOSSIG | 0.7500 | 0.7500 | 0.7500 | 0.0000 |
| s1kd-tools-doc -> S1000D_spec_sample | 0.2617 | 0.2049 | 0.1815 | +0.0234 |

## Training loss

Train loss is logged every step: first step -> mean of the last 3 steps. It is contrastive + MEP generative loss on unnormalised
dot-product scores, so absolute values are large and spiky; compare within a row, not across rows. Dev loss is a single evaluation
at the end of the stage (`eval_steps` did not fire more than once per run).

**Pretrain** (one run per source; the three pairs sharing a source have identical loss values):

| Source | Steps | Train loss | Dev loss |
| --- | --- | --- | --- |
| BIKE_7 | 70 | 3047 -> 440 | 20.8 |
| s1kd-tools-doc | 30 | 3036 -> 3294 | 69.3 |
| FOSSIG | 10 | 2472 -> 4575 | 70.8 |
| S1000D_spec_sample | 190 | 3060 -> 61 | 15.0 |

**Finetune:**

| Pair | Steps | Train loss | Dev loss |
| --- | --- | --- | --- |
| BIKE_7 -> FOSSIG | 12 | 348 -> 331 | 0.691 |
| BIKE_7 -> S1000D_spec_sample | 228 | 336 -> 93 | 5.36 |
| BIKE_7 -> s1kd-tools-doc | 36 | 388 -> 413 | 1.31 |
| FOSSIG -> BIKE_7 | 84 | 4308 -> 2599 | 0.965 |
| FOSSIG -> S1000D_spec_sample | 228 | 4400 -> 205 | 4.59 |
| FOSSIG -> s1kd-tools-doc | 36 | 4412 -> 3655 | 1.38 |
| S1000D_spec_sample -> BIKE_7 | 84 | 63 -> 49 | 1.59 |
| S1000D_spec_sample -> FOSSIG | 12 | 41 -> 69 | 0.691 |
| S1000D_spec_sample -> s1kd-tools-doc | 36 | 62 -> 71 | 2.48 |
| s1kd-tools-doc -> BIKE_7 | 84 | 3892 -> 1693 | 43.6 |
| s1kd-tools-doc -> FOSSIG | 12 | 3592 -> 3152 | 0.691 |
| s1kd-tools-doc -> S1000D_spec_sample | 228 | 3860 -> 211 | 8.35 |

## Observations

- **Training now moves where there is data.** Runs with many steps (S1000D_spec_sample pretrain, 190 steps: 3060 -> 61; the three
  S1000D-target finetunes, 228 steps: 3.6x to 21x drop) show a real loss decrease, unlike the accumulation-8 run. Runs on FOSSIG
  (10-12 steps) and several s1kd-tools-doc runs (30-36 steps) still show no clear decrease.
- **The S1000D_spec_sample target is the only informative one** (21 test queries). Its test MRR spread across pretrain sources is now
  0.176-0.228, versus 0.176-0.184 with accumulation 8, so the pretrain source now matters more. This is one seed. **Update:** later seed runs (`results/bs32_vs_ga1/`) show one pair scoring 0.14-0.32 across seeds with this same
  configuration, larger than this spread, so it is not evidence that the pretrain source matters.
- **The other targets are not reliable** (FOSSIG 2, s1kd-tools-doc 3, BIKE_7 6 test queries): MRR moves in steps of 0.05-0.25, so the
  +/-0.12-0.23 changes in the comparison table are one or two queries flipping. `S1000D_spec_sample -> BIKE_7` dev MRR 1.0 on 3 queries
  vs. test 0.639 is the same effect.
- **Run-to-run noise exists.** An earlier accumulation-8 re-run of `FOSSIG -> s1kd-tools-doc` gave test 0.611 vs. the 0.7500 recorded in
  the original run, on identical settings.
- **Not yet done:** zero-shot evaluation with the ga1 pretrain checkpoints (the zero-shot table in `results/README.md` uses the
  accumulation-8 checkpoints), a BM25 baseline, a no-pretraining baseline, and multiple seeds beyond the three-seed S1000D-target check in `results/bs32_vs_ga1/`.

## Reproduce

Delete `runs/{pretrain,finetune,retrieve}/ga1/<pair>` and `results/ga1/<pair>` first if a pair already exists (`pretrain.sh` refuses a
non-empty output dir), and do not launch two loops on the same tag at once.

```bash
cd /workspace/t2xml/santa
export EXTRA_TRAIN_ARGS="--gradient_accumulation_steps 1 --logging_steps 1 --evaluation_strategy steps --eval_steps 5 --report_to none"
for m in $(python3 -c "import json;print('\n'.join(x['name'] for x in json.load(open('model_configs.json'))['models']))"); do
  MODEL_NAME=$m RUN_TAG=/ga1/$m CUDA_VISIBLE_DEVICES=0 bash shell/run_model.sh 2>&1 | tee runs/ga1_$m.log
done
```
