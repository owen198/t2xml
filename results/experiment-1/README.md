# experiment-1

First SANTA pretrain+finetune run on the current dataset (`pretrain/`, `retrieval/`
as of `fc8fb78`). Hyperparameters actually used: [`config.json`](config.json)
(cross-checked against each run's `training_args.bin`, not just script defaults).

## Status

- **Pretrain**: done. Best checkpoint = the final save, self-referential SDA dev `eval_mrr = 0.7829`. See
  [`best_dev_pretrain.json`](best_dev_pretrain.json).
- **Finetune**: done. Best checkpoint = the final save, retrieval dev `eval_mrr = 0.8856`. See
  [`best_dev_finetune.json`](best_dev_finetune.json).
- **Retrieval eval (test)**: `eval_mrr = 0.8767`. See [`eval_test.json`](eval_test.json).
  Suspected this may be inflated by how the retrieval dataset itself is
  constructed (see the open-variables discussion in `datasets/README.md`) --
  not yet root-caused.

## Files

- `config.json` -- hyperparameters used (pretrain + finetune).
- `best_dev_pretrain.json` -- per-checkpoint pretrain dev MRR + winning checkpoint.
- `best_dev_finetune.json` -- per-checkpoint finetune dev MRR + winning checkpoint.
- `eval_test.json` -- final retrieval eval on the held-out test split.

