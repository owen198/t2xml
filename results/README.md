# Results
### Pre-training Source Dev Evaluation

| Pre-training | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| MRR | 0.6111 | 0.5208 | 0.7500 | 0.2323 |

Note: Each pretrain checkpoint is evaluated on its own source's dev split.

### Zero-shot Test Evaluation

The table reports each pretrain checkpoint's `eval_test.json` MRR when evaluated directly on the target retrieval test split, no fine-tuning.

| Test target \ Pre-training | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| bike | — | 0.4083 | 0.4083 | 0.4083 |
| s1kd-tools-doc | 0.6111 | — | 0.6111 | 0.6111 |
| fossig | 0.7500 | 0.7500 | — | 0.7500 |
| s1000d-spec | 0.1736 | 0.1739 | 0.1736 | — |

### Fine-tuning Test Evaluation

Same-source cells are intentionally blank because no `*_to_*` self-training runs were executed.
The table reports each run's `eval_test.json` MRR.

| Fine-tuning \ Pre-training | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| bike | — | 0.4083 | 0.4083 | 0.4083 |
| s1kd-tools-doc | 0.6111 | — | 0.7500 | 0.6111 |
| fossig | 0.7500 | 0.7500 | — | 0.7500 |
| s1000d-spec | 0.1760 | 0.1815 | 0.1835 | — |

Fine-tuned raw numbers live in `results/<model_name>/{best_dev_pretrain,best_dev_finetune,eval_test}.json`.
Zero-shot raw numbers live in `results/zero_shot/<model_name>/eval_test.json`.

### Observation
Within each row of the zero-shot and fine-tuning tables, all models use the same test
queries, test corpus, and qrels; only the pretrain source differs. In the fine-tuning table,
models within a row also share the same finetune source and checkpoint selection split.
Identical row values therefore mean that the different pretrain sources did not change the
final rank pattern on that target's test set, or that the target's small/coarse test split
could not reveal the difference. The only places these tables can show a pretraining-source
effect are column-wise differences within a row, such as fine-tuned `S1000D_spec_sample`
MRR: 0.1760 / 0.1815 / 0.1835. Comparing zero-shot vs. fine-tuned rows shows that target
fine-tuning changed little for most targets under this setup.

## Caveats 

- **Per-folder test splits**: FOSSIG (2-doc test), s1kd-tools-doc (3-doc), and even
  BIKE_7 (6-doc) are all small enough that MRR is a coarse, discrete-valued statistic 
  (e.g. FOSSIG's 2-query dev set can only land on MRR ∈ {0.5, 0.75, 1.0}), not a stable
  estimate of retrieval quality. Only `S1000D_spec_sample` (21-doc test) is large enough to
  produce a remotely continuous score. See the split-ratio row in `datasets/README.md`'s
  "Open experiment variables" table.

- **No hard negatives**: all 12 runs use pure in-batch negatives (`train_n_passages 1`),
  same as SANTA's own first-round finetune -- see the corresponding row in
  `datasets/README.md`'s hyperparameters table.


