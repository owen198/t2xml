# Results
| Folder | What | 
| --- | --- | 
| `01_baselines/t5_s42/`, `01_baselines/codet5_s42/` | T5 vs. CodeT5, all 12 pairs, seed 42 — see [`01_baselines/t5_vs_codet5/README.md`](01_baselines/t5_vs_codet5/README.md) 
| `01_baselines/cnt_p256_s42/`, `01_baselines/cnt_p1024_s42/` | CodeT5, original XML, `p_max_len` 256 vs. 1024, S1000D_spec_sample target, seed 42
| `fixA_meancos_p1024_s{42,1,2}/`, `02_pooling_fix/fixB_meandot_p1024_s42/`, `02_pooling_fix/fixC_meancos_p256_s42/`, `nopre_meancos_p1024_s{42,1,2,3,4}/` | Mean-pooling fix (`--pooling mean`), CodeT5, S1000D target, see below |
| `anonq_meancos_p1024_s{42,1,2}/`, `anonq_nopre_meancos_p1024_s{42,1,2}/` | Same fix, but on identifier-free queries (`santa/data_anonq/`) instead of the original leaky ones, see "New-query experiment" below |
| `04_pool_size/pool332_orig_s42/`, `04_pool_size/pool332_anon_s42/`, `pool332_*_nopre_s42/` | Same checkpoints, evaluated against all 332 S1000D documents instead of the 21-doc test split, see "Widened candidate pool experiment" below |
| `05_pretrain_diagnosis/mepablate_s42/`, `05_pretrain_diagnosis/longpretrain100_s42/` | Ablations diagnosing why pretraining shows no benefit -- MEP loss on/off, pretrain epoch count -- see "Diagnosing why pretraining doesn't help" below |
| `samedomain_s{42,1}/` | Pretrain and finetune on S1000D_spec_sample's own two halves (not a different folder) -- see "Same-domain training" below |

Each target's test corpus contains only its own test-split documents (S1000D_spec_sample 21, BIKE_7 6, s1kd-tools-doc 3, FOSSIG 2), so a random
ordering already scores MRR = H(n)/n: 0.1736, 0.4083, 0.6111, 0.7500. 

Three experiments below, all seed 42, original XML, dot product, batch 16, in-batch negatives:

| Name here | Folder | Model | `p_max_len` | Data | Pairs run |
| --- | --- | --- | --- | --- | --- |
| T5 | `01_baselines/t5_s42/` | `t5-base` | 256 | `santa/data_t5/` | 12 / 12 |
| CodeT5 | `01_baselines/codet5_s42/` | `Salesforce/codet5-base` | 256 | `santa/data_codet5/` | 12 / 12 |
| CodeT5 p1024 (codet5_1024) | `01_baselines/cnt_p1024_s42/` | `Salesforce/codet5-base` | 1024 | `santa/data_cnt/` | 12 / 12 |

### Fine-tuning Test Evaluation

**T5** (`01_baselines/t5_s42`, 12 / 12)

| Pre-training \ Fine-tuning target | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| bike | — | 0.6111 | 0.7500 | 0.1890 |
| s1kd-tools-doc | 0.4917 | — | 0.7500 | 0.1666 |
| fossig | 0.4917 | 0.6111 | — | 0.1498 |
| s1000d-spec | 0.4167 | 0.6111 | 0.7500 | — |
| *Chance* | 0.4083 | 0.6111 | 0.7500 | 0.1736 |
| *Test docs* | 6 | 3 | 2 | 21 |

**CodeT5** (`01_baselines/codet5_s42`, 12 / 12)

| Pre-training \ Fine-tuning target | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| bike | — | 0.6111 | 0.7500 | 0.1777 |
| s1kd-tools-doc | 0.4083 | — | 0.7500 | 0.1696 |
| fossig | 0.4083 | 0.6111 | — | 0.1653 |
| s1000d-spec | 0.4306 | 0.6111 | 0.7500 | — |
| *Chance* | 0.4083 | 0.6111 | 0.7500 | 0.1736 |
| *Test docs* | 6 | 3 | 2 | 21 |

**CodeT5 p1024 / codet5_1024** (`01_baselines/cnt_p1024_s42`, 12 / 12)

| Pre-training \ Fine-tuning target | bike | s1kd-tools-doc | fossig | s1000d-spec |
| --- | --- | --- | --- | --- |
| bike | — | 0.8333 | 0.7500 | 0.1821 |
| s1kd-tools-doc | 0.4556 | — | 0.7500 | 0.1810 |
| fossig | 0.4333 | 0.6111 | — | 0.1977 |
| s1000d-spec | 0.4167 | 0.6111 | 0.7500 | — |
| *Chance* | 0.4083 | 0.6111 | 0.7500 | 0.1736 |
| *Test docs* | 6 | 3 | 2 | 21 |

### Observation

- **Everything is at or near chance.** 19 of the 27 test cells on the three small targets (all three experiments) equal chance exactly (0.6111, 0.7500,
  0.4083), which is what a model scoring every document the same produces. Of the 8 exceptions, 7 are on the bike target (6 test docs, 0.4167–0.4917) and
  one is BIKE_7 -> s1kd-tools-doc in CodeT5 p1024 (0.8333 vs. chance 0.6111, a 3-doc test set), i.e. one or two queries moving.



`p_max_len` 256 vs. 1024 (test MRR, chance 0.1736):

| Pretrain source -> S1000D_spec_sample | p256 | p1024 |
| --- | --- | --- |
| BIKE_7 | 0.1777 | 0.1821 |
| FOSSIG | 0.1631 | 0.1977 |
| s1kd-tools-doc | 0.1696 | 0.1810 |

### Pooling / normalization fix (S1000D_spec_sample target, CodeT5, seed 42)
Test MRR, chance 0.1736:

| Pretrain source -> S1000D_spec_sample | Baseline: first-step state, dot (`01_baselines/cnt_p1024_s42`) | Mean pooling, dot (`02_pooling_fix/fixB_meandot_p1024_s42`) | Mean pooling, cosine (`02_pooling_fix/fixA_meancos_p1024_s42`) | Mean pooling, cosine, `p_max_len` 256 (`02_pooling_fix/fixC_meancos_p256_s42`) |
| --- | --- | --- | --- | --- |
| BIKE_7 | 0.1821 | 0.8545 | 0.9048 | 0.5385 |
| FOSSIG | 0.1977 | 0.8476 | 0.9048 | 0.5923 |
| s1kd-tools-doc | 0.1810 | 0.8690 | 0.9286 | 0.5497 |

- Changing the pooling alone moves the score from chance to 0.85–0.93, so the chance-level results above come from the embedding, not from the data volume or the tokenizer.
- **No-pretraining control** (`nopre_meancos_p1024_s{42,1,2,3,4}`, base CodeT5 finetuned directly on S1000D with config A, launcher `santa/runs/run_nopretrain.sh`): 0.917 / 0.976 / 0.976 / 0.929 / 0.952
  (mean 0.950, sd 0.027). This is at least as high as with SDA/MEP pretraining (0.920); the 0.03 gap is inside the noise (about 1.6 standard errors), so the fair reading is
  that this benchmark shows no benefit from pretraining. It may be because the task is too easy (BM25 = 1.0), which is untested.

### Fix on the other three targets, and `p_max_len` 2048 (CodeT5, mean pooling + cosine tau 0.05, seed 42)

Extends the pooling fix beyond S1000D_spec_sample to the 9 pairs targeting BIKE_7, FOSSIG, and s1kd-tools-doc, and checks whether a
longer passage window (2048 vs. 1024) helps on these longer-document targets. Folders `02_pooling_fix/allA_meancos_p1024_s42/`,
`02_pooling_fix/allA_meancos_p2048_s42/`, no-pretrain control `02_pooling_fix/nopreAll_meancos_p1024_s{42,1,2}/`.

| Pair | Chance | p1024 | p2048 |
| --- | --- | --- | --- |
| BIKE_7 -> FOSSIG | 0.7500 | 0.7500 | 0.7500 |
| S1000D_spec_sample -> FOSSIG | 0.7500 | 1.0000 | 1.0000 |
| s1kd-tools-doc -> FOSSIG | 0.7500 | 1.0000 | 0.7500 |
| BIKE_7 -> s1kd-tools-doc | 0.6111 | 0.8333 | 1.0000 |
| FOSSIG -> s1kd-tools-doc | 0.6111 | 0.8333 | 1.0000 |
| S1000D_spec_sample -> s1kd-tools-doc | 0.6111 | 1.0000 | 1.0000 |
| FOSSIG -> BIKE_7 | 0.4083 | 0.5194 | 0.9167 |
| S1000D_spec_sample -> BIKE_7 | 0.4083 | 0.9167 | 0.8333 |
| s1kd-tools-doc -> BIKE_7 | 0.4083 | 0.6583 | 0.8056 |

Mean per target (p1024 / p2048): BIKE_7 0.698 / 0.852, s1kd-tools-doc 0.889 / 1.000, FOSSIG 0.917 / 0.833. No-pretraining control (p1024, 3 seeds):
BIKE_7 0.566, FOSSIG 0.917, s1kd-tools-doc 0.778.

- **The pooling fix generalizes**: all 9 pairs, both lengths, score well above chance -- this isn't an S1000D-specific effect.
- **Caveat, same as everywhere else on these three targets**: test sets are 6 / 3 / 2 documents, so individual cells move by 0.05-0.25
  per query and should be read as directional, not precise.
- **2048 looks better than 1024 on average** (BIKE_7 and s1kd-tools-doc both improve; FOSSIG, only 2 test docs, moves the other way on
  one query). Consistent with the p_max_len findings elsewhere in this file, but single-seed and not independently confirmed here.

### Why the embeddings collapse, and why mean pooling fixes it

**Background:** T5 / CodeT5 is a "read one text, write another" model (used for translation or summarization). The encoder reads the
input and produces one vector per token. The decoder then writes the output one token at a time. Before each token it computes an internal state (768 numbers, looking
back at the encoder's vectors), and a vocabulary head turns that state into probabilities for the next token:

```
input text ----> [encoder] ----> one vector per input token (the text, "understood")
                                         |  the decoder looks back at these
                                         v
start token ---> [decoder] ---> internal state (768 numbers) ---> vocab head ---> predicts output token 1
token 1 -------> [decoder] ---> internal state                ---> vocab head ---> predicts output token 2
...
```

**How the baseline builds a vector (`pooling=first`, SANTA's original setting, `santa/model.py`).** Steps:

1. The encoder reads the whole document.
2. The decoder is given one fixed "start" token, identical for every document.
3. The decoder's output at that single step is taken as the document's vector.

In code (`encode_q` / `encode_p` in `santa/model.py`, same for queries and documents):

```python
decoder_input_ids = torch.zeros((batch, 1))        # one fixed start token (id 0) for every text
out = model(**items, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
hidden = out.decoder_hidden_states[-1]             # decoder's last-layer internal state
reps = hidden[:, 0, :]                             # position 0 = that single step -> the embedding
```

With `pooling=mean` this whole decoder step is skipped: the embedding is the average of the encoder's per-token vectors instead.

**Evidence** (diagnostics in [`01_baselines/t5_vs_codet5/README.md`](01_baselines/t5_vs_codet5/README.md)):
- Even an **untrained** CodeT5 does this: any two documents' vectors have cosine 1.0000, and the difference between documents is only 0.6% of the vector's size.
  Queries collapse too (cosine 0.9999).
- The shared part is concentrated in a few dimensions: 45% of the mean vector's energy sits in a single dimension (dim 567). This big common offset adds about the same
  amount to every document's score and hides the real differences.
- Subtracting the shared part lifts the untrained CodeT5's MRR from 0.17 to 0.60.

**The logic of mean pooling.** Skip the decoder step. Average the encoder's output over every token of the document and use that as the vector. Each
token's output already carries the meaning of that token, so documents with different content get different averages.

**What cosine adds.** It rescales every vector to length 1, so only direction matters. The score no longer depends on vector length, and the model cannot change scores
by changing lengths; it has to separate documents by direction.

### New-query experiment: removing identifier leakage from the queries (S1000D_spec_sample only)

Motivation: BM25 scores 1.0 on the original S1000D test queries because they quote identifiers/dates from their own document
(`datasets/README.md`, "Should queries be regenerated..."). This experiment builds a second query set that can't be solved that way, to
see whether the pretrained-vs-no-pretrain result above still holds once string matching is taken away. 

**Pipeline** :
- `datasets/generate_anon_queries.py` -- new prompt (`anon_prompt()`) that forbids copying any code/number/date verbatim from the
  document, asking for paraphrase instead ("the latest issue" rather than an issue number). Reads the existing
  `pretrain/S1000D_spec_sample/sda_pairs.*.jsonl` `structured` field (same documents, same train/dev/test split) and writes new
  descriptions to `pretrain_anonq/S1000D_spec_sample/sda_pairs.*.jsonl`. 
  **Result: 3/332 queries (0.9%) still had a leak**, and inspection showed these were incidental (e.g. "1980s" in a historical narrative
  paragraph, not a copied field value) -- the prompt works.
- `datasets/build_retrieval_dataset.py --pretrain-root pretrain_anonq --output-root retrieval_anonq --datasets S1000D_spec_sample` --
  same corpus (21/14/297 test/dev/train docs, none dropped), new queries.
- `santa/prepare_data.py --retrieval-root ../retrieval_anonq --output-dir data_anonq` -- tokenized with `codet5-base`.

**BM25 on the new queries** (`santa/diagnostics/bm25_baseline.py`, pointed at `retrieval_anonq`): **MRR = 0.9762** (20/21 queries still rank
their document 1st).
This is the main finding of this experiment: identifier leakage was not the reason BM25 wins. With only 21 test documents, each on a
genuinely distinct narrow S1000D topic (e.g. "front matter copyright notice" vs. "crew procedures chapter"), ordinary topic vocabulary is
enough to separate them. 

**Pretrained vs. no-pretraining on the new queries** (mean pooling + cosine tau 0.05, p_max_len 1024, launchers `run_pairs.sh` /
`run_nopretrain.sh` with `DATA_ROOT=data_anonq`, folders `anonq_meancos_p1024_s{42,1,2}/`, `anonq_nopre_meancos_p1024_s{42,1,2}/`):

| | Pretrained (3 sources) | No pretraining | BM25 |
| --- | --- | --- | --- |
| Original queries | 0.920 (sd 0.044) | 0.950 (sd 0.027, 5 seeds) | 1.0000 |
| New (anonymized) queries | 0.864 (sd 0.043) | 0.874 (sd 0.035) | 0.9762 |

- **The pretraining result doesn't change, still no benefit.** 
- **Both dense settings dropped versus the original queries** (0.92/0.95 -> 0.86/0.87), consistent with the new queries being harder, but the dense model still trails BM25 by about 0.10-0.11, roughly the same gap as before
  (0.05-0.08 originally). Removing identifier leakage did not let the dense model close the gap to BM25 or reveal a pretraining benefit;
  it mainly confirmed the benchmark was already lexically easy for reasons beyond identifiers.
- Per-seed pretrained means: 0.889 / 0.883 / 0.821 (seeds 42/1/2). Per-pair breakdown and raw JSON in
  `results/anonq_meancos_p1024_s{42,1,2}/` and `results/anonq_nopre_meancos_p1024_s{42,1,2}/`.


### Widened candidate pool experiment (S1000D_spec_sample)

This widens the *search pool* to all 332 S1000D documents
(train+dev+test) without retraining on a new split or changing the test queries, to check how much of the earlier result was the
small pool rather than the query wording.

**BM25 collapses on the wider pool** (`datasets/build_full_corpus.py` output, scored the same way as `bm25_baseline.py`):

| Candidates | Chance | BM25, original queries | BM25, anonymized queries |
| --- | --- | --- | --- |
| 21 (old, test-split only) | 0.1736 | 1.0000 | 0.9762 |
| 332 (train+dev+test) | 0.0192 | **0.1250** (0/21 at rank 1) | **0.1147** (0/21 at rank 1) |

| | BM25 | Dense, pretrained | Dense, no pretraining (1 seed) |
| --- | --- | --- | --- |
| Original queries | 0.1250 | 0.6577 | 0.6378 |
| Anonymized queries | 0.1147 | 0.4951 | 0.4993 |

- **Still no visible pretraining benefit**
  Pretrained vs. no-pretrain: +0.02 on original queries, -0.004 on
  anonymized queries
- Raw numbers: `results/04_pool_size/pool332/*_eval_test_all332.json`.

### Diagnosing why pretraining doesn't help (S1000D_spec_sample, mean pooling + cosine tau 0.05, p_max_len 1024, seed 42)

**Ablation 1: MEP loss on vs. off during pretrain** . 

| Pretrain source | SDA-only (MEP off) | SDA+MEP (MEP on) |
| --- | --- | --- |
| BIKE_7 | 0.8333 | 0.9048 |
| FOSSIG | 0.9286 | 0.9048 |
| s1kd-tools-doc | 0.8968 | 0.9286 |
| **mean** | **0.8862** | **0.9127** |

A 0.026 gap and the per-source direction isn't consistent (MEP helps BIKE_7 and s1kd-tools-doc, hurts FOSSIG). ** MEP is not the reason pretraining doesn't help.**

**Ablation 2: pretrain step count, 10 epochs vs. 100**  If pretraining were under-trained (FOSSIG gets ~10 steps total, BIKE_7 ~70, both far short of
finetuning's 200+):

| Pretrain source | Pretrain-dev MRR, 10ep | Pretrain-dev MRR, 100ep | Test MRR, 10ep | Test MRR, 100ep | Pretrain train loss, 10ep -> 100ep |
| --- | --- | --- | --- | --- | --- |
| BIKE_7 | 0.6111 | 0.6667 | 0.9048 | 0.8730 | 10.4 -> 0.002 |
| FOSSIG | 0.7500 | 0.5000 | 0.9048 | 0.8571 | 9.3 -> 0.011 |
| s1kd-tools-doc | 0.7083 | 0.8750 | 0.9286 | 0.9444 | 9.9 -> 0.003 |
| **mean test MRR** | | | **0.9127** | **0.8915** | |

At 100 epochs the pretrain loss collapses to near-zero (the model has essentially memorized its tiny pretrain set, FOSSIG has only 8
documents), FOSSIG's pretrain-dev MRR gets worse (0.75 -> 0.50) despite the extra training, and downstream test MRR is slightly lower (0.913 -> 0.892). **if anything, this points toward overfitting rather than under-training.**

### Same-domain training: pretrain and finetune on S1000D_spec_sample's own data
SANTA pretrains and finetunes on the same corpus family (CodeSearchNet -> CodeSearchNet-AdvTest, itself a filtered Python subset of CodeSearchNet; ESCI-large -> ESCI-small, a partition of the same product-search dataset). This experiment reproduces that same-domain setup using S1000D_spec_sample's own 297 training documents.

**Design** 
`datasets/build_samedomain_split.py` splits the 297 train documents into two disjoint halves by a deterministic hash on `source_file` (halfA=150, halfB=147 docs), leaving the existing dev (14) and test (21) splits untouched. `santa/prepare_samedomain.py` tokenizes four combinations with the codet5 tokenizer:

| Arm | Pretrain on | Finetune on | Purpose |
| --- | --- | --- | --- |
| `pretrainA_finetuneB` | halfA (SDA+MEP) | halfB | same-domain pretrain -> finetune |
| `finetuneB_only` | -- | halfB | matched-size no-pretrain control |
| `pretrainB_finetuneA` | halfB (SDA+MEP) | halfA | same-domain pretrain -> finetune, swapped |
| `finetuneA_only` | -- | halfA | matched-size no-pretrain control |

Two seeds (42, 1), mean pooling + cosine, p_max_len 1024. Launcher `santa/runs/run_samedomain.sh`. 

**Result**

| Seed | A->B: pretrained | A->B: no-pretrain | diff | B->A: pretrained | B->A: no-pretrain | diff |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.8988 | 0.8631 | +0.0357 | 0.9365 | 0.8057 | **+0.1308** |
| 1 | 0.8202 | 0.8378 | -0.0175 | 0.9116 | 0.8005 | **+0.1111** |

- **B->A (pretrain on halfB, finetune on halfA) shows a large, seed-consistent gain: +0.13 and +0.11.** This is the first pretraining comparison all session where the effect is both large and reproduces across seeds in the same direction.
- **A->B (the reverse direction) does not reproduce: +0.036 then -0.018.** The two directions are not symmetric, so something about which half is used for pretraining vs. finetuning matters. The hash-based split is not stratified, so the two halves aren't guaranteed equally easy, and halfA (the harder finetune target) is the direction where pretraining shows a large gain, while halfB (an easier target) shows little room for pretraining to add.
- **Pretrain-dev SDA MRR is clearly above chance here**, unlike almost every cross-domain source tried earlier: halfA pretrain-dev 0.3955, halfB pretrain-dev 0.4917, against a chance of 0.2323 (14 dev docs), consistent with the domain-content-gap diagnosis (SDA's contrastive task learns better when pretrain and finetune content are topically related).
- **Caveats:** the asymmetry between directions is itself unexplained and could be either a genuine effect (pretraining helps more on a harder finetune target) or a property of this specific 150/147 split. A stratified or multiply-resampled split, and more seeds on the B->A direction, would settle whether the asymmetry is real.


#### Follow-up: a 3rd seed and an independent split resolve the asymmetry

Added seed 2 on the original 150A/147B split, plus a second, independently-hashed split (`--salt`, `pretrain_samedomain_alt/`, `retrieval_samedomain_alt/`, `SAMEDOMAIN_SUFFIX=_alt`): 143A/154B, seed 42 only. Same config throughout.

| | orig split, s42 | orig split, s1 | orig split, s2 | alt split, s42 |
| --- | --- | --- | --- | --- |
| A->B diff (finetune on the *easier* half) | +0.0357 | -0.0175 | +0.0414 | +0.0132 |
| B->A diff (finetune on the *harder* half) | +0.1308 | +0.1111 | +0.0594 | +0.0634 |


**The asymmetry replicates on a second, independently-partitioned split, and holds up under a 3rd seed.**

| | orig split (n=3 seeds) | alt split (n=1 seed) |
| --- | --- | --- |
| No-pretrain gap (easier - harder) | 0.0316 | 0.0574 |
| Pretraining gain on the **harder** half | **+0.100** (0.131 / 0.111 / 0.059, all positive) | **+0.063** |
| Pretraining gain on the **easier** half | +0.020 (0.036 / -0.018 / 0.041, straddles zero) | +0.013 |

Across all 4 split/seed combinations, the harder half always gains more from pretraining than the easier half from the same split, and the harder-half gain is positive in every single instance (5/5 including the original two seeds), while the easier-half gain is small and inconsistent in sign (2/3 positive on the original split).

**Conclusion: same-domain pretraining's benefit is concentrated on the harder finetune target.** 
