# T5 vs. CodeT5

## Test MRR (chance = mean reciprocal rank of a random ordering of the test corpus, H(n)/n)

| Pair | Test docs | Chance | T5 | CodeT5 |
| --- | --- | --- | --- | --- |
| BIKE_7 -> FOSSIG | 2 | 0.7500 | 0.7500 | 0.7500 |
| BIKE_7 -> S1000D_spec_sample | 21 | 0.1736 | 0.1890 | 0.1777 |
| BIKE_7 -> s1kd-tools-doc | 3 | 0.6111 | 0.6111 | 0.6111 |
| FOSSIG -> BIKE_7 | 6 | 0.4083 | 0.4917 | 0.4083 |
| FOSSIG -> S1000D_spec_sample | 21 | 0.1736 | 0.1498 | 0.1653 |
| FOSSIG -> s1kd-tools-doc | 3 | 0.6111 | 0.6111 | 0.6111 |
| S1000D_spec_sample -> BIKE_7 | 6 | 0.4083 | 0.4167 | 0.4306 |
| S1000D_spec_sample -> FOSSIG | 2 | 0.7500 | 0.7500 | 0.7500 |
| S1000D_spec_sample -> s1kd-tools-doc | 3 | 0.6111 | 0.6111 | 0.6111 |
| s1kd-tools-doc -> BIKE_7 | 6 | 0.4083 | 0.4917 | 0.4083 |
| s1kd-tools-doc -> FOSSIG | 2 | 0.7500 | 0.7500 | 0.7500 |
| s1kd-tools-doc -> S1000D_spec_sample | 21 | 0.1736 | 0.1666 | 0.1696 |

S1000D_spec_sample target, mean of 3 pairs: T5 0.1685, CodeT5 0.1708, chance 0.1736. Mean margin over chance across all 12 pairs: T5 +0.013,
CodeT5 +0.001.

## Observations
- **Most cells equal chance exactly** (0.7500, 0.6111, and BIKE_7's 0.4083). An exact match to H(n)/n is what a model that gives every document the same score
  produces when ties are broken by corpus order, so these targets carry no information about the model. Pretrain dev MRR is at chance too
  (e.g. 0.6111 on a 3-document dev pool).
- **T5 is not clearly better than CodeT5.** The few cells where T5 is above chance (FOSSIG -> BIKE_7 and s1kd-tools-doc -> BIKE_7, 0.4917;
  BIKE_7 pretrain dev 0.7778) are 6-query or 3-query sets, one query moving. Its pretrain loss does start much lower than CodeT5's
  (S1000D pretrain: first logged loss 387 vs. 3715), so T5 optimizes more easily, but that has not translated into retrieval.

## Embedding diagnostic (S1000D_spec_sample test set: 21 docs, 21 queries)

Scripts: `santa/diagnostics/embedding_spread.py` (encodes exactly like `index_xml.py`) and `santa/diagnostics/bm25_baseline.py`.
"rel. spread" = norm of the per-dimension std across items divided by the norm of the mean vector (0 = every item has the same embedding).

| Model | \|p\| | doc-doc cos | rel. spread doc | q-q cos | rel. spread q | Test MRR |
| --- | --- | --- | --- | --- | --- |
| CodeT5 untrained base | 145 | 1.0000 | 0.0057 | 0.9999 | 0.0093 | 0.1739 |
| CodeT5 untrained base, p_max_len 1024 | 145 | 1.0000 | 0.0026 | 0.9999 | 0.0093 | 0.1739 |
| CodeT5, BIKE_7 pretrain only | 167 | 0.9999 | 0.0111 | 0.9992 | 0.0298 | 0.1672 |
| CodeT5, BIKE_7 -> S1000D | 168 | 0.9999 | 0.0099 | 0.9875 | 0.1125 | 0.1777 |
| CodeT5, BIKE_7 -> S1000D, p_max_len 1024 | 167 | 1.0000 | 0.0021 | 0.9856 | 0.1208 | 0.1821 |
| T5 untrained base | 24 | 0.9997 | 0.0270 | 0.9991 | 0.0378 | 0.1733 |
| T5, BIKE_7 -> S1000D | 10 | 1.0000 | 0.0068 | 0.9697 | 0.1787 | 0.1890 |
| **BM25 (same 21 docs)** | | | | | | **1.0000** |

- **The representation is collapsed before any training.** The untrained CodeT5 already gives every document (cos 1.0000, spread 0.6%) *and* every query
  (cos 0.9999, and queries are diverse free text) almost the same vector: a large shared component plus a tiny variation. Training does not change
  that for documents (spread stays 0.2-1%, and is smaller at p_max_len 1024). It does spread queries out (0.01 -> 0.11-0.18) and shrinks their norm
  (145 -> 42-54), i.e. it mostly lowers the score scale rather than making documents distinguishable. With near-constant document vectors the ranking
  is decided by tiny differences that are the same for every query, hence chance-level MRR.
- **Scale differs between models.** CodeT5 scores range over tens (softmax saturates); T5's range over about 0.1-0.4 (softmax almost uniform, weak gradient).
  Both fail, for opposite reasons.
- **BM25 scores 1.0000** (every query ranks its document first), so the documents are easily separable lexically. The queries quote identifiers from their
  own documents (see the query-leakage row in `datasets/README.md`), so this benchmark on S1000D mostly rewards string matching. Any dense number
  has to be read against BM25 = 1.0, not only against chance.

### Centering check (`santa/diagnostics/centering.py`)

Subtract the per-set mean from the document and the query embeddings (test-set mean, so this is a diagnostic, not a valid retrieval method), then rank by cosine.
Same 21 queries, single seed.

| Model | Raw dot MRR | Centered-cosine MRR | Energy of mean vector in top 1 / 5 / 20 of 768 dims |
| --- | --- | --- | --- |
| CodeT5 untrained | 0.1739 | **0.5992** | 0.45 / 0.69 / 0.73 (dim 567) |
| T5 untrained | 0.1733 | 0.2479 | 0.26 / 0.86 / 0.98 |
| CodeT5 BIKE_7 -> S1000D | 0.1777 | 0.2536 | 0.02 / 0.06 / 0.16 |
| CodeT5 BIKE_7 -> S1000D, p_max_len 1024 | 0.1821 | 0.2127 | 0.02 / 0.06 / 0.16 |

The untrained CodeT5 does carry usable signal; it is hidden under a shared offset concentrated in a few outlier dimensions. Finetuning with the raw
dot-product loss lowered the centered score (0.60 -> 0.25) instead of exposing it. The 0.60 may partly reflect lexical cues in the queries (BM25 = 1.0).
