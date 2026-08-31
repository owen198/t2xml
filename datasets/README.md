# S1000D sample datasets

This directory contains four S1000D Common Source Database (CSDB) samples used for structure-aware XML research. Keep each folder self-contained — references among its data modules, publications, and data-management lists are relative to the other files in that folder.

## Dataset inventory

| Folder | Source and purpose | Local inventory |
| --- | --- | --- |
| [`BIKE`](BIKE/) | **S1000D Bike sample data set, Issue 6.** A fictional bicycle product dataset with descriptive, procedural, and BREX (business-rule exchange) content. Example only — does **not** demonstrate S1000D conformance. | 106 XML files: 106 data modules (`DMC-*`). |
| [`FOSSIG`](FOSSIG/) | [FOSSIG](https://github.com/kibook/FOSSIG) — applies S1000D to free-software documentation, with reusable business rules and a generic Standard Numbering System (SNS). | 22 files: 10 data modules, 1 publication module, 1 data-management list, and 10 supporting configuration/illustration files. |
| [`S1000D spec sample`](S1000D%20spec%20sample/) | [kibook/S1000D](https://github.com/kibook/S1000D), a sample CSDB representing the S1000D Issue 4.2 specification. | 333 files: 330 data modules, 1 publication module, 1 data-management list, and 1 defaults file. |
| [`s1kd-tools-doc`](s1kd-tools-doc/) | [kibook/s1kd-tools-doc](https://github.com/kibook/s1kd-tools-doc) — documents the `s1kd-tools` toolset, produced with [s1kd-tools](https://github.com/kibook/s1kd-tools) and S1000D XSL stylesheets. | 50 files: 44 data modules, 1 publication module, 1 data-management list, 2 defaults/type configuration files, and 2 PNG illustrations. |

**Total local inventory:** 511 files, including 496 XML files: 490 data modules, 3 publication modules, and 3 data-management lists.

## S1000D file types

The filename prefixes identify the principal XML object types:

- `DMC-` — data module code; an individual S1000D information module. In these samples, XML data modules use the root element `dmodule`.
- `PMC-` — publication module code; an ordered publication structure that references data modules. Its XML root element is `pm`.
- `DML-` — data-management list code; a list of managed S1000D objects. Its XML root element is `dml`.
- `ICN-` — information control number; associated graphic or other non-XML information object.

Configuration/support files such as `.defaults`, `.dmtypes`, `.fmtypes`, and `.icncatalog` are dataset-specific inputs used by the associated S1000D tooling and publication workflows.

## S1000D schemas (`xsd/`)

[`xsd/`](xsd/) holds the 32 flat S1000D Issue 6 XML Schema modules (`descript.xsd`, `proced.xsd`, `pm.xsd`, `dml.xsd`, `brex.xsd`, etc.), covering every information-kind content model plus the shared/common-construct and cross-reference-table schemas. `preprocess.py` uses them as the authoritative list of S1000D element names (see below). 

## Use and limitations

- The Bike data set is an **Issue 6 sample**, suitable for studying XML structure, references, and representative content types, but not for claiming tool or model conformance.
- Before schema validation, select the XSD and BREX rules appropriate to the dataset's S1000D issue and project rules. XML well-formedness alone does not establish S1000D validity.

## Preprocessing scripts

- **`preprocess.py`** — Stage 1: walks every `.XML` file under `BIKE/`, `FOSSIG/`, `S1000D spec sample/`, and `s1kd-tools-doc/` and emits `(description, XML snippet)` positive pairs.
  - **Element whitelist**: every element name declared across the schemas in [`xsd/`](xsd/) is a candidate chunk boundary, cached at `../.preprocess_cache/s1000d_element_whitelist.json`.
  - **Chunking**: the tree is walked recursively; every whitelisted element is emitted as its own (nested) snippet.
  - **Description generation**: snippets with ≥40 characters of text go to the **LLM tier** (one cached, checkpointed Claude Haiku 4.5 call per unique snippet; falls back to an extractive summary if uncalled). Attribute-only metadata (`dmCode`, `security`, `issueInfo`, ...) goes to the **template tier** — hand-written templates, or a generic tag+attributes fallback.
  - **Filtering**, modeled on [CodeSearchNet](https://arxiv.org/abs/1909.09436)'s cleaning heuristics: drop trivial snippets (no attributes/text/children); cap exact-duplicate snippets (`--dedup-cap`, default 20); cap near-duplicate siblings under the same parent (`--sibling-cap`, default 20; their children are dropped too); drop descriptions under 3 words.
  - **Output**: `../pretrain/sda_pairs.{train,dev,test}.jsonl`. Each record: `{"structured", "text", "source_file", "element", "xpath", "tier", "generated_by"}`.
  - Run: `python3 preprocess.py [--input-dirs ...] [--output-dir ...] [--dedup-cap N] [--max-llm-calls N] [--llm-concurrency N] [--llm-model ...]`. Default safety cap is 500 live calls/run; a full from-scratch run needs `--max-llm-calls 20000`+ (~19,800 unique LLM-tier snippets).

- **`build_xml_entity.py`** — Stage 2: builds Masked Entity Prediction (MEP) examples from `sda_pairs.*.jsonl`, following [SANTA](https://arxiv.org/abs/2305.19912).
  - **Entities**: identifier-like attribute values (`modelIdentCode`, `systemCode`, `issueNumber`, ...) and capitalized/technical tokens in prose text (`techName`, `enterpriseName`, ...) — never element tags, so structure stays intact.
  - **Detection**: attributes matched by name pattern (contains "code"/"ident"/"number"/"classification"/"isocode", plus `inWork`/`year`/`month`/`day`); text entities via a capitalization + digit-required code-pattern heuristic (avoids flagging English compounds/XPath syntax), optionally supplemented by NLTK NNP/NNPS tagging (local `.venv/`, not a default dependency).
  - **Masking**: each distinct entity string gets one shared sentinel token (`<extra_id_0>`, ...) everywhere it appears in a snippet.
  - **Downsampling** (`--mask-ratio`, default 0.5): only this fraction of a snippet's unique entities is masked, capped at 99.
  - **Output**: `../pretrain/mep_pairs.{train,dev,test}.jsonl`. Each record: `{"structured", "masked_structured", "label", "num_entities", "source_file", "element", "xpath"}`. Snippets with zero entities are dropped.
  - Run: `python3 build_xml_entity.py [--input-dir ...] [--output-dir ...] [--mask-ratio F] [--random-seed N]`.

- **`build_retrieval_dataset.py`** — Stage 3: builds a retrieval finetuning/evaluation dataset from `sda_pairs.*.jsonl`, mirroring SANTA's Adv/ESCI-small benchmarks (Table 1 of the paper).
  - **Cleanup vs. Stage 1**: documents are deduped to one corpus entry per unique content hash, and generic fallback descriptions that map to many different documents (e.g. `"dm ref element."` → 355 docs in train) are dropped as ambiguous queries — "the correct document" needs to be well-defined for retrieval eval, unlike Stage 1's alignment pretraining.
  - **Splits**: reuses Stage 1's file-level train/dev/test assignment.
  - **Output**: `../retrieval/{corpus,queries}.{train,dev,test}.jsonl` and `qrels.{train,dev,test}.tsv`, in the BEIR/OpenMatch layout. `corpus.jsonl`: `{"docid", "structured", "element", "xpath", "source_file"}`. `queries.jsonl`: `{"qid", "text", "element", "xpath", "source_file"}`. `qrels.tsv`: `query-id\tcorpus-id\tscore` (always `1`).
  - Run: `python3 build_retrieval_dataset.py [--input-dir ...] [--output-dir ...]`.

## Preprocessing data statistics

**Stage 1 — Structured Data Alignment pairs** (`sda_pairs.*.jsonl`, from `preprocess.py`):

| Split | Total pairs | Template tier | LLM/prose tier |
| --- | --- | --- | --- |
| Train | 42,231 | 23,291 | 18,940 |
| Dev | 2,009 | 1,413 | 596 |
| Test | 788 | 439 | 349 |

All LLM-tier pairs carry a real Claude-written description (`generated_by: "llm"`/`"llm_cached"`) — none fell back to the extractive summary.

**Stage 2 — Masked Entity Prediction pairs** (`mep_pairs.*.jsonl`, from `build_xml_entity.py`, `--mask-ratio 0.5` default):

| Split | SDA pairs in | Dropped (no entities) | MEP pairs out | Avg entities/example | Entity share of tokens |
| --- | --- | --- | --- | --- | --- |
| Train | 42,231 | 15,199 | 27,032 | 3.75 | 3.8% |
| Dev | 2,009 | 904 | 1,105 | 3.72 | 4.5% |
| Test | 788 | 229 | 559 | 4.20 | 5.9% |

MEP pairs out is ~4% lower than before a code-pattern fix (28,117→27,032 in train): snippets whose only "entity" was a false-positive hyphenated word or XPath keyword now correctly drop instead of keeping a bad mask target.

"Entity share of tokens" (analog of SANTA's Table 5 "Entities" column) is the fraction of tokens in the unmasked `structured` XML identified as entities. It runs lower than SANTA (15–29%) mainly because XML markup dilutes the token count. A handful of large whole-`dmodule` snippets hit the 99-entity cap.


**SDA pairs by folder and split:**

| Folder | Train | Dev | Test | Total |
| --- | --- | --- | --- | --- |
| BIKE | 23,578 | 982 | 286 | 24,846 |
| FOSSIG | 450 | 130 | 113 | 693 |
| S1000D spec sample | 10,682 | 191 | 154 | 11,027 |
| s1kd-tools-doc | 7,521 | 706 | 235 | 8,462 |
| **Total** | | | | **45,028** |

**MEP pairs and entity share by folder:**

| Folder | MEP pairs | Entity share of tokens |
| --- | --- | --- |
| BIKE | 14,935 | 3.2% |
| FOSSIG | 502 | 8.2% |
| S1000D spec sample | 8,744 | 6.1% |
| s1kd-tools-doc | 4,515 | 3.0% |

FOSSIG has the highest entity share despite being the smallest folder — its SNS/business-rule metadata packs more identifiers per snippet than BIKE's prose-heavy procedural content. BIKE dominates pair *counts* simply by being the largest, most deeply-tagged folder.

**Stage 3 — Retrieval finetuning/evaluation dataset** (`retrieval/{corpus,queries}.*.jsonl` + `qrels.*.tsv`, from `build_retrieval_dataset.py`), the analog of Table 1:

| Split | SDA pairs in | Query-Doc pairs | Corpus (unique docs) | Ambiguous queries dropped |
| --- | --- | --- | --- | --- |
| Train | 42,231 | 25,146 | 25,987 | 388 |
| Dev | 2,009 | 1,659 | 1,714 | 20 |
| Test | 788 | 720 | 750 | 13 |

## Provenance

The `FOSSIG`, `S1000D spec sample`, and `s1kd-tools-doc` folders correspond to CSDB examples linked from the [s1kd-tools repository](https://github.com/kibook/s1kd-tools). The Bike description above is the accompanying source notice for the locally included S1000D Bike Issue 6 sample data. Consult the upstream repositories and the S1000D specification for licensing, current releases, schemas, and authoritative conformance guidance.
