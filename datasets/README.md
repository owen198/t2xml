# S1000D sample datasets

This directory contains four S1000D Common Source Database (CSDB) samples used for structure-aware XML research. They are examples of real S1000D document structures. Retain each folder as a self-contained dataset: S1000D references among data modules, publications, and data-management lists are relative to the other files in its own folder.

## Dataset inventory

| Folder | Source and purpose | Local inventory |
| --- | --- | --- |
| [`BIKE`](BIKE/) | **S1000D Bike sample data set, Issue 6.** A fictional bicycle product dataset. It includes examples of descriptive, procedural, and BREX (business-rule exchange) content. The dataset is supplied as an example only; processing it does **not** demonstrate S1000D conformance. | 106 XML files: 106 data modules (`DMC-*`). |
| [`FOSSIG`](FOSSIG/) | [FOSSIG](https://github.com/kibook/FOSSIG), the Free Open Source Software Interest Group sample. It demonstrates applying S1000D to free-software documentation and includes work on reusable business rules and a generic Standard Numbering System (SNS). | 22 files: 10 data modules, 1 publication module, 1 data-management list, and 10 supporting configuration/illustration files. |
| [`S1000D spec sample`](S1000D%20spec%20sample/) | [kibook/S1000D](https://github.com/kibook/S1000D), a sample CSDB representing the S1000D Issue 4.2 specification. | 333 files: 330 data modules, 1 publication module, 1 data-management list, and 1 defaults file. |
| [`s1kd-tools-doc`](s1kd-tools-doc/) | [kibook/s1kd-tools-doc](https://github.com/kibook/s1kd-tools-doc), an example S1000D publication produced with [s1kd-tools](https://github.com/kibook/s1kd-tools) and S1000D XSL stylesheets. Its modules document the `s1kd-tools` toolset. | 50 files: 44 data modules, 1 publication module, 1 data-management list, 2 defaults/type configuration files, and 2 PNG illustrations. |

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
  - **Element whitelist**: every element name declared across the schemas in [`xsd/`](xsd/) is a candidate chunk boundary. The derived list is cached at `../.preprocess_cache/s1000d_element_whitelist.json`.
  - **Chunking**: the tree is walked recursively; every element whose tag is in the whitelist is emitted as its own snippet, nested (so e.g. `identAndStatusSection` and its child `dmAddress` are both emitted separately).
  - **Description generation**: a snippet with ≥40 characters of text content is routed to the **LLM tier** (one Claude Haiku 4.5 call per unique snippet; falls back to an extractive summary if no key is set, the call fails, or the `--max-llm-calls` safety cap is hit before that snippet's turn). Everything else (attribute-only metadata like `dmCode`, `security`, `issueInfo`) is routed to the **template tier** — hand-written templates for the common S1000D metadata elements, a generic tag+attributes fallback otherwise. LLM-tier resolution is deferred until after the full corpus is walked (so exact-dup/sibling caps are already final) and then dispatched concurrently (`--llm-concurrency`, default 8) against a cache keyed by snippet content, checkpointed to disk every 200 calls so an interrupted run doesn't lose progress.
  - **Filtering**, modeled on the [CodeSearchNet](https://arxiv.org/abs/1909.09436) corpus-cleaning heuristics:
    - *Trivial snippets dropped* : a non-prose element with no attributes, no text, and no children (e.g. a bare `<unverified/>` marker) carries ~no information on its own and is skipped.
    - *Exact-duplicate cap* (`--dedup-cap`, default 20): the same byte-identical snippet recurring across many files (e.g. `<security securityClassification="01"/>`) is capped corpus-wide.
    - *Near-duplicate sibling cap* (`--sibling-cap`, default 20): elements are hashed on structural *shape* (tag + attribute names, values and text stripped) scoped to `(file, parent, shape)`, so many same-shape siblings under one parent (e.g. hundreds of `<dmRef>` entries in one reference list) are capped, without touching e.g. one `<dmCode>` per file under its own `<dmAddress>` — those never share a parent context. When a snippet is dropped as an excess sibling, its children are skipped too (a capped `<dmRef>`'s nested `<dmRefIdent>`/`<dmCode>` are part of the same redundant list entry, not independent candidates).
    - *Short-description cap*: a generated description under 3 words is dropped.
  - **Output**: `../pretrain/sda_pairs.{train,dev,test}.jsonl` (repo root), split by source file. Each record: `{"structured": <xml string>, "text": <description>, "source_file", "element", "xpath", "tier", "generated_by"}`.
  - Run with `python3 preprocess.py [--input-dirs ...] [--output-dir ...] [--dedup-cap N] [--max-llm-calls N] [--llm-concurrency N] [--llm-model ...]`. The safety cap defaults to 500 live calls/run; the full corpus needs ~19,800 unique LLM-tier snippets described, so a full from-scratch run needs `--max-llm-calls` raised accordingly (e.g. `20000`) — at Haiku 4.5 pricing this runs a few dollars end to end, and the per-snippet cache means re-running with a smaller cap is always safe (it only tops up whatever wasn't covered last time).

- **`build_xml_entity.py`** — Stage 2: builds Masked Entity Prediction (MEP) examples from Stage 1's `sda_pairs.*.jsonl`, following [SANTA](https://arxiv.org/abs/2305.19912)'s ([`build_code_entity.py`](https://github.com/OpenMatch/SANTA/blob/master/processing/Code/build_code_entity.py), [`build_product_entity.py`](https://github.com/OpenMatch/SANTA/blob/master/processing/Product/build_product_entity.py)).
  - **Entity recognition**: The XML analog of a code identifier is an attribute value that names or codes something (`modelIdentCode`, `systemCode`, `issueNumber`, `languageIsoCode`, ...); the analog of a product proper noun is a capitalized/technical token inside prose text content (`techName`, `enterpriseName`, free text). **Entities are not element tags themselves** — masking a whole tag would remove the very structure the model is supposed to learn to condition on. Masking every identifier-like attribute value still leaves every tag name, attribute name, and piece of markup intact, exactly as masking every code identifier still leaves every keyword and operator in place.
  - **Identifying entities**: attribute values are candidates when the attribute name matches an identifier-like pattern (contains "code", "ident", "number", "classification", or "isocode", plus a small explicit list: `inWork`, `year`, `month`, `day`). Text-node entities use a capitalization/code-pattern heuristic as the *primary* signal — not NLTK POS tagging — because most S1000D text nodes (`<techName>`, `<infoName>`, ...) are short title/label fragments that use Title Case as a style convention, capitalizing every major word regardless of its grammatical role; checked directly, NLTK correctly tags words like "Management"/"Error" in such fragments as plain `NN` (common noun), which is grammatically right but means POS tagging alone silently misses exactly the label words this needs to catch. The code-pattern half of the heuristic requires a digit (`ICN-C0419-S1000D0382-001-01`, not `cross-referencing`) to avoid flagging ordinary English compound words or embedded XPath syntax (`following-sibling`, `normalize-space`) as entities — verified against 3,331 real text nodes from the corpus, this alone removed the false-positive matches on 26 of them (XPath keywords, English compound words) with no loss elsewhere. When NLTK (`nltk` + `averaged_perceptron_tagger_eng` + `punkt_tab`) is available, its NNP/NNPS-tagged tokens are added as a **pure supplement** — anything it finds beyond the primary heuristic (e.g. a genuine non-ASCII proper noun like "Provençal" that the ASCII-only regex can't see) is kept, but nothing the primary heuristic already found is ever removed by it. NLTK isn't a system-wide dependency; it's installed in a local `datasets/.venv/` (`.venv/bin/python3 build_xml_entity.py` to use it) so the default behavior via plain `python3` is unaffected.
  - **Masking**: the same entity string anywhere in a snippet always gets the same sentinel token (`<extra_id_0>`, `<extra_id_1>`, ...). Sentinels are substituted into a copied XML tree; an escaping-safe placeholder stands in during the tree mutation and is swapped for the real `<extra_id_N>` string only after serialization.
  - **Downsampling** (`--mask-ratio`, default 0.5): only this fraction of a snippet's unique candidate entities is actually masked, capped at 99 regardless (T5 has 100 sentinel tokens, `<extra_id_0>`–`<extra_id_99>`; one is reserved as the label sequence's trailing terminator, matching SANTA's own convention). 
  - **Output**: `../pretrain/mep_pairs.{train,dev,test}.jsonl`. Each record: `{"structured": <original xml>, "masked_structured": <xml with entities replaced by sentinels>, "label": <"<extra_id_0> value0 <extra_id_1> value1 ... <extra_id_k>">, "num_entities", "source_file", "element", "xpath"}`. A snippet with zero identified entities is dropped (nothing to predict).
  - Run with `python3 build_xml_entity.py [--input-dir ...] [--output-dir ...] [--mask-ratio F] [--random-seed N]`.

- **`build_retrieval_dataset.py`** — Stage 3: builds a retrieval finetuning/evaluation dataset from Stage 1's `sda_pairs.*.jsonl`, mirroring SANTA's finetuning benchmarks (Adv for code, ESCI (small) for product — Table 1 of the paper).
  - **Why not just reformat `sda_pairs`**: Stage 1 intentionally keeps up to `--dedup-cap`/`--sibling-cap` near-identical snippets across the corpus — useful signal for contrastive *alignment* pretraining, but fatal for retrieval *evaluation*, where "the correct document" must be well-defined. Two cleanups are applied that SANTA gets for free by using separately-curated benchmarks (Adv, ESCI-small) instead of its own pretraining corpora:
    1. **Corpus dedup**: documents are collapsed to one corpus entry per unique content hash, so no two corpus entries are ever equally "correct" for the same query.
    2. **Ambiguous-query filtering**: some generic-tier fallback descriptions (e.g. `"dm ref ident element."`, emitted whenever an element has no dedicated template and no prose text) are identical across hundreds of structurally different documents — one such text maps to 622 distinct documents in the train split alone. A query like that has no well-defined right answer, so it's dropped entirely rather than assigned an arbitrary "correct" pick.
  - **Splits**: reuses Stage 1's file-level train/dev/test assignment, so retrieval-dev/test stay disjoint from whatever the continuous-pretraining stage (SDA/MEP) actually trains the encoder on.
  - **Output**: `../retrieval/{corpus,queries}.{train,dev,test}.jsonl` and `qrels.{train,dev,test}.tsv` (repo root, a sibling of `pretrain/` rather than nested inside it — it's a downstream finetuning/eval artifact, not continuous-pretraining data), in the query-id/corpus-id/score layout BEIR and OpenMatch (SANTA's own training/eval toolkit) expect. `corpus.jsonl`: `{"docid", "structured", "element", "xpath", "source_file"}`. `queries.jsonl`: `{"qid", "text", "element", "xpath", "source_file"}`. `qrels.tsv`: `query-id\tcorpus-id\tscore` (always `1` — every surviving query has exactly one gold document after the ambiguity filter).
  - Run with `python3 build_retrieval_dataset.py [--input-dir ...] [--output-dir ...]`.

## Preprocessing data statistics

**Stage 1 — Structured Data Alignment pairs** (`sda_pairs.*.jsonl`, from `preprocess.py`):

| Split | Total pairs | Template tier | LLM/prose tier |
| --- | --- | --- | --- |
| Train | 42,231 | 23,291 | 18,940 |
| Dev | 2,009 | 1,413 | 596 |
| Test | 788 | 439 | 349 |

Every LLM-tier pair above now carries a real Claude-written description (`generated_by: "llm"`/`"llm_cached"`) — 0 fell back to the naive extractive summary, after raising `--max-llm-calls` to 20,000 and running the full ~19,800-snippet backlog (14,184 new live calls; ~1,000 were already cached from earlier partial runs).

**Stage 2 — Masked Entity Prediction pairs** (`mep_pairs.*.jsonl`, from `build_xml_entity.py`, `--mask-ratio 0.5` default):

| Split | SDA pairs in | Dropped (no entities) | MEP pairs out | Avg entities/example | Entity share of tokens |
| --- | --- | --- | --- | --- | --- |
| Train | 42,231 | 15,199 | 27,032 | 3.75 | 3.8% |
| Dev | 2,009 | 904 | 1,105 | 3.72 | 4.5% |
| Test | 788 | 229 | 559 | 4.20 | 5.9% |

MEP pairs out is ~4% lower than before the code-pattern fix (28,117→27,032 in train) — a handful of snippets whose only "entity" was a false-positive hyphenated word or embedded XPath keyword now correctly have none, so they're dropped instead of kept with a bad mask target. Coverage traded for correctness, deliberately.

"Entity share of tokens" is the analog of Table 5's "Entities" column in the SANTA paper: the fraction of whitespace/punctuation-delimited tokens in the (unmasked) `structured` XML that get identified as entities. It runs lower here (SANTA: 15–29%) mostly because the token count includes XML markup (tag names, angle brackets, attribute names) that code/product text didn't have — the entity tokens themselves are a similar-sized set, just diluted by more non-entity structure. A handful of large whole-`dmodule` snippets hit the 99-entity hard cap.


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

FOSSIG's entity share is highest despite being the smallest folder — its data modules document free/open-source software project metadata (SNS, business rules), which packs proportionally more identifier-like attribute values and proper nouns per snippet than BIKE's more prose-heavy procedural content. BIKE dominates pair *counts* simply because it's the largest folder (106 of 496 XML files) with the deepest, most uniformly-tagged data module structure.

**Stage 3 — Retrieval finetuning/evaluation dataset** (`retrieval/{corpus,queries}.*.jsonl` + `qrels.*.tsv`, from `build_retrieval_dataset.py`), the analog of Table 1:

| Split | SDA pairs in | Query-Doc pairs | Corpus (unique docs) | Ambiguous queries dropped |
| --- | --- | --- | --- | --- |
| Train | 42,231 | 25,146 | 25,987 | 388 |
| Dev | 2,009 | 1,659 | 1,714 | 20 |
| Test | 788 | 720 | 750 | 13 |

## Provenance

The `FOSSIG`, `S1000D spec sample`, and `s1kd-tools-doc` folders correspond to CSDB examples linked from the [s1kd-tools repository](https://github.com/kibook/s1kd-tools). The Bike description above is the accompanying source notice for the locally included S1000D Bike Issue 6 sample data. Consult the upstream repositories and the S1000D specification for licensing, current releases, schemas, and authoritative conformance guidance.
