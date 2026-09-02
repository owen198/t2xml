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

## Open experiment variables (dataset design)

Design questions tracked for experimentation. Notes are generated by AI, should not be viewed as definite. 

### Stage 1 — SDA (`preprocess.py`)

| Variable | Current default | Question | Notes |
| --- | --- | --- | --- |
| SDA snippet text placement (in-tag vs. stripped) | Text stays inline inside tags (`ET.tostring` of the element, e.g. `<para>Remove wheels</para>`) | Does this actually teach the model to associate structure with content? | - SDA's contrastive loss (description vs. whole snippet, in-batch negatives) works on the *pooled* embedding, not per-token — doesn't force tag↔word attribution on its own; that job is MEP's.<br>- Keeping text inline is still right for SDA — preserves structure as a discriminative signal, what makes this "structure-aware" vs. plain-text retrieval pretraining.<br>- Candidate ablation: SDA-only with tag-stripped doc side vs. tag-inline doc side, compare downstream MEP/retrieval eval. |
| Tag semantics/definitions surfaced in NL description text | Descriptions (LLM and template tier) describe content only, not what the tag itself means | Should descriptions be schema/XSD-definition-aware, at least for structural tags? | - Prose tags (`para`, etc.): content dominates meaning, tag definition likely adds little.<br>- Structural/attribute-only tags (`dmCode`, `security`, `issueInfo`, `dmRef`, ...): the tag *is* most of the meaning; template tier already encodes this via hand-written per-tag templates.<br>- Open: does explicit tag-definition text (XSD `xs:annotation`/`xs:documentation`) measurably help on that structural subset, vs. no effect applied uniformly. |
| CodeSearchNet-style dedup/sibling cap magnitude (`--dedup-cap`/`--sibling-cap`, both default 20) | Exact-duplicate and near-duplicate-sibling snippets are *capped* at 20 occurrences (corpus-wide / per file+parent+shape), not deduped to 1 | Is 20 the right cap, or should this follow CSN's stricter dedup-to-1? | - CSN dedupes hard because a duplicated *function* is genuinely redundant; XML boilerplate isn't — e.g. `<security securityClassification="01"/>` recurring is normal, not noise.<br>- Leaning toward keeping a cap (not full dedup), but 20 was picked, not derived.<br>- Candidate ablation: cap ∈ {1, 20, uncapped}, compare corpus size vs. downstream eval.<br>- Separate issue: `exact_dup_counts`/`sibling_shape_counts` accumulate in file-walk order — BIKE (walked first) wins capped slots before FOSSIG's version is considered; worth fixing independent of the cap value. |
| Segmentation unit: every whitelisted element vs. CSN's one-function-per-example | Every whitelisted element, at every depth, becomes its own snippet — a `<procedure>` and its child `<mainProcedure>` are both emitted, heavily overlapping | Should segmentation stay whitelist-every-element, or move toward CSN's non-overlapping one-unit-per-example? | - CSN's unit (a function) is self-contained and non-overlapping by construction; this pipeline's unit isn't.<br>- Nesting means a snippet and its own descendant snippet can land in the same batch — an in-batch "negative" can be a near-duplicate of the positive (false negative for contrastive loss).<br>- Multi-scale coverage (whole-subtree + single-element) may be worth that cost, but it's a deliberate divergence from CSN's segmentation philosophy. |
| Minimum structural complexity per snippet (e.g. "≥3 tags") | No floor beyond `is_trivial()` (drops only *completely* empty elements: no attributes, no text, no children) | Should a minimum tag/descendant count be required for a snippet to count as "structured" for SDA? | - Checked `pretrain/sda_pairs.train.jsonl`: 46.5% of pairs have exactly 1 element (no children), 13.3% have 2 — a blanket "≥3 elements" rule would drop **~60% of the corpus**.<br>- Of that ≤2-element group: 17,335 template-tier (attribute-dense, low tag-count/high info density), 7,927 LLM-tier (bare `para` prose, no nested tags — fine per the row above).<br>- Raw descendant count is a poor floor — conflates "shallow" with "low-information," false for attribute-heavy elements and plain prose.<br>- If a floor is added, scope it to the template/structural tier specifically, not corpus-wide. |
| LLM description prompt (`_llm_prompt()` in `preprocess.py`) | Single zero-shot user message, no system prompt, no few-shot examples; `claude-haiku-4-5-20251001`, `max_tokens=120`, snippet truncated to `LLM_PROMPT_MAX_CHARS=4000`. Current text verbatim:<br><br>`You are labeling training data for a dense retrieval model that must learn to match natural-language descriptions to structured S1000D XML snippets.`<br>`Write ONE concise sentence (max ~40 words) in plain English describing what the following XML snippet documents or specifies. Describe the meaning/content, not the XML syntax itself.`<br><br>`<snippet tag="{tag}">{snippet}</snippet>` | Every LLM-tier query in the benchmark is a direct product of this prompt — was it validated, and what does each instruction cost or buy? | - Never ablated: written once, then ~19,800 snippets were labeled and cached under it. The cache is keyed on snippet hash only, **not on prompt text or model** — changing the prompt silently reuses old descriptions unless the cache is invalidated. That's the first thing to fix before any prompt experiment.<br>- "Describe the meaning/content, not the XML syntax itself" pushes descriptions *away* from tag names — arguably counterproductive for a benchmark whose whole point is structure-aware retrieval, and related to the "Tag semantics/definitions" row above.<br>- "max ~40 words" / `max_tokens=120` bounds query length (matches the measured q_max_len=50 distribution), but the one-sentence cap may be why sibling snippets in a nesting chain get near-identical descriptions — there's not enough room to differentiate `dmAddress` from its parent `identAndStatusSection`.<br>- Nothing discourages quoting identifier values verbatim — directly feeds the query-leakage issue in the Stage 3 rows below.<br>- The snippet-only prompt has no context about *where* the element sits (xpath/ancestors are available but not passed), so the model can't say what distinguishes this snippet from its parent even in principle.<br>- 4000-char truncation is silent: the p99=1898-token/max-314k-token snippets (see `p_max_len` row) get described from a prefix only, with no marker that content was cut.<br>- Candidate experiments: add xpath/ancestor context; drop or invert the "not the XML syntax" clause; forbid verbatim identifier copying; few-shot with hand-written exemplars; compare downstream retrieval MRR across variants (needs the cache-keying fix first). |

### Stage 2 — masking tasks (MEP `build_xml_entity.py`, and proposed MTP)

| Variable | Current default | Question | Notes |
| --- | --- | --- | --- |
| Masked Tag Prediction (MTP): mask element tag names instead of entity values | Not implemented — MEP is the only masking task, and by design never touches tag names (see its module docstring): masking a value teaches the model to infer content from structure, not the reverse | Should a second pretraining task (or only have MTP) mask element *tag names* , so the model learns to infer structure from content/context — the "masked tag prediction" eval task already named as a goal in the top-level README but never built as a pretraining objective. | - Complements MEP rather than replacing it — MEP recovers a value from surrounding structure; MTP would recover a tag name from surrounding content/attributes/sibling tags, the inverse direction.<br>- No cross-occurrence dedup: MEP shares one sentinel per identifier string since repetition signals "the same entity named twice"; repeated tag names (e.g. three sibling `<proceduralStep>`s) are structurally normal, not an identity signal — each masked occurrence gets its own sentinel/label.<br>- Per-snippet ratio: flat count over all tag occurrences, mirroring MEP's `round(len(order) * mask_ratio)` (e.g. 12 of 60 tags at 0.2), with no per-sibling-group stratification — a small "sibling group" (same-parent children sharing one tag name) fully masked by chance is accepted as noise, not actively prevented.<br>- Singleton children (no same-tag sibling, e.g. a lone `<closeRqmt>`) stay eligible — no nearby same-tag example either way, so masking isn't a regression.<br>- Root tag always stays unmasked — no parent/sibling context to recover it from.<br>- Scope: element tag names only, not attribute names (attribute-value masking is already MEP's job).<br>- Forces the row above's decision: MTP's eligibility floor (needs a parent+child pair) is stricter than MEP's `no_entities` drop — the 46.5%-single-element finding means a large share of the corpus is ineligible for MTP even where it's fine for MEP/SDA. |

### Stage 3 — retrieval dataset construction (`build_retrieval_dataset.py`)

| Variable | Current default | Question | Notes |
| --- | --- | --- | --- |
| Query/doc overlap from deriving the retrieval benchmark out of `sda_pairs.*.jsonl` | `evaluate_xml.sh` on the test split scored MRR 0.88 — content-hash dedup (Pass 1) only collapses byte-identical `structured` strings into one corpus entry | Is that 0.88 inflated by overlap the dedup step doesn't catch, rather than reflecting genuine retrieval difficulty? | - **Confirmed**, not just suspected: inspected `retrieval/corpus.test.jsonl` directly — e.g. `doc_0`..`doc_4` are the same source file's `dmodule` → `identAndStatusSection` → `dmAddress` → `dmIdent` → `dmCode` nesting chain, each literally a substring of the previous one, kept as separate corpus entries because their XML text differs slightly (so exact-hash dedup doesn't merge them). Their paired queries (`q_0`..`q_4`) are near-paraphrases sharing the same rare anchors ("mountain bicycle", "June 2024"/"June 19, 2024", "issue 009", "ASD").<br>- Quantified on the test split: all 25 distinct source files contribute more than one corpus doc each, and there are **3,556 nested (literal substring) doc pairs among only 750 corpus docs** — this is the nesting-inclusive segmentation from Stage 1 (see "Segmentation unit" row above) emitting a whole-subtree snippet *and* its own descendant snippets as separate SDA pairs, which survive Pass 1's exact-hash dedup as distinct corpus docs since they're near- not exact-duplicates.<br>- A retriever can hit MRR 0.88 by matching on source-file-unique rare tokens shared across a whole nested family (specific `dmCode` attribute values, issue numbers, dates, title text) rather than genuinely discriminating structure — those anchors are rare enough globally (unique per source file) that even shallow lexical matching within a family could rank correctly, inflating MRR relative to a benchmark built from non-overlapping units (SANTA's own Adv/ESCI-small).<br>- Pass 2's ambiguous-text drop only removes queries whose *exact* text string maps to >1 distinct doc hash — it doesn't detect or penalize near-duplicate *documents* the way it does near-duplicate query text.<br>- Fix has to happen in `build_retrieval_dataset.py`, not by regenerating `queries.*.jsonl` alone — corpus/queries/qrels are derived together from the same records, so re-running the current logic reproduces the identical overlap. Candidate policies for collapsing each nesting chain (same `source_file`, substring relationship), illustrated on the `doc_0`..`doc_4` family: **(A) keep deepest/most specific node** — keep only `doc_4`/`q_4` (the bare `<dmCode .../>`), drop the four ancestors; corpus becomes small specific snippets, less boilerplate per entry. **(B) keep shallowest/whole-subtree node** — keep only `doc_0`/`q_0` (the full `<dmodule>`), drop the four descendants; richer documents, opposite trade-off. **(C) keep all levels but exclude same-chain docs from each other's candidate pool** — `q_2` would be ranked against only the ~745 unrelated docs, never `doc_0`/`doc_1`/`doc_3`/`doc_4`. **C is not recommended**: those siblings are exactly the near-duplicates that make the family hard, so removing them as candidates makes the query strictly *easier* and inflates MRR further rather than fixing it. **Leaning A** — it removes the "narrow to a family via rare shared tokens, then guess among near-duplicate members" shortcut by leaving one member per chain, and favors specific snippets over boilerplate-heavy whole documents. Analogous to the "Segmentation unit" row's open question about non-overlapping units. |
| Should queries be regenerated (LLM-rewritten/anonymized) to remove verbatim identifier copying? | Queries are reused as-is from Stage 1's `text` field — template-tier descriptions are a direct rendering of the element's own attributes, and LLM-tier ones freely quote identifiers/dates from the snippet | Does a query that recites its document's own identifier strings measure retrieval, or just string matching? | - Separate from the nesting-chain issue above and **not fixed by option A** — it survives any corpus-side dedup, since it's about what the query text contains.<br>- Visible in the same sample: `q_4` (template tier) is "Data module code for model S1000DBIKE, system AAA-D00-0-0, assembly 00-00AA, info code 00WA, item location D." against a `doc_4` that is exactly `<dmCode modelIdentCode="S1000DBIKE" systemCode="D00" ... infoCode="00W" .../>` — every content word is a verbatim attribute value. LLM-tier queries leak too, just less mechanically ("issue 009", "June 19, 2024", "ASD").<br>- These identifiers are globally rare (near-unique per source file), so exact-match on them can rank the positive first with no structural understanding at all — a plausible second contributor to the 0.88 alongside nesting overlap.<br>- Counterpoint (why this isn't obviously a bug): a real S1000D user *does* search by `dmCode`/issue number, so identifier-bearing queries are realistic. The question is whether the benchmark should be *dominated* by them, not whether they belong at all.<br>- Candidate directions: (a) regenerate queries with an LLM prompted to describe function/purpose without quoting identifier values; (b) keep them but report a lexical-overlap-stratified breakdown (high- vs. low-overlap query subsets) so the structural-retrieval claim rests on the low-overlap slice; (c) leave template-tier queries as-is and treat them as a known-easy subset. (b) is the cheapest — it's a scoring change, needs no data regeneration.<br>- Untested either way: no measurement yet of how much of the 0.88 a pure BM25/exact-match baseline recovers, which would separate lexical leakage from genuine model contribution. That baseline is the natural first experiment here. |

### Stage 3 — training hyperparameters (pretrain/finetune, `santa/`)

| Variable | Current default | Question | Notes |
| --- | --- | --- | --- |
| In-batch vs. hard negatives | `--train_n_passages 1` in both `pretrain.sh` and `finetune.sh` — `trainer.py`'s `get_process_fn` sets `negs = []` whenever `train_n_passages == 1`, so every query has exactly one (positive) passage | Should hard negatives be mined and added via `train_n_passages > 1`, or is pure in-batch-negatives sufficient? | - `SModel.forward` (`model.py`) computes the full `q_reps @ p_reps.T` batch score matrix and does cross-entropy against the diagonal — in-batch negatives by construction, currently the *only* negative source.<br>- Worth an ablation once a hard-negative-mining step exists (e.g. BM25 or a first-pass dense retriever over the corpus). |
| `p_max_len` (default 256, both scripts) | Static padding to `p_max_len` for every batch (`data_collator.py`'s `QPCollator` uses `padding='max_length'`, not dynamic padding) | Is 256 the right passage budget? | - Measured on `pretrain/sda_pairs.train.jsonl`'s `structured` field (codet5-base tokenizer): mean 209.5, p50=46, p90=224, p99=1898, max=314,624.<br>- Median passage (46 tok) is >80% padding at 256 — wasted compute every batch.<br>- Tail (p99=1898, max 314k) gets truncated to <15% of its content — these are the whole-subtree parent snippets from the nesting-inclusive segmentation (Stage 1 row above).<br>- One scalar can't serve both ends; worth deciding whether to (a) change the scalar, or (b) cap/filter/split oversized snippets upstream instead of relying on silent truncation. |
| `q_max_len` (default 50, both scripts) | Static padding, same mechanism as `p_max_len` | Is 50 the right query budget? | - Measured on the `text` (query) field: mean 26.4, p50=25, p90=42, p99=58, max=84.<br>- Only the top ~1% of queries get clipped (p99=58 vs. cap 50) — minor, lower priority than `p_max_len`. |
| `l_max_len` (default 64, both scripts) | Static padding, same mechanism | Is 64 the right MEP-label budget? | - Measured on `pretrain/mep_pairs.train.jsonl`'s `label` field: mean 14.8, p50=10, p90=25, p99=99, max=663.<br>- p90 fits comfortably under 64; only a thin tail (p99=99) exceeds it — lower priority than `p_max_len`. |
| Learning rate (5e-5 pretrain / 2e-5 finetune) and epochs (10 pretrain / 12 finetune) | Copied verbatim from SANTA's own CodeSearchNet protocol (paper Appendix A.4: lr=2e-5 for the 5 CodeSearch languages, epoch=12; pretrain lr=5e-5, epoch=10) | Do CSN's code-domain values transfer to t2xml's XML domain and corpus scale? | - Unlike the length caps above, this needs an actual training run's loss curve, not a static check.<br>- Risk: t2xml's pretrain corpus (27,032 MEP train examples) is several orders of magnitude smaller than CodeSearchNet's, so the same epoch count means far more repetition per example and real overfitting risk the paper's numbers say nothing about.<br>- Treat lr/epochs as a joint sweep once training is actually run, not as inherited defaults. |

## Provenance

The `FOSSIG`, `S1000D spec sample`, and `s1kd-tools-doc` folders correspond to CSDB examples linked from the [s1kd-tools repository](https://github.com/kibook/s1kd-tools). The Bike description above is the accompanying source notice for the locally included S1000D Bike Issue 6 sample data. Consult the upstream repositories and the S1000D specification for licensing, current releases, schemas, and authoritative conformance guidance.
