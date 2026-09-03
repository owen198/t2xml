#!/usr/bin/env python3
"""Stage 3: build a SANTA-style retrieval finetuning/evaluation dataset from
Stage 1's (description, XML snippet) pairs.

Mirrors SANTA's finetuning benchmarks (Adv for code, ESCI (small) for
product -- Table 1 of the paper): a query is a natural-language description,
its positive document is a structured data snippet, and retrieval is scored
by ranking documents from a fixed per-split candidate corpus.

Why this can't just be a reformatting of sda_pairs.*.jsonl: Stage 1
intentionally keeps up to --dedup-cap/--sibling-cap near-identical snippets
across the corpus (fine for contrastive *alignment* pretraining, where
seeing the same boilerplate shape repeatedly is useful signal). A retrieval
*evaluation* needs "the correct document" to be well-defined, which breaks
in two ways SANTA's own Adv/ESCI-small benchmarks don't have to deal with
(they're separately curated, deduplicated datasets):
  1. Exact-duplicate documents: the same content appearing under multiple
     source files/xpaths would make several corpus entries equally
     "correct," so this stage collapses them to one corpus entry per unique
     document (by content hash).
  2. Non-discriminative queries: some generic-tier fallback descriptions
     (e.g. "dm ref ident element.", used whenever an element has no
     dedicated template and no prose text) are identical across hundreds of
     structurally different documents -- checked empirically, one such text
     maps to 622 distinct documents in the train split alone. Such a query
     has no well-defined right answer and is dropped rather than kept with
     a made-up "correct" pick.

Output format follows the queries.jsonl / corpus.jsonl / qrels.tsv layout
BEIR and OpenMatch (SANTA's own training/eval toolkit) expect, so it plugs
into that tooling directly.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PRETRAIN_DIR = REPO_ROOT / "pretrain"


def content_hash(xml_snippet: str) -> str:
    # Matches preprocess.py's own normalization so a document that was
    # deduplicated there under the same rule is recognized as the same
    # document here too.
    normalized = re.sub(r"\s+", " ", xml_snippet).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_split(in_path: Path, corpus_out: Path, queries_out: Path, qrels_out: Path, stats: dict):
    records = []
    with in_path.open() as f:
        for line in f:
            records.append(json.loads(line))
    stats["records_in"] = len(records)

    # Pass 1: corpus = one entry per unique document (by content hash),
    # keeping the first-seen record as the representative.
    hash_to_docid = {}
    corpus_rows = []
    for rec in records:
        h = content_hash(rec["structured"])
        if h not in hash_to_docid:
            hash_to_docid[h] = f"doc_{len(corpus_rows)}"
            corpus_rows.append({
                "docid": hash_to_docid[h],
                "structured": rec["structured"],
                "element": rec.get("element"),
                "xpath": rec.get("xpath"),
                "source_file": rec.get("source_file"),
            })
    stats["corpus_size"] = len(corpus_rows)

    # Pass 2: find which query texts are ambiguous (map to more than one
    # distinct document) so they can be excluded entirely -- see module
    # docstring point 2.
    text_to_hashes = defaultdict(set)
    for rec in records:
        text_to_hashes[rec["text"]].add(content_hash(rec["structured"]))
    ambiguous_texts = {t for t, hs in text_to_hashes.items() if len(hs) > 1}
    stats["ambiguous_texts_dropped"] = len(ambiguous_texts)

    # Pass 3: emit one query per surviving unique text, with its single
    # qrels-positive docid.
    seen_texts = set()
    query_rows = []
    qrels_rows = []
    for rec in records:
        text = rec["text"]
        if text in ambiguous_texts or text in seen_texts:
            continue
        seen_texts.add(text)
        docid = hash_to_docid[content_hash(rec["structured"])]
        qid = f"q_{len(query_rows)}"
        query_rows.append({
            "qid": qid,
            "text": text,
            "element": rec.get("element"),
            "xpath": rec.get("xpath"),
            "source_file": rec.get("source_file"),
        })
        qrels_rows.append((qid, docid))
    stats["queries_out"] = len(query_rows)

    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    with corpus_out.open("w") as f:
        for row in corpus_rows:
            f.write(json.dumps(row) + "\n")
    with queries_out.open("w") as f:
        for row in query_rows:
            f.write(json.dumps(row) + "\n")
    with qrels_out.open("w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, docid in qrels_rows:
            f.write(f"{qid}\t{docid}\t1\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build a retrieval finetuning/eval dataset (queries/corpus/qrels) from SDA pairs."
    )
    parser.add_argument("--input-dir", type=str, default=str(PRETRAIN_DIR))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "retrieval"))
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    for split in args.splits:
        in_path = in_dir / f"sda_pairs.{split}.jsonl"
        if not in_path.exists():
            print(f"warning: missing {in_path}, skipping split {split}", file=sys.stderr)
            continue
        stats = {}
        build_split(
            in_path,
            out_dir / f"corpus.{split}.jsonl",
            out_dir / f"queries.{split}.jsonl",
            out_dir / f"qrels.{split}.tsv",
            stats,
        )
        print(f"--- {split} ---")
        print(f"sda pairs in:              {stats['records_in']}")
        print(f"corpus size (unique docs): {stats['corpus_size']}")
        print(f"ambiguous texts dropped:   {stats['ambiguous_texts_dropped']}")
        print(f"queries out (1:1 qrels):   {stats['queries_out']}")
        print(f"-> {out_dir / f'corpus.{split}.jsonl'}")
        print(f"-> {out_dir / f'queries.{split}.jsonl'}")
        print(f"-> {out_dir / f'qrels.{split}.tsv'}")


if __name__ == "__main__":
    main()
