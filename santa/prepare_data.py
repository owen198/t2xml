#!/usr/bin/env python3
"""Convert t2xml's pretrain/retrieval datasets into the pre-tokenized
{query, positives, negatives, labels} JSONL schema train_santa.py expects
(see trainer.py's Sdataset.create_one_example / get_process_fn).

Pretrain rows join sda_pairs.*.jsonl (query text) with mep_pairs.*.jsonl
(masked positive + entity label) on (source_file, element, xpath). SANTA's
own processing/Code/build_code_entity.py builds both signals onto a single
row from one source document; t2xml built them as two separate files that
happen to share that same key, so joining reproduces the same row shape.

Finetune rows come straight from retrieval/{corpus,queries,qrels}: one row
per qrels-positive pair, no masking, negatives left empty -- SANTA's own
first-round finetune (shell/finetune-code.sh) does the same and relies on
in-batch negatives.
"""
import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT = SCRIPT_DIR / "data"

# Offline safety-net cap so pathological documents don't blow up the output
# file; the real truncation happens at train time via --q_max_len/--p_max_len
# /--l_max_len (see trainer.py's create_one_example), same two-stage
# approach SANTA's own build_code.py/build_code_entity.py use.
SAFETY_MAX_LEN = 512


def encode(tokenizer, text):
    return tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=SAFETY_MAX_LEN
    )["input_ids"]


def build_pretrain_split(tokenizer, split, pretrain_dir, out_dir):
    sda_path = pretrain_dir / f"sda_pairs.{split}.jsonl"
    mep_path = pretrain_dir / f"mep_pairs.{split}.jsonl"
    if not sda_path.exists() or not mep_path.exists():
        print(f"warning: missing sda/mep pairs for split {split}, skipping")
        return

    sda_text_by_key = {}
    with sda_path.open() as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["source_file"], rec["element"], rec["xpath"])
            sda_text_by_key[key] = rec["text"]

    n_in, n_matched = 0, 0
    out_path = out_dir / f"pretrain.{split}.jsonl"
    with mep_path.open() as f, out_path.open("w") as out:
        for line in f:
            n_in += 1
            rec = json.loads(line)
            key = (rec["source_file"], rec["element"], rec["xpath"])
            text = sda_text_by_key.get(key)
            if text is None:
                continue
            n_matched += 1
            row = {
                "query": encode(tokenizer, text),
                "positives": [encode(tokenizer, rec["masked_structured"])],
                "negatives": [],
                "labels": encode(tokenizer, rec["label"]),
            }
            out.write(json.dumps(row) + "\n")
    print(f"pretrain.{split}: {n_matched}/{n_in} mep rows matched an sda text -> {out_path}")


def build_finetune_split(tokenizer, split, retrieval_dir, out_dir):
    corpus_path = retrieval_dir / f"corpus.{split}.jsonl"
    queries_path = retrieval_dir / f"queries.{split}.jsonl"
    qrels_path = retrieval_dir / f"qrels.{split}.tsv"
    if not (corpus_path.exists() and queries_path.exists() and qrels_path.exists()):
        print(f"warning: missing retrieval files for split {split}, skipping")
        return

    corpus_text = {}
    with corpus_path.open() as f:
        for line in f:
            rec = json.loads(line)
            corpus_text[rec["docid"]] = rec["structured"]

    query_text = {}
    with queries_path.open() as f:
        for line in f:
            rec = json.loads(line)
            query_text[rec["qid"]] = rec["text"]

    n_in, n_written = 0, 0
    out_path = out_dir / f"finetune.{split}.jsonl"
    with qrels_path.open() as f, out_path.open("w") as out:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_in += 1
            qid, docid = row["query-id"], row["corpus-id"]
            if qid not in query_text or docid not in corpus_text:
                continue
            n_written += 1
            out_row = {
                "query": encode(tokenizer, query_text[qid]),
                "positives": [encode(tokenizer, corpus_text[docid])],
                "negatives": [],
            }
            out.write(json.dumps(out_row) + "\n")
    print(f"finetune.{split}: {n_written}/{n_in} qrels rows written -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="t5-base")
    parser.add_argument("--pretrain-dir", default=str(REPO_ROOT / "pretrain"))
    parser.add_argument("--retrieval-dir", default=str(REPO_ROOT / "retrieval"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    pretrain_dir = Path(args.pretrain_dir)
    retrieval_dir = Path(args.retrieval_dir)
    for split in args.splits:
        build_pretrain_split(tokenizer, split, pretrain_dir, out_dir)
        build_finetune_split(tokenizer, split, retrieval_dir, out_dir)


if __name__ == "__main__":
    main()
