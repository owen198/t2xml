#!/usr/bin/env python3
"""Convert t2xml's pretrain/retrieval datasets into the pre-tokenized
{query, positives, negatives, labels} JSONL schema train_santa.py expects
(see trainer.py's Sdataset.create_one_example / get_process_fn).

Each SANTA model in model_configs.json pairs one dataset folder as its
pretrain source with a *different* dataset folder as its finetune source --
no cross-folder mixing within a single model.

Pretrain rows join <pretrain-root>/<pretrain_source>/sda_pairs.*.jsonl (query
text) with the matching mep_pairs.*.jsonl (masked positive + tag label) on
source_file -- whole-file granularity means exactly one SDA and one MEP
record per source file, so source_file alone is a unique join key (no more
need for the old per-element (source_file, element, xpath) disambiguators).

Finetune rows come straight from <retrieval-root>/<finetune_source>/
{corpus,queries,qrels}: one row per qrels-positive pair, no masking,
negatives left empty -- SANTA's own first-round finetune (shell/finetune-
code.sh) does the same and relies on in-batch negatives.
"""
import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT = SCRIPT_DIR / "data"
DEFAULT_MODEL_CONFIG = SCRIPT_DIR / "model_configs.json"

# Offline safety-net cap so pathological documents don't blow up the output
# file; the real truncation happens at train time via --q_max_len/--p_max_len
# /--l_max_len (see trainer.py's create_one_example), same two-stage
# approach SANTA's own build_code.py/build_code_entity.py use.
SAFETY_MAX_LEN = 512


def encode(tokenizer, text):
    return tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=SAFETY_MAX_LEN
    )["input_ids"]


def load_model_configs(path: Path) -> dict[str, dict]:
    cfg = json.loads(Path(path).read_text())
    return {m["name"]: m for m in cfg["models"]}


def build_pretrain_split(tokenizer, split, pretrain_dir, out_dir):
    sda_path = pretrain_dir / f"sda_pairs.{split}.jsonl"
    mep_path = pretrain_dir / f"mep_pairs.{split}.jsonl"
    if not sda_path.exists() or not mep_path.exists():
        print(f"warning: missing sda/mep pairs for split {split} under {pretrain_dir}, skipping")
        return

    sda_text_by_key = {}
    with sda_path.open() as f:
        for line in f:
            rec = json.loads(line)
            sda_text_by_key[rec["source_file"]] = rec["text"]

    n_in, n_matched = 0, 0
    out_path = out_dir / f"pretrain.{split}.jsonl"
    with mep_path.open() as f, out_path.open("w") as out:
        for line in f:
            n_in += 1
            rec = json.loads(line)
            text = sda_text_by_key.get(rec["source_file"])
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
        print(f"warning: missing retrieval files for split {split} under {retrieval_dir}, skipping")
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
    global SAFETY_MAX_LEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base",
                         help="Tokenizer for the pre-tokenized ids; must be the same checkpoint family as the MODEL "
                              "pretrain.sh trains (default Salesforce/codet5-base), or the ids are garbage to the model.")
    parser.add_argument("--pretrain-root", default=str(REPO_ROOT / "pretrain"),
                         help="Directory containing one subdir per dataset folder (sda_pairs/mep_pairs).")
    parser.add_argument("--retrieval-root", default=str(REPO_ROOT / "retrieval"),
                         help="Directory containing one subdir per dataset folder (corpus/queries/qrels).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT),
                         help="Directory under which one subdir per model is written.")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--models", nargs="+", default=None,
                         help="Model names from model_configs.json to build (default: all).")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--safety-max-len", type=int, default=SAFETY_MAX_LEN,
                         help="Offline per-text token cap; must be >= the --p_max_len used at train time.")
    args = parser.parse_args()
    SAFETY_MAX_LEN = args.safety_max_len

    configs = load_model_configs(args.model_config)
    if args.models:
        unknown = [m for m in args.models if m not in configs]
        if unknown:
            raise SystemExit(f"unknown model name(s) not in {args.model_config}: {unknown}")
        targets = {n: configs[n] for n in args.models}
    else:
        targets = configs

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    pretrain_root = Path(args.pretrain_root)
    retrieval_root = Path(args.retrieval_root)
    out_root = Path(args.output_dir)

    for name, cfg in targets.items():
        model_out_dir = out_root / name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        pretrain_dir = pretrain_root / cfg["pretrain_source"]
        retrieval_dir = retrieval_root / cfg["finetune_source"]
        print(f"=== {name} (pretrain={cfg['pretrain_source']}, finetune={cfg['finetune_source']}) ===")
        for split in args.splits:
            build_pretrain_split(tokenizer, split, pretrain_dir, model_out_dir)
            build_finetune_split(tokenizer, split, retrieval_dir, model_out_dir)


if __name__ == "__main__":
    main()
