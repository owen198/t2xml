#!/usr/bin/env python3
"""Same-domain pretrain-vs-finetune experiment (results/README.md, "Same-domain training").

Splits S1000D_spec_sample's own 297 train documents into two disjoint halves (deterministic hash on
source_file, not touching the existing dev/test split), so pretraining and finetuning can happen on the
SAME dataset's own content -- the setup SANTA itself actually uses (pretrain and finetune on the same
corpus family), as opposed to every other experiment here, which pretrains on a different S1000D folder.

Two-way swap instead of full k-fold: half A -> pretrain / half B -> finetune, and the reverse. Each
direction also gets a finetune-only control on the SAME half used for finetuning in that direction, so
data volume is held constant when judging whether pretraining helped.

Non-destructive: reads existing pretrain/S1000D_spec_sample/{sda,mep}_pairs.*.jsonl and
retrieval/S1000D_spec_sample/{corpus,queries,qrels}.*.jsonl, writes new files only under
pretrain_samedomain/ and retrieval_samedomain/ (halfA/halfB subfolders), plus tokenized output under
santa/data_samedomain/.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDER = "S1000D_spec_sample"


SALT = ""


def half_of(source_file: str) -> str:
    return "halfA" if int(hashlib.md5((SALT + source_file).encode()).hexdigest(), 16) % 2 == 0 else "halfB"


def split_pretrain(pretrain_dir: Path, out_root: Path):
    sda = [json.loads(l) for l in (pretrain_dir / "sda_pairs.train.jsonl").open()]
    mep = [json.loads(l) for l in (pretrain_dir / "mep_pairs.train.jsonl").open()]
    counts = {"halfA": 0, "halfB": 0}
    for half in ("halfA", "halfB"):
        out_dir = out_root / half / FOLDER
        out_dir.mkdir(parents=True, exist_ok=True)
        sda_h = [r for r in sda if half_of(r["source_file"]) == half]
        mep_h = [r for r in mep if half_of(r["source_file"]) == half]
        counts[half] = len(sda_h)
        with (out_dir / "sda_pairs.train.jsonl").open("w") as f:
            for r in sda_h:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (out_dir / "mep_pairs.train.jsonl").open("w") as f:
            for r in mep_h:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # dev is untouched by the split (never part of train); needed as-is for best-dev-pretrain.sh.
        for name in ("sda_pairs.dev.jsonl", "mep_pairs.dev.jsonl"):
            (out_dir / name).write_text((pretrain_dir / name).read_text())
    print(f"pretrain split: halfA={counts['halfA']} halfB={counts['halfB']} docs (of {len(sda)} train docs)")


def split_retrieval(retrieval_dir: Path, out_root: Path):
    corpus = [json.loads(l) for l in (retrieval_dir / "corpus.train.jsonl").open()]
    queries = [json.loads(l) for l in (retrieval_dir / "queries.train.jsonl").open()]
    qrels_lines = (retrieval_dir / "qrels.train.tsv").read_text().splitlines()
    header, qrels_rows = qrels_lines[0], qrels_lines[1:]
    docid_half = {r["docid"]: half_of(r["source_file"]) for r in corpus}
    qid_half = {r["qid"]: half_of(r["source_file"]) for r in queries}

    counts = {"halfA": 0, "halfB": 0}
    for half in ("halfA", "halfB"):
        out_dir = out_root / half / FOLDER
        out_dir.mkdir(parents=True, exist_ok=True)
        c_h = [r for r in corpus if docid_half[r["docid"]] == half]
        q_h = [r for r in queries if qid_half[r["qid"]] == half]
        counts[half] = len(c_h)
        with (out_dir / "corpus.train.jsonl").open("w") as f:
            for r in c_h:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (out_dir / "queries.train.jsonl").open("w") as f:
            for r in q_h:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (out_dir / "qrels.train.tsv").open("w") as f:
            f.write(header + "\n")
            for line in qrels_rows:
                qid, docid, score = line.split("\t")
                if qid_half.get(qid) == half:
                    f.write(line + "\n")
        # dev/test untouched -- same eval set for every arm.
        for name in ("corpus.dev.jsonl", "queries.dev.jsonl", "qrels.dev.tsv",
                     "corpus.test.jsonl", "queries.test.jsonl", "qrels.test.tsv"):
            (out_dir / name).write_text((retrieval_dir / name).read_text())
    print(f"retrieval split: halfA={counts['halfA']} halfB={counts['halfB']} docs")


def main():
    global SALT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrain-root", default=str(REPO_ROOT / "pretrain"))
    ap.add_argument("--retrieval-root", default=str(REPO_ROOT / "retrieval"))
    ap.add_argument("--pretrain-out", default=str(REPO_ROOT / "pretrain_samedomain"))
    ap.add_argument("--retrieval-out", default=str(REPO_ROOT / "retrieval_samedomain"))
    ap.add_argument("--salt", default="", help="Change to get a different halfA/halfB partition (e.g. for a robustness check).")
    args = ap.parse_args()
    global SALT
    SALT = args.salt
    split_pretrain(Path(args.pretrain_root) / FOLDER, Path(args.pretrain_out))
    split_retrieval(Path(args.retrieval_root) / FOLDER, Path(args.retrieval_out))


if __name__ == "__main__":
    main()
