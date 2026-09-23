#!/usr/bin/env python3
"""Build a widened S1000D_spec_sample search corpus (all train+dev+test documents, 332 total) for the
"candidate pool size" experiment in results/README.md. The test queries and split assignment are
unchanged -- only what the model has to search over changes, from 21 candidates to 332.

`docid` in corpus.{train,dev,test}.jsonl is a per-split running index (doc_0, doc_1, ...), NOT globally
unique -- doc_0 in train/dev/test are three different documents (verified: source_file differs). This
script remaps every docid to f"{split}_{docid}" before merging, so nothing silently collides, and rewrites
qrels.test.tsv to point at the remapped id of the *same* document. Queries are untouched.

Writes new files only: corpus.all332.jsonl, qrels.test.all332.tsv, alongside the existing files in
--root. Does not modify corpus.{train,dev,test}.jsonl or qrels.test.tsv.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="e.g. retrieval/S1000D_spec_sample or retrieval_anonq/S1000D_spec_sample")
    args = ap.parse_args()
    root = Path(args.root)

    all_docs = []
    remap = {}  # (split, old_docid) -> new_docid
    for split in ("train", "dev", "test"):
        for line in (root / f"corpus.{split}.jsonl").open():
            r = json.loads(line)
            new_id = f"{split}_{r['docid']}"
            remap[(split, r["docid"])] = new_id
            r2 = dict(r)
            r2["docid"] = new_id
            all_docs.append(r2)

    out_corpus = root / "corpus.all332.jsonl"
    with out_corpus.open("w") as f:
        for r in all_docs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(all_docs)} documents -> {out_corpus}")

    # qrels.test.tsv: query-id, corpus-id, score. The corpus-id there is a *test-split* docid,
    # so remap using the ("test", old_id) key.
    in_qrels = root / "qrels.test.tsv"
    out_qrels = root / "qrels.test.all332.tsv"
    lines = in_qrels.read_text().splitlines()
    header, rows = lines[0], lines[1:]
    with out_qrels.open("w") as f:
        f.write(header + "\n")
        for line in rows:
            qid, docid, score = line.split("\t")
            new_docid = remap[("test", docid)]
            f.write(f"{qid}\t{new_docid}\t{score}\n")
    print(f"{len(rows)} qrels rows -> {out_qrels}")


if __name__ == "__main__":
    main()
