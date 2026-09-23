"""Split queries by how much they lexically overlap their gold document's own *discriminative* identifiers
(rare across the corpus, not boilerplate like "s1000d" or a shared year), then score BM25 and a dense
checkpoint on each half separately. Read-only: does not modify retrieval/ or pretrain/."""
import json, os, re, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(REPO, "retrieval", "S1000D_spec_sample")
IDENT = re.compile(r"\b(?:[A-Za-z]*\d{2,}[A-Za-z0-9\-]*|[A-Za-z0-9]+(?:-[A-Za-z0-9]+){2,})\b")
RARE_DF_FRAC = 0.5  # an identifier token must appear in fewer than this fraction of the corpus to count


def idents(text):
    return {m.group(0).lower() for m in IDENT.finditer(text)}


def load(path, id_field, text_field):
    rows = [json.loads(l) for l in open(path)]
    return {str(r[id_field]): r[text_field] for r in rows}


def build_df(corpus_texts):
    import collections
    df = collections.Counter()
    for t in corpus_texts:
        df.update(idents(t))
    return df, len(corpus_texts)


def leakage_score(query, doc, df, n, rare_df_frac=RARE_DF_FRAC):
    qi = {t for t in idents(query) if df.get(t, 0) / n < rare_df_frac}
    if not qi:
        return 0.0, qi, set()
    di = idents(doc)
    hit = qi & di
    return len(hit) / len(qi), qi, hit


def main():
    corpus = load(f"{R}/corpus.test.jsonl", "docid", "structured")
    train_corpus = load(f"{R}/corpus.train.jsonl", "docid", "structured")
    queries = load(f"{R}/queries.test.jsonl", "qid", "text")
    qrels = {l.split()[0]: l.split()[1] for l in list(open(f"{R}/qrels.test.tsv"))[1:]}

    df, n = build_df(list(corpus.values()) + list(train_corpus.values()))
    common = [t for t, c in df.most_common(10)]
    print(f"corpus size for df: {n}; most common identifier-like tokens (excluded as non-discriminative): {common}")

    rows = []
    for qid, q in queries.items():
        s, qi, hit = leakage_score(q, corpus[qrels[qid]], df, n)
        rows.append((qid, s, qi, hit))
    rows.sort(key=lambda r: r[1])
    print(f"\n{len(rows)} queries; leakage score = fraction of a query's *rare* (corpus-discriminative) identifier-like")
    print("tokens that appear verbatim in its own gold document.")
    for qid, s, qi, hit in rows:
        print(f"  {qid:>3s}  score={s:.2f}  rare idents in query={sorted(qi) or '-'}  matched in doc={sorted(hit) or '-'}")
    n2 = len(rows)
    low = [r[0] for r in rows[: n2 // 2]]
    high = [r[0] for r in rows[n2 // 2 :]]
    print("\nlow-leak qids:", low)
    print("high-leak qids:", high)
    return corpus, queries, qrels, {r[0]: r[1] for r in rows}, low, high


if __name__ == "__main__":
    main()
