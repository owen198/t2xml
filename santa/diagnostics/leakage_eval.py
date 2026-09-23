"""Score BM25 and a finetuned dense checkpoint per-query on the S1000D test set, split into the
low-leak / high-leak halves from leakage_split.py. Read-only except for loading a checkpoint you already
trained under santa/runs/finetune/<tag>/<pair>/checkpoints."""
import json, math, os, re, sys, collections
import numpy as np
import torch
SANTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SANTA)
sys.path.insert(0, SANTA)
sys.path.insert(0, os.path.join(SANTA, "evaluate_xml"))
sys.path.insert(0, os.path.join(SANTA, "diagnostics"))
from leakage_split import load, main as split_main
from transformers import AutoTokenizer
from model import SModel
from santa_arguments import SantaArguments
from openmatch.arguments import ModelArguments
from index_xml import encode_all

R = os.path.join(REPO, "retrieval", "S1000D_spec_sample")
tok_re = lambda s: re.findall(r"[A-Za-z0-9_]+", s.lower())


def bm25_ranks(corpus, cid, train_texts, queries, qrels):
    df = collections.Counter()
    all_texts = train_texts + [corpus[c] for c in cid]
    for d in all_texts:
        df.update(set(tok_re(d)))
    N = len(all_texts)
    idf = lambda t: math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
    docs = [collections.Counter(tok_re(corpus[c])) for c in cid]
    L = [sum(c.values()) for c in docs]
    avg = sum(L) / len(L)

    def score(q, i, k1=1.2, b=0.75):
        return sum(
            idf(t) * docs[i][t] * (k1 + 1) / (docs[i][t] + k1 * (1 - b + b * L[i] / avg))
            for t in set(tok_re(q)) if t in docs[i]
        )

    ranks = {}
    for qid, qtext in queries.items():
        s = [score(qtext, i) for i in range(len(cid))]
        order = np.argsort(-np.array(s), kind="stable").tolist()
        ranks[qid] = order.index(cid.index(qrels[qid])) + 1
    return ranks


def dense_ranks(ckpt_path, corpus, cid, queries, qrels, p_max_len=1024):
    tok = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
    m = SModel.build(model_args=ModelArguments(model_name_or_path=ckpt_path, pooling="mean"),
                      santa_args=SantaArguments(use_generate=False))
    dev = torch.device("cuda")
    m.to(dev).eval()
    ctxt = [corpus[c] for c in cid]
    qid_list = list(queries)
    qtxt = [queries[q] for q in qid_list]
    P = encode_all(m, tok, ctxt, p_max_len, 21, dev, is_query=False).float()
    Q = encode_all(m, tok, qtxt, 50, 21, dev, is_query=True).float()
    n = torch.nn.functional.normalize
    S = n(Q, dim=-1) @ n(P, dim=-1).T
    ranks = {}
    for i, qid in enumerate(qid_list):
        order = torch.argsort(S[i], descending=True).tolist()
        ranks[qid] = order.index(cid.index(qrels[qid])) + 1
    del m
    torch.cuda.empty_cache()
    return ranks


def mrr(ranks, subset):
    return float(np.mean([1 / ranks[q] for q in subset]))


def main(ckpt_paths):
    corpus, queries, qrels, scores, low, high = split_main()
    cid = list(corpus)
    train_texts = list(load(f"{R}/corpus.train.jsonl", "docid", "structured").values())
    br = bm25_ranks(corpus, cid, train_texts, queries, qrels)
    print(f"\nBM25          low-leak (n={len(low)}) MRR={mrr(br, low):.4f}   high-leak (n={len(high)}) MRR={mrr(br, high):.4f}   all MRR={mrr(br, list(queries)):.4f}")
    for name, ckpt in ckpt_paths.items():
        dr = dense_ranks(ckpt, corpus, cid, queries, qrels)
        print(f"{name:14s} low-leak (n={len(low)}) MRR={mrr(dr, low):.4f}   high-leak (n={len(high)}) MRR={mrr(dr, high):.4f}   all MRR={mrr(dr, list(queries)):.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", nargs=2, metavar=("NAME", "PATH"), default=[])
    args = ap.parse_args()
    main({n: p for n, p in args.ckpt})
