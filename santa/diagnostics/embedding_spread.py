"""Embedding-collapse diagnostic: encode S1000D test corpus/queries exactly like index_xml.py and report how much
the representations actually vary across inputs."""
import json, sys, os
import numpy as np
import torch
SANTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SANTA)
sys.path.insert(0, SANTA)
sys.path.insert(0, os.path.join(SANTA, "evaluate_xml"))
from transformers import AutoTokenizer
from model import SModel
from santa_arguments import SantaArguments
from openmatch.arguments import ModelArguments
from index_xml import encode_all, load_jsonl

R = os.path.join(REPO, "retrieval", "S1000D_spec_sample")
dev = torch.device("cuda")
cid, ctxt = load_jsonl(f"{R}/corpus.test.jsonl", "docid", "structured")
qid, qtxt = load_jsonl(f"{R}/queries.test.jsonl", "qid", "text")
qrels = {}
for l in list(open(f"{R}/qrels.test.tsv"))[1:]:
    q, d, _ = l.split()
    qrels[q] = d


def stats(name, path, p_max_len=256):
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    m = SModel.build(model_args=ModelArguments(model_name_or_path=path), santa_args=SantaArguments(use_generate=False))
    m.to(dev).eval()
    P = encode_all(m, tok, ctxt, p_max_len, 21, dev, is_query=False).float()
    Q = encode_all(m, tok, qtxt, 50, 21, dev, is_query=True).float()
    del m
    torch.cuda.empty_cache()

    def cosmat(A):
        A = torch.nn.functional.normalize(A, dim=-1)
        S = A @ A.T
        return S[~torch.eye(len(A), dtype=bool)]

    pc, qc = cosmat(P), cosmat(Q)
    # relative spread: per-dim std across items divided by norm of the mean vector
    rel = lambda A: (A.std(0).norm() / A.mean(0).norm()).item()
    S = Q @ P.T
    idx = {d: i for i, d in enumerate(cid)}
    ranks = []
    for i, q in enumerate(qid):
        order = torch.argsort(S[i], descending=True).tolist()
        ranks.append(order.index(idx[qrels[q]]) + 1)
    mrr = np.mean([1 / r for r in ranks])
    # how much do scores vary across docs for one query vs across queries for one doc
    print(f"{name:34s} |p|={P.norm(dim=1).mean():8.1f} |q|={Q.norm(dim=1).mean():8.1f} "
          f"doc-doc cos mean={pc.mean():.4f} min={pc.min():.4f} | q-q cos mean={qc.mean():.4f} | "
          f"rel.spread doc={rel(P):.4f} q={rel(Q):.4f} | uniq docs={len(torch.unique(P.round(decimals=3), dim=0))}/21 | "
          f"score range/query={ (S.max(1).values - S.min(1).values).mean():.3f} | MRR={mrr:.4f} ranks={ranks}")


runs = os.path.join(SANTA, "runs")
for fam, hf in [("codet5", "Salesforce/codet5-base"), ("t5", "t5-base")]:
    stats(f"{fam} untrained base", hf)
    for src in ["BIKE_7", "s1kd-tools-doc"]:
        n = f"{src}_to_S1000D_spec_sample"
        stats(f"{fam} {src} pretrain-only", f"{runs}/pretrain/{fam}_s42/{n}/checkpoints")
        stats(f"{fam} {src} -> S1000D finetuned", f"{runs}/finetune/{fam}_s42/{n}/checkpoints")
