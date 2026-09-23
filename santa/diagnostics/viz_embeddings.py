"""Visualize SANTA query/passage embeddings with PCA or t-SNE.

Projects each checkpoint's embeddings to 2D independently and plots one
subplot per checkpoint (passages vs. queries, optionally linked to their gold
passage) so collapse/separation can be compared side by side.

Each --model PATH can be either a live checkpoint dir / HF model id (gets
encoded on the fly, needs a GPU and the retrieval jsonl text) or a
results/**/embeddings.npz file produced by shell/index-xml.sh's
--save_embeddings hook (loaded straight off disk, no model or GPU needed --
this is the path to use once you've deleted the checkpoint to save space).

Examples:
  # live checkpoints
  python viz_embeddings.py \
    --model "codet5 base" Salesforce/codet5-base \
    --model "BIKE_7 -> S1000D finetuned" ../runs/finetune/codet5_s42/BIKE_7_to_S1000D_spec_sample/checkpoints \
    --method tsne --links --out embeddings_tsne.png

  # cached embeddings, no checkpoint needed
  python viz_embeddings.py \
    --model "BIKE_7 -> FOSSIG" ../../results/BIKE_7_to_FOSSIG/embeddings.npz \
    --model "BIKE_7 -> S1000D" ../../results/BIKE_7_to_S1000D_spec_sample/embeddings.npz \
    --method tsne --out embeddings_tsne.png

Note: --links assumes every --model entry's ids come from the same
--retrieval-dir/--split (i.e. you're comparing checkpoints of one experiment
pair). Comparing cached embeddings.npz files from *different* experiment
pairs works for the scatter itself, but drop --links -- ids like "doc_0" are
reused across pairs and would draw meaningless cross-pair connections.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluate_xml"))
from model import SModel
from santa_arguments import SantaArguments
from openmatch.arguments import ModelArguments
from index_xml import encode_all, load_jsonl


def load_qrels(path):
    qrels = {}
    for line in list(open(path))[1:]:
        q, d, _ = line.split()
        qrels[q] = d
    return qrels


def encode_checkpoint(path, ctxt, qtxt, p_max_len, q_max_len, batch_size, device):
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    m = SModel.build(model_args=ModelArguments(model_name_or_path=path), santa_args=SantaArguments(use_generate=False))
    m.to(device).eval()
    P = encode_all(m, tok, ctxt, p_max_len, batch_size, device, is_query=False).float()
    Q = encode_all(m, tok, qtxt, q_max_len, batch_size, device, is_query=True).float()
    del m
    torch.cuda.empty_cache()
    return P.numpy(), Q.numpy()


def load_cache(path):
    data = np.load(path, allow_pickle=False)
    return (data["corpus_embs"].astype("float32"), data["query_embs"].astype("float32"),
            data["corpus_ids"].tolist(), data["query_ids"].tolist())


def project(P, Q, method, perplexity, n_iter, random_state):
    X = np.concatenate([P, Q], axis=0)
    if method == "pca":
        X2 = PCA(n_components=2, random_state=random_state).fit_transform(X)
    else:
        perplexity = min(perplexity, max(1.0, X.shape[0] - 1))
        X2 = TSNE(n_components=2, random_state=random_state, perplexity=perplexity,
                  max_iter=n_iter, init="pca").fit_transform(X)
    return X2[:len(P)], X2[len(P):]


def doc_cos_mean(P):
    A = torch.nn.functional.normalize(torch.from_numpy(P), dim=-1)
    S = A @ A.T
    off_diag = S[~torch.eye(len(A), dtype=bool)]
    return off_diag.mean().item()


def plot_checkpoint(ax, label, P2, Q2, cid, qid, qrels, links):
    doc_idx = {d: i for i, d in enumerate(cid)}
    if links:
        for i, q in enumerate(qid):
            gold = qrels.get(q)
            if gold is None or gold not in doc_idx:
                continue
            j = doc_idx[gold]
            ax.plot([Q2[i, 0], P2[j, 0]], [Q2[i, 1], P2[j, 1]],
                    color="gray", linewidth=0.4, alpha=0.35, zorder=1)
    ax.scatter(P2[:, 0], P2[:, 1], s=18, c="#4C78A8", alpha=0.8, label="passage", zorder=2)
    ax.scatter(Q2[:, 0], Q2[:, 1], s=28, c="#F58518", alpha=0.9, marker="^", label="query", zorder=3)
    ax.set_title(label, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", "-m", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True,
                        help="Label + (HF model id / checkpoint dir / cached embeddings.npz); repeat for multiple")
    parser.add_argument("--retrieval-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "retrieval", "S1000D_spec_sample"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", choices=["pca", "tsne"], default="tsne")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--p-max-len", type=int, default=256)
    parser.add_argument("--q-max-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--links", action="store_true", help="Draw a line from each query to its gold passage")
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument("--out", "-o", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    R = args.retrieval_dir
    needs_live = any(not path.endswith(".npz") for _, path in args.model)
    default_cid = default_qid = ctxt = qtxt = None
    qrels = {}
    if needs_live or args.links:
        cid_, ctxt = load_jsonl(f"{R}/corpus.{args.split}.jsonl", "docid", "structured")
        qid_, qtxt = load_jsonl(f"{R}/queries.{args.split}.jsonl", "qid", "text")
        default_cid, default_qid = cid_, qid_
        qrels = load_qrels(f"{R}/qrels.{args.split}.tsv")

    n = len(args.model)
    ncols = min(args.ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for i, (label, path) in enumerate(args.model):
        if path.endswith(".npz"):
            print(f"[{i + 1}/{n}] loading cache {label} ({path})", file=sys.stderr)
            P, Q, cid, qid = load_cache(path)
        else:
            print(f"[{i + 1}/{n}] encoding {label} ({path})", file=sys.stderr)
            P, Q = encode_checkpoint(path, ctxt, qtxt, args.p_max_len, args.q_max_len, args.batch_size, device)
            cid, qid = default_cid, default_qid
        P2, Q2 = project(P, Q, args.method, args.tsne_perplexity, args.tsne_n_iter, args.random_state)
        ax = axes[i // ncols][i % ncols]
        plot_checkpoint(ax, f"{label}\ndoc-doc cos={doc_cos_mean(P):.3f}", P2, Q2, cid, qid, qrels, args.links)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(f"SANTA embeddings ({args.method}, {os.path.basename(R)}/{args.split})")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
