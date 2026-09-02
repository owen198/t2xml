import argparse
import glob
import json
import logging
import os
import shutil
import sys

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer

SANTA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SANTA_DIR)
sys.path.insert(0, os.path.join(SANTA_DIR, "evaluate_xml"))
from model import SModel
from santa_arguments import SantaArguments
from openmatch.arguments import ModelArguments
from index_xml import load_jsonl, encode_all
from evaluate_xml import load_qrels, calculate_mrr

logger = logging.getLogger(__name__)


def find_checkpoints(model_path):
    # Numbered checkpoint-N subdirs are periodic `save_steps` saves. The final,
    # most-converged save from train_santa.py's trainer.save_model() call lands
    # directly in model_path itself, not in its own checkpoint-N subdir (unlike
    # SANTA's own pretrain-code.sh/finetune-code.sh, which materialize it into
    # one with a mkdir+cp step) -- so model_path itself is a candidate too.
    candidates = []
    if os.path.exists(os.path.join(model_path, "openmatch_config.json")):
        candidates.append(model_path)
    step_dirs = glob.glob(os.path.join(model_path, "checkpoint-*"))
    step_dirs.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
    candidates.extend(step_dirs)
    return candidates


def evaluate_checkpoint(checkpoint, corpus_texts, corpus_ids, query_texts, query_ids,
                         qrels, args, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model_args = ModelArguments(model_name_or_path=checkpoint)
    santa_args = SantaArguments(use_generate=False)
    model = SModel.build(model_args=model_args, santa_args=santa_args)
    model.to(device)
    model.eval()

    corpus_embs = encode_all(model, tokenizer, corpus_texts, args.p_max_len,
                              args.per_device_eval_batch_size, device, is_query=False)
    query_embs = encode_all(model, tokenizer, query_texts, args.q_max_len,
                             args.per_device_eval_batch_size, device, is_query=True)

    topk = min(args.topk, len(corpus_ids))
    corpus_np = np.ascontiguousarray(corpus_embs.numpy(), dtype="float32")
    query_np = np.ascontiguousarray(query_embs.numpy(), dtype="float32")
    index = faiss.IndexFlatIP(corpus_np.shape[1])
    index.add(corpus_np)
    _, top_indices = index.search(query_np, topk)

    data_samples = [
        {"query": qid, "doc": [corpus_ids[i] for i in top_indices[qi]]}
        for qi, qid in enumerate(query_ids)
    ]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return calculate_mrr(data_samples, qrels)["eval_mrr"]


def select_best(model_path, candidates, corpus_texts, corpus_ids, query_texts, query_ids,
                 qrels, args, device, results_path=None):
    """Evaluate every candidate checkpoint, log each score, and copy the best
    one to model_path/best_dev (skipping the copy if model_path itself won --
    it's already directly usable). Shared by evaluate_xml_finetune.py and
    evaluate_xml_pretrain.py; only what counts as "corpus"/"query"/"qrels"
    differs between the two stages."""
    best_dev_mrr = 0.0
    best_checkpoint = None
    per_checkpoint = {}
    for checkpoint in candidates:
        eval_mrr = evaluate_checkpoint(checkpoint, corpus_texts, corpus_ids, query_texts,
                                        query_ids, qrels, args, device)
        per_checkpoint[checkpoint] = eval_mrr
        logger.info("checkpoint %s: eval_mrr=%.4f (best so far: %.4f)",
                    checkpoint, eval_mrr, best_dev_mrr)
        if eval_mrr > best_dev_mrr:
            best_dev_mrr = eval_mrr
            best_checkpoint = checkpoint

    logger.info("***** Best checkpoint *****")
    logger.info("  %s: eval_mrr=%.4f", best_checkpoint, best_dev_mrr)

    best_path = os.path.join(model_path, "best_dev")
    if best_checkpoint == model_path:
        # Don't shutil.copytree(model_path, best_path) here -- model_path also
        # contains the checkpoint-N subdirs (and a stale best_dev from a prior
        # run), so a full recursive copy would duplicate all of those into
        # best_dev too. Only the final save's own top-level files belong here.
        os.makedirs(best_path, exist_ok=True)
        for name in os.listdir(model_path):
            src = os.path.join(model_path, name)
            if os.path.isfile(src):
                shutil.copy2(src, best_path)
        logger.info("Best checkpoint is model_path itself; copied its top-level files to %s",
                    best_path)
    else:
        shutil.copytree(best_checkpoint, best_path, dirs_exist_ok=True)
        logger.info("Copied best checkpoint to %s", best_path)

    if results_path:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({
                "best_checkpoint": best_checkpoint,
                "best_dev_mrr": best_dev_mrr,
                "per_checkpoint": per_checkpoint,
            }, f, indent=2)
        logger.info("Wrote results to %s", results_path)

    return best_checkpoint, best_dev_mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="finetune.sh's --output_dir (contains checkpoint-N subdirs and/or is itself a final save)")
    parser.add_argument("--corpus_path", type=str, required=True, help="retrieval/corpus.dev.jsonl")
    parser.add_argument("--query_path", type=str, required=True, help="retrieval/queries.dev.jsonl")
    parser.add_argument("--qrels_path", type=str, required=True, help="retrieval/qrels.dev.tsv")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--q_max_len", type=int, default=50)
    parser.add_argument("--p_max_len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--results_path", type=str, default=None,
                        help="Optional: also write best-checkpoint/per-checkpoint MRR as JSON here "
                             "(e.g. results/experiment-1/best_dev_finetune.json)")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = find_checkpoints(args.model_path)
    if not candidates:
        raise ValueError(f"No checkpoints (openmatch_config.json) found under {args.model_path}")
    logger.info("Evaluating %d checkpoints: %s", len(candidates), candidates)

    corpus_ids, corpus_texts = load_jsonl(args.corpus_path, "docid", "structured")
    query_ids, query_texts = load_jsonl(args.query_path, "qid", "text")
    qrels = load_qrels(args.qrels_path)
    logger.info("corpus: %d docs, queries: %d", len(corpus_ids), len(query_ids))

    select_best(args.model_path, candidates, corpus_texts, corpus_ids, query_texts, query_ids,
                qrels, args, device, results_path=args.results_path)


if __name__ == "__main__":
    main()
