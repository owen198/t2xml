import argparse
import json
import logging
import os
import sys

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SModel
from santa_arguments import SantaArguments
from openmatch.arguments import ModelArguments

logger = logging.getLogger(__name__)


def load_jsonl(path, id_field, text_field):
    ids, texts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.append(str(rec[id_field]))
            texts.append(rec[text_field])
    return ids, texts


def encode_all(model, tokenizer, texts, max_len, batch_size, device, is_query):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="query" if is_query else "corpus"):
        batch = tokenizer(
            texts[i:i + batch_size],
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            if is_query:
                _, reps = model.encode_query(batch)
            else:
                _, reps, _ = model.encode_passage(batch, labels=None)
        embs.append(reps.cpu())
    return torch.cat(embs, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Finetuned SANTA checkpoint directory")
    parser.add_argument("--corpus_path", type=str, required=True, help="retrieval/corpus.<split>.jsonl")
    parser.add_argument("--query_path", type=str, required=True, help="retrieval/queries.<split>.jsonl")
    parser.add_argument("--trec_save_path", type=str, required=True, help="Where to write the TREC-format run")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--q_max_len", type=int, default=50)
    parser.add_argument("--p_max_len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model_args = ModelArguments(model_name_or_path=args.model_name_or_path)
    # use_generate=False: index/retrieve only needs encoder-side representations,
    # never the generative (MEP) branch encode_passage() takes when it's True.
    santa_args = SantaArguments(use_generate=False)
    model = SModel.build(model_args=model_args, santa_args=santa_args)
    model.to(device)
    model.eval()

    corpus_ids, corpus_texts = load_jsonl(args.corpus_path, "docid", "structured")
    query_ids, query_texts = load_jsonl(args.query_path, "qid", "text")
    logger.info("corpus: %d docs, queries: %d", len(corpus_ids), len(query_ids))

    corpus_embs = encode_all(model, tokenizer, corpus_texts, args.p_max_len,
                              args.per_device_eval_batch_size, device, is_query=False)
    query_embs = encode_all(model, tokenizer, query_texts, args.q_max_len,
                             args.per_device_eval_batch_size, device, is_query=True)

    # IndexFlatIP: exact (not approximate) inner-product search. SModel isn't
    # normalized (see model.py), so plain inner product matches the
    # torch.matmul(q_reps, p_reps.T) scoring used in training and in
    # evaluate_code_finetune.py/evaluate_shop_pretrain.py.
    topk = min(args.topk, len(corpus_ids))
    corpus_np = np.ascontiguousarray(corpus_embs.numpy(), dtype="float32")
    query_np = np.ascontiguousarray(query_embs.numpy(), dtype="float32")
    index = faiss.IndexFlatIP(corpus_np.shape[1])
    index.add(corpus_np)
    top_scores, top_indices = index.search(query_np, topk)

    os.makedirs(os.path.dirname(args.trec_save_path), exist_ok=True)
    with open(args.trec_save_path, "w") as f:
        for qi, qid in enumerate(query_ids):
            ranked = zip(top_indices[qi].tolist(), top_scores[qi].tolist())
            for rank, (idx, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {corpus_ids[idx]} {rank} {score:.6f} santa\n")

    logger.info("Wrote TREC run to %s", args.trec_save_path)


if __name__ == "__main__":
    main()
