import argparse
import json
import logging
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)

def load_qrels(qrels_path):
    # qrels.tsv: query-id \t corpus-id \t score, with a header row
    # (see datasets/build_retrieval_dataset.py).
    qrels = {}
    with open(qrels_path) as f:
        next(f)  # header
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(docid)
    return qrels

def calculate_mrr(data_samples, qrels):
    ranks = []
    for item in data_samples:
        relevant = qrels.get(item['query'], set())
        rank = 0
        find = False
        for docid in item['doc'][:100]:
            # MRR@100
            if find is False:
                rank += 1
            if docid in relevant:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)
    result = {
        "eval_mrr": float(np.mean(ranks))
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_save_path", type=str, default=None, help="The path to the inference.trec")
    parser.add_argument("--qrels_path", type=str, default=None, help="The path to qrels.{split}.tsv")
    parser.add_argument("--results_path", type=str, default=None,
                        help="Optional: also write the result dict as JSON here (e.g. results/experiment-1/eval.json)")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # t2xml's qid/docid are separate id spaces (q_N / doc_N -- see
    # datasets/build_retrieval_dataset.py), unlike CodeSearchNet's shared-url
    # convention that evaluate_code.py relies on, so ground truth here comes
    # from qrels rather than query_id == passage_id.
    qrels = load_qrels(args.qrels_path)
    # read csv
    new_columns = ['query_id', 'Q0', 'passage_id', 'rank', 'score', 'tool']
    df = pd.read_csv(args.trec_save_path, delimiter=" ", header=None, on_bad_lines='skip')
    df.columns = new_columns
    query2id = {}
    data_samples = []
    q_last = ''
    # get data samples for mrr
    for (index, row) in tqdm(df.iterrows()):
        if q_last != row['query_id']:
            query2id = {}
            query2id["query"] = row['query_id']
            query2id['doc'] = []
            q_last = row['query_id']
            data_samples.append(query2id)
        query2id['doc'].append(row['passage_id'])

    # The mrr code are copied and modified from santa/evaluate_code/evaluate_code.py
    # (itself from Unxicode https://github.com/microsoft/CodeBERT/blob/master/UniXcoder/downstream-tasks/code-search/run.py)
    result = calculate_mrr(data_samples, qrels)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 3)))

    if args.results_path:
        os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump({**result, "trec_save_path": args.trec_save_path, "qrels_path": args.qrels_path},
                       f, indent=2)
        logger.info("Wrote results to %s", args.results_path)

if __name__ == "__main__":
    main()
