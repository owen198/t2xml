import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_xml_finetune import find_checkpoints, select_best

logger = logging.getLogger(__name__)


def load_sda_dev_pairs(path):
    # Self-referential codebase: row i's `text` should retrieve row i's own
    # `structured` snippet, using every row in the file as the "corpus" --
    # the same convention CodeSearchNet's own pretrain-dev eval uses (one
    # valid.jsonl file passed as both --eval_data_file and --codebase_file;
    # see SANTA's shell/dev_code_pretrain.sh). This is the pretraining task's
    # own SDA dev pairs, not the Stage 3 retrieval benchmark (retrieval/*.jsonl)
    # -- picking a pretrain checkpoint shouldn't peek at the finetune benchmark.
    texts, structured = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append(rec["text"])
            structured.append(rec["structured"])
    ids = [str(i) for i in range(len(texts))]
    return ids, texts, structured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="pretrain.sh's --output_dir (contains checkpoint-N subdirs and/or is itself a final save)")
    parser.add_argument("--eval_path", type=str, required=True,
                        help="pretrain/sda_pairs.dev.jsonl -- used as both queries and codebase")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--q_max_len", type=int, default=50)
    parser.add_argument("--p_max_len", type=int, default=256)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--results_path", type=str, default=None,
                        help="Optional: also write best-checkpoint/per-checkpoint MRR as JSON here "
                             "(e.g. results/experiment-1/best_dev_pretrain.json)")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = find_checkpoints(args.model_path)
    if not candidates:
        raise ValueError(f"No checkpoints (openmatch_config.json) found under {args.model_path}")
    logger.info("Evaluating %d checkpoints: %s", len(candidates), candidates)

    ids, texts, structured = load_sda_dev_pairs(args.eval_path)
    qrels = {i: {i} for i in ids}
    logger.info("SDA dev pairs (self-referential codebase): %d", len(ids))

    select_best(args.model_path, candidates, structured, ids, texts, ids, qrels, args, device,
                results_path=args.results_path)


if __name__ == "__main__":
    main()
