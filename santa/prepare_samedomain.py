#!/usr/bin/env python3
"""Tokenize the same-domain split (datasets/build_samedomain_split.py output) into the four combinations
needed for the swap experiment. Reuses prepare_data.py's own encode/build_* functions so the format
exactly matches every other experiment's data_*/ folder."""
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import build_pretrain_split, build_finetune_split

import os
REPO_ROOT = Path(__file__).resolve().parent.parent
SUFFIX = os.environ.get("SAMEDOMAIN_SUFFIX", "")
OUT = Path(__file__).resolve().parent / f"data_samedomain{SUFFIX}"
PRETRAIN_ROOT = REPO_ROOT / f"pretrain_samedomain{SUFFIX}"
RETRIEVAL_ROOT = REPO_ROOT / f"retrieval_samedomain{SUFFIX}"

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=False)

combos = {
    "pretrainA_finetuneB": (PRETRAIN_ROOT / "halfA/S1000D_spec_sample",
                             RETRIEVAL_ROOT / "halfB/S1000D_spec_sample"),
    "pretrainB_finetuneA": (PRETRAIN_ROOT / "halfB/S1000D_spec_sample",
                             RETRIEVAL_ROOT / "halfA/S1000D_spec_sample"),
}
finetune_only = {
    "finetuneB_only": RETRIEVAL_ROOT / "halfB/S1000D_spec_sample",
    "finetuneA_only": RETRIEVAL_ROOT / "halfA/S1000D_spec_sample",
}

for name, (pretrain_dir, retrieval_dir) in combos.items():
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        build_pretrain_split(tokenizer, split, pretrain_dir, out_dir)
        build_finetune_split(tokenizer, split, retrieval_dir, out_dir)

for name, retrieval_dir in finetune_only.items():
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        build_finetune_split(tokenizer, split, retrieval_dir, out_dir)
