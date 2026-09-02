#!/usr/bin/env python3
"""
Check Poincaré hierarchy property for all Poincaré .pth.best files in xsd_models.

For each checkpoint:
- Loads embeddings and tag list
- Loads parent-child edges from xsd/proced_closure.csv
- Checks if child norm > parent norm for each edge
- Reports percentage of correct edges and lists a few violations
"""
import os
import glob
import torch
import numpy as np
import csv
import argparse
from find_closest_tags import poincare_dist

# Find all relevant Poincaré checkpoints
def find_poincare_checkpoints(root="/Users/apple/Desktop/paperrrrtranslation/poincare_xsd/xsd_models"):
    patterns = [
        os.path.join(root, "poincare", "*.pth.best"),
        os.path.join(root, "all_poincare", "*.pth.best"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)

def load_embeddings(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "embeddings" in ckpt:
        emb = ckpt["embeddings"]
    else:
        state = ckpt.get("model", ckpt)
        cand_keys = [
            k for k in state.keys()
            if "lt.weight" in k or "emb" in k or k.endswith("weight")
        ]
        if not cand_keys:
            raise KeyError("Could not find embeddings in checkpoint")
        emb = state[cand_keys[0]]
    emb = emb.cpu().detach().numpy() if hasattr(emb, "cpu") else np.asarray(emb)
    objects = ckpt.get("objects", None)
    if objects is not None:
        objects = [str(o) for o in objects]
    return emb, objects

def load_edges(edge_path, objects):
    name2idx = {str(v): i for i, v in enumerate(objects)}
    edges = []
    with open(edge_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 2:
                continue
            child, parent = row[0].strip(), row[1].strip()  # child, parent
            if parent in name2idx and child in name2idx:
                edges.append((name2idx[parent], name2idx[child]))  # 轉成 index
    return edges

def check_hierarchy(emb, edges, objects):
    norms = np.linalg.norm(emb, axis=1)
    out_of_ball = np.where(norms >= 1)[0]
    eps = 1e-5  # ε = 10^-5
    if len(out_of_ball) > 0:
        print(f"  WARNING: {len(out_of_ball)} embeddings have norm >= 1 (max norm: {norms.max():.6f}). Projecting back into unit ball with ε={eps}.")
        # Project out-of-ball vectors back into the ball
        for idx in out_of_ball:
            emb[idx] = emb[idx] / (norms[idx] + eps)
        # Recompute norms for hierarchy check
        norms = np.linalg.norm(emb, axis=1)
    else:
        print(f"  All embeddings are within the unit ball (max norm: {norms.max():.6f}).")

    # 檢查 child 是否比 parent 更靠近原點（Poincaré距離）
    origin = np.zeros(emb.shape[1])
    violations = []
    for parent, child in edges:
        d_parent = poincare_dist(emb[parent], origin)
        d_child = poincare_dist(emb[child], origin)
        if d_child < d_parent:
            violations.append((parent, child, d_parent, d_child))
    correct = len(edges) - len(violations)
    total = len(edges)
    error_rate = len(violations) / total if total > 0 else 0
    print(f"\n用Poincaré距離檢查階層：")
    print(f"  Edges checked: {total}")
    print(f"  Hierarchy correct: {correct}/{total} ({100*correct/total:.2f}%)")
    print(f"  Hierarchy error rate: {100*error_rate:.2f}%")
    if violations:
        print("  Example violations (up to 5):")
        for parent, child, d_parent, d_child in violations[:5]:
            print(f"    parent: {objects[parent]} (d={d_parent:.4f}), child: {objects[child]} (d={d_child:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Check Poincaré hierarchy property for a checkpoint.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Poincaré checkpoint (.pth or .pth.best)')
    parser.add_argument('--edges', type=str, default=os.path.join('xsd', 'proced_closure.csv'), help='Path to edge CSV file')
    args = parser.parse_args()

    print(f"Checking hierarchy for {args.checkpoint}")
    emb, objects = load_embeddings(args.checkpoint)
    if objects is None:
        print("  No objects found in checkpoint, aborting.")
        return
    edges = load_edges(args.edges, objects)
    if not edges:
        print("  No valid edges found, aborting.")
        return
    check_hierarchy(emb, edges, objects)

if __name__ == "__main__":
    main()

"""
python check_poincare_hierarchy.py --checkpoint ../xsd_models/all_poincare/all_xsd_2.pth.best --edges ../xsd/all_xsd_closure.csv
"""