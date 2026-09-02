#!/usr/bin/env python3
"""
Find the top 5 closest tags to a given tag in Poincaré and Euclidean embedding spaces.

Usage:
  python find_closest_tags.py --tag TAG --poincare-checkpoint PATH --euclidean-checkpoint PATH

Example:
  python find_closest_tags.py --tag my_tag \
    --poincare-checkpoint xsd_models/poincare/proced_xsd_2.pth \
    --euclidean-checkpoint xsd_models/euclidean/e_proced_xsd_2.pth
"""
import argparse
import numpy as np
import torch
import sys

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

def poincare_dist(u, v, eps=1e-5):
    u_norm = np.sum(u ** 2)
    v_norm = np.sum(v ** 2)
    u_norm = np.clip(u_norm, 0, 1 - eps)
    v_norm = np.clip(v_norm, 0, 1 - eps)
    diff = np.sum((u - v) ** 2)
    alpha = 1 - u_norm
    beta = 1 - v_norm
    x = 1 + 2 * diff / (alpha * beta)
    z = np.sqrt(np.maximum(x ** 2 - 1, 0))
    return np.log(x + z)

def find_topk(emb, objects, tag, dist_fn, k=5):
    if objects is None:
        raise ValueError("No tag list found in checkpoint.")
    if tag not in objects:
        raise ValueError(f"Tag '{tag}' not found in objects list.")
    idx = objects.index(tag)
    v = emb[idx]
    dists = np.array([dist_fn(v, emb[i]) if i != idx else np.inf for i in range(len(objects))])
    topk_idx = np.argsort(dists)[:k]
    return [(objects[i], float(dists[i])) for i in topk_idx]

def main():
    parser = argparse.ArgumentParser(description="Find closest tags in embedding spaces.")
    parser.add_argument("--tag", required=True, help="Tag to search for.")
    parser.add_argument("--poincare-checkpoint", required=True, help="Path to Poincaré checkpoint (.pth)")
    parser.add_argument("--euclidean-checkpoint", required=True, help="Path to Euclidean checkpoint (.pth)")
    parser.add_argument("--topk", type=int, default=5, help="Number of closest tags to show.")
    args = parser.parse_args()

    # Poincaré
    print(f"\n[Poincaré] Searching in {args.poincare_checkpoint}")
    emb_p, obj_p = load_embeddings(args.poincare_checkpoint)
    try:
        topk_p = find_topk(emb_p, obj_p, args.tag, poincare_dist, k=args.topk)
        for i, (tag, dist) in enumerate(topk_p, 1):
            print(f"{i}. {tag}\t(dist={dist:.6f})")
    except Exception as e:
        print(f"Error: {e}")


    # Euclidean
    print(f"\n[Euclidean] Searching in {args.euclidean_checkpoint}")
    emb_e, obj_e = load_embeddings(args.euclidean_checkpoint)
    try:
        topk_e = find_topk(emb_e, obj_e, args.tag, lambda u, v: np.linalg.norm(u - v), k=args.topk)
        for i, (tag, dist) in enumerate(topk_e, 1):
            print(f"{i}. {tag}\t(dist={dist:.6f})")
    except Exception as e:
        print(f"Error: {e}")

    # Output the embedding of the tag
    print(f"\nEmbedding for tag '{args.tag}':")
    if obj_p and args.tag in obj_p:
        idx_p = obj_p.index(args.tag)
        print(f"[Poincaré] {emb_p[idx_p]}")
    else:
        print("[Poincaré] Tag not found.")
    if obj_e and args.tag in obj_e:
        idx_e = obj_e.index(args.tag)
        print(f"[Euclidean] {emb_e[idx_e]}")
    else:
        print("[Euclidean] Tag not found.")

if __name__ == "__main__":
    main()


#python find_closest_tags.py --tag dmodule --poincare-checkpoint /Users/apple/Desktop/paperrrrtranslation/poincare_xsd/xsd_models/poincare/proced_xsd_2.pth --euclidean-checkpoint /Users/apple/Desktop/paperrrrtranslation/poincare_xsd/xsd_models/euclidean/e_proced_xsd_2.pth