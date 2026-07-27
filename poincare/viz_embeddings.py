#!/usr/bin/env python3
"""Visualize saved embedding checkpoints.

Examples:
  python viz_embeddings.py --checkpoint mammals.pth --method pca --annotate 20 --out mammals.png

  python viz_embeddings.py \
    --checkpoint xsd_models/poincare/xsd_2.pth \
    --style wn \
    --edges xsd/xsd_closure.csv \
    --edge-format names \
    --radial \
    --fit-to-disk \
    --annotate 10 \
    --label-mode top-degree \
    --max-edges 250 \
    --edge-alpha 0.03 \
    --edge-width 0.2 \
    --node-size-by depth \
    --min-node-size 6 \
    --max-node-size 120 \
    --out xsd_cleaner.png
"""

import argparse
import csv
import re
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

try:
    from adjustText import adjust_text as _adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False


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
    return emb, ckpt.get("objects", None)


def poincare_distances(emb, eps=1e-5):
    sqnorms = np.sum(emb ** 2, axis=1)
    sqnorms = np.clip(sqnorms, 0, 1 - eps)
    diff = emb[:, np.newaxis, :] - emb[np.newaxis, :, :]
    sqdist = np.sum(diff ** 2, axis=2)
    alpha = 1 - sqnorms[:, np.newaxis]
    beta = 1 - sqnorms[np.newaxis, :]
    x = 1 + 2 * sqdist / (alpha * beta)
    z = np.sqrt(np.maximum(x ** 2 - 1, 0))
    return np.log(x + z)


def project(emb, method="pca", random_state=42, **kwargs):
    if emb.shape[1] == 2:
        return emb
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(emb)
    if method == "tsne":
        return TSNE(n_components=2, random_state=random_state, **kwargs).fit_transform(emb)
    if method == "poincare-mds":
        dist_matrix = poincare_distances(emb)
        return MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            **kwargs,
        ).fit_transform(dist_matrix)
    raise ValueError(f"Unknown method: {method}")


def clean_label(label):
    s = str(label)
    return re.sub(r"\.[nvar]\.[0-9]+$", "", s)


def load_edges(path, objects=None, fmt="auto"):
    pairs = []
    name2idx = None
    if objects is not None:
        name2idx = {str(v): i for i, v in enumerate(objects)}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            row = [c.strip() for c in row if c is not None]
            if len(row) < 2:
                continue
            if row[0].startswith("#"):
                continue
            a, b = row[0], row[1]
            if a == "id1" and b == "id2":
                continue

            if fmt == "indices":
                try:
                    pairs.append((int(a), int(b)))
                except ValueError:
                    pass
                continue

            if fmt == "names":
                if name2idx is not None and a in name2idx and b in name2idx:
                    pairs.append((name2idx[a], name2idx[b]))
                continue

            try:
                pairs.append((int(a), int(b)))
            except ValueError:
                if name2idx is not None and a in name2idx and b in name2idx:
                    pairs.append((name2idx[a], name2idx[b]))
    return pairs


def sample_edges(edges, max_edges=None, seed=42):
    if edges is None:
        return None
    deduped = list(dict.fromkeys(edges))
    if max_edges is None or max_edges <= 0 or len(deduped) <= max_edges:
        return deduped
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(deduped), size=max_edges, replace=False))
    return [deduped[i] for i in idx]


def compute_roots_and_children(edges, n_nodes=None):
    children = {}
    has_parent = set()
    for p, c in edges:
        children.setdefault(p, []).append(c)
        children.setdefault(c, [])
        has_parent.add(c)
    if n_nodes is not None:
        for i in range(n_nodes):
            children.setdefault(i, [])
    roots = [n for n in children if n not in has_parent]
    if not roots:
        roots = [n for n, ch in children.items() if ch]
    return roots, children


def compute_depths(roots, children):
    depth = {n: None for n in children}
    q = deque()
    for r in roots:
        depth[r] = 0
        q.append(r)
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            if depth[v] is None or depth[v] > depth[u] + 1:
                depth[v] = depth[u] + 1
                q.append(v)
    finite = [d for d in depth.values() if d is not None]
    maxd = max(finite) if finite else 0
    for k, v in depth.items():
        if v is None:
            depth[k] = maxd + 1
    return depth


def reduce_transitive_closure_edges(edges):
    """Approximate transitive reduction for parent->descendant closure edges.

    Keeps edge (p, c) only when there is no intermediate node m such that
    p->m and m->c are both present.
    """
    if not edges:
        return []

    children = {}
    for p, c in edges:
        children.setdefault(p, set()).add(c)
        children.setdefault(c, set())

    reduced = []
    for p, ch in children.items():
        if not ch:
            continue
        ch_list = list(ch)
        for c in ch_list:
            is_transitive = False
            for m in ch_list:
                if m == c:
                    continue
                if c in children.get(m, set()):
                    is_transitive = True
                    break
            if not is_transitive:
                reduced.append((p, c))
    return reduced


def subtree_size(node, children, memo, visiting=None):
    if node in memo:
        return memo[node]
    if visiting is None:
        visiting = set()
    if node in visiting:
        return 0
    visiting.add(node)
    size = 1
    for child in children.get(node, []):
        size += subtree_size(child, children, memo, visiting)
    visiting.remove(node)
    memo[node] = size
    return size


def assign_radial_positions(roots, children, depth, radius_scale=0.95):
    memo = {}
    for n in children:
        subtree_size(n, children, memo)

    max_depth = max(depth.values()) if depth else 1
    nodes = sorted(children.keys())
    idx_map = {n: i for i, n in enumerate(nodes)}
    X = np.zeros((len(nodes), 2), dtype=float)

    # Iterative variant of layout() to avoid Python recursion limits on deep graphs.
    # For performance on large closure graphs, avoid per-path set allocations and
    # traverse only depth-increasing edges.
    stack = [(roots, 0.0, 2 * np.pi, 0)]
    while stack:
        node_list, start_angle, end_angle, current_depth = stack.pop()
        if not node_list:
            continue

        angle_span = end_angle - start_angle
        total = sum(max(memo.get(n, 1), 1) for n in node_list)
        acc = start_angle

        deferred = []
        for n in node_list:
            share = max(memo.get(n, 1), 1) / total if total > 0 else 1.0 / len(node_list)
            sub_start = acc
            sub_end = acc + angle_span * share
            angle = 0.5 * (sub_start + sub_end)
            radius = (current_depth / (max_depth + 1.0)) * radius_scale
            X[idx_map[n], 0] = radius * np.cos(angle)
            X[idx_map[n], 1] = radius * np.sin(angle)

            child_nodes = [
                c for c in children.get(n, [])
                if depth.get(c, current_depth + 1) > depth.get(n, current_depth)
            ]
            if child_nodes:
                deferred.append((child_nodes, sub_start, sub_end, current_depth + 1))
            acc = sub_end

        # Push in reverse so processing order stays left-to-right.
        for item in reversed(deferred):
            stack.append(item)

    return X, nodes


def choose_annotation_indices(n_points, annotate, edges=None, mode="first"):
    if annotate <= 0 or n_points <= 0:
        return []
    annotate = min(annotate, n_points)
    if mode == "top-degree" and edges is not None:
        degree = np.zeros(n_points, dtype=int)
        for i, j in edges:
            if 0 <= i < n_points:
                degree[i] += 1
            if 0 <= j < n_points:
                degree[j] += 1
        return np.argsort(-degree, kind="stable")[:annotate].tolist()
    return list(range(annotate))


def compute_node_sizes(n_points, mode="fixed", depth_values=None, edges=None,
                       min_size=10.0, max_size=80.0):
    if mode == "depth" and depth_values is not None:
        depth_values = np.asarray(depth_values, dtype=float)
        max_depth = depth_values.max() if len(depth_values) else 0
        if max_depth <= 0:
            return np.full(n_points, max_size, dtype=float)
        scaled = 1.0 - np.clip(depth_values / max_depth, 0.0, 1.0)
        return min_size + scaled * (max_size - min_size)

    if mode == "degree" and edges is not None:
        degree = np.zeros(n_points, dtype=float)
        for i, j in edges:
            if 0 <= i < n_points:
                degree[i] += 1
            if 0 <= j < n_points:
                degree[j] += 1
        max_degree = degree.max() if len(degree) else 0
        if max_degree <= 0:
            return np.full(n_points, min_size, dtype=float)
        scaled = degree / max_degree
        return min_size + scaled * (max_size - min_size)

    if mode == "children" and edges is not None:
        # Number of outgoing edges from each node (interpreted as child count
        # when edges are in parent->child direction).
        child_count = np.zeros(n_points, dtype=float)
        for i, _ in edges:
            if 0 <= i < n_points:
                child_count[i] += 1
        max_children = child_count.max() if len(child_count) else 0
        if max_children <= 0:
            return np.full(n_points, min_size, dtype=float)
        scaled = child_count / max_children
        return min_size + scaled * (max_size - min_size)

    return np.full(n_points, min_size, dtype=float)


def draw_edges(ax, X, edges, edge_alpha=0.08, edge_width=0.3, edge_color="gray",
               edge_highlight=False, edge_color_values=None, edge_cmap="plasma"):
    if edges is None:
        return
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    cmap_fn = None
    norm_fn = None
    if edge_color_values is not None:
        cmap_fn = _cm.get_cmap(edge_cmap)
        vmin = float(np.min(edge_color_values))
        vmax = float(np.max(edge_color_values))
        if vmax <= vmin:
            vmax = vmin + 1e-9
        norm_fn = _mc.Normalize(vmin=vmin, vmax=vmax)
    for k, (i, j) in enumerate(edges):
        if i < 0 or j < 0 or i >= X.shape[0] or j >= X.shape[0]:
            continue
        xs = [X[i, 0], X[j, 0]]
        ys = [X[i, 1], X[j, 1]]
        color = cmap_fn(norm_fn(edge_color_values[k])) if edge_color_values is not None else edge_color
        if edge_highlight:
            ax.plot(xs, ys, color="white", linewidth=edge_width * 4,
                    alpha=min(edge_alpha * 2.5, 0.85), zorder=0, solid_capstyle="round")
        ax.plot(xs, ys, color=color, linewidth=edge_width, alpha=edge_alpha, zorder=1)


def _compute_edge_color_values(edges, X, mode, node_color_values=None):
    if edges is None or mode == "single":
        return None
    radii = np.linalg.norm(X, axis=1)
    result = []
    for i, j in edges:
        if mode == "source-depth" and node_color_values is not None:
            val = node_color_values[i] if 0 <= i < len(node_color_values) else 0.0
        elif mode == "mean-radius":
            vi = radii[i] if 0 <= i < len(radii) else 0.0
            vj = radii[j] if 0 <= j < len(radii) else 0.0
            val = (vi + vj) * 0.5
        else:  # source-radius
            val = radii[i] if 0 <= i < len(radii) else 0.0
        result.append(val)
    return np.array(result, dtype=float)


def plot_points(X, objects=None, annotate=0, out=None, cmap="viridis", edges=None,
                edge_alpha=0.08, edge_width=0.3, color_values=None,
                label_mode="first", size_values=None, constant_color=None,
                edge_color="gray", edge_highlight=False,
                edge_color_by="single", edge_cmap="plasma",
                label_fontsize=8.0):
    colors = np.linalg.norm(X, axis=1) if color_values is None else color_values
    sizes = np.full(X.shape[0], 10.0, dtype=float) if size_values is None else size_values
    fig, ax = plt.subplots(figsize=(8, 8))
    ecv = _compute_edge_color_values(edges, X, edge_color_by, color_values)
    draw_edges(ax, X, edges, edge_alpha=edge_alpha, edge_width=edge_width, edge_color=edge_color,
               edge_highlight=edge_highlight, edge_color_values=ecv, edge_cmap=edge_cmap)
    scatter_kwargs = dict(s=sizes, alpha=0.85, edgecolors="none", zorder=2)
    if constant_color is not None:
        ax.scatter(X[:, 0], X[:, 1], color=constant_color, **scatter_kwargs)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, **scatter_kwargs)
    ax.set_title("Embeddings (2D projection)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if annotate and objects is not None:
        texts = []
        for i in choose_annotation_indices(len(objects), annotate, edges=edges, mode=label_mode):
            norm = max(float(np.hypot(X[i, 0], X[i, 1])), 1e-9)
            ox, oy = X[i, 0] / norm * 0.06, X[i, 1] / norm * 0.06
            t = ax.annotate(
                clean_label(objects[i]),
                xy=(X[i, 0], X[i, 1]),
                xytext=(X[i, 0] + ox, X[i, 1] + oy),
                fontsize=label_fontsize,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="k", lw=0.4),
                zorder=3,
            )
            texts.append(t)
        if _HAS_ADJUST_TEXT and texts:
            try:
                _adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))
            except Exception:
                pass

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=300)
        print(f"Saved figure to {out}")
    else:
        plt.show()


def plot_wn_style(X, objects=None, annotate=0, out=None, cmap="magma", edges=None,
                  edge_alpha=0.08, edge_width=0.3, color_values=None,
                  label_mode="first", size_values=None, constant_color=None,
                  edge_color="gray", edge_highlight=False,
                  edge_color_by="single", edge_cmap="plasma",
                  label_fontsize=7.0):
    theta = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#f8f8f8")
    ax.plot(np.cos(theta), np.sin(theta), color="#555555", linewidth=1.0, alpha=0.5)

    ecv = _compute_edge_color_values(edges, X, edge_color_by, color_values)
    draw_edges(ax, X, edges, edge_alpha=edge_alpha, edge_width=edge_width, edge_color=edge_color,
               edge_highlight=edge_highlight, edge_color_values=ecv, edge_cmap=edge_cmap)
    colors = np.linalg.norm(X, axis=1) if color_values is None else color_values
    sizes = np.full(X.shape[0], 35.0, dtype=float) if size_values is None else size_values
    scatter_kwargs = dict(s=sizes, alpha=1.0, edgecolors="k", linewidths=0.25, zorder=2)
    if constant_color is not None:
        ax.scatter(X[:, 0], X[:, 1], color=constant_color, **scatter_kwargs)
    else:
        ax.scatter(
            X[:, 0], X[:, 1], c=colors, cmap=cmap, **scatter_kwargs,
        )

    if annotate and objects is not None:
        texts = []
        for i in choose_annotation_indices(len(objects), annotate, edges=edges, mode=label_mode):
            norm = max(float(np.hypot(X[i, 0], X[i, 1])), 1e-9)
            ox, oy = X[i, 0] / norm * 0.05, X[i, 1] / norm * 0.05
            t = ax.text(
                X[i, 0] + ox, X[i, 1] + oy, clean_label(objects[i]),
                fontsize=label_fontsize, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec="k", lw=0.4),
                zorder=3,
            )
            texts.append(t)
        if _HAS_ADJUST_TEXT and texts:
            try:
                _adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))
            except Exception:
                pass

    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Saved figure to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings from a checkpoint")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--method", choices=["pca", "tsne", "poincare-mds"], default="pca")
    parser.add_argument("--style", choices=["simple", "wn"], default="simple")
    parser.add_argument("--fit-to-disk", action="store_true", default=False)
    parser.add_argument("--convert-lorentz", action="store_true", default=False)
    parser.add_argument("--radial", action="store_true", default=False)
    parser.add_argument("--edge-direction", choices=["parent-child", "child-parent"], default="parent-child")
    parser.add_argument("--annotate", type=int, default=0)
    parser.add_argument("--label-fontsize", type=float, default=8.0,
                        help="Font size for annotation labels")
    parser.add_argument("--label-mode", choices=["first", "top-degree"], default="first")
    parser.add_argument("--edges", type=str, default=None)
    parser.add_argument("--edge-format", choices=["auto", "names", "indices"], default="auto")
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--edge-alpha", type=float, default=0.08)
    parser.add_argument("--edge-width", type=float, default=0.3)
    parser.add_argument("--edge-sample-seed", type=int, default=42)
    parser.add_argument("--edge-color", default="#7f7f7f",
                        help="Color for plotted edges/links")
    parser.add_argument("--edge-highlight", action="store_true", default=False,
                        help="Draw edges with a white background stroke for higher contrast")
    parser.add_argument("--edge-color-by",
                        choices=["single", "source-radius", "source-depth", "mean-radius"],
                        default="single",
                        help="Color each edge individually: single=uniform, "
                             "source-radius=by source node distance from origin, "
                             "source-depth=by source node tree depth, "
                             "mean-radius=average radius of both endpoints")
    parser.add_argument("--edge-cmap", default="plasma",
                        help="Matplotlib colormap name used when --edge-color-by is not single")
    parser.add_argument("--color-by", choices=["radius", "depth", "fixed"], default="radius")
    parser.add_argument("--node-color", default="#4C78A8",
                        help="Matplotlib color name/hex when using --color-by fixed")
    parser.add_argument("--node-size-by", choices=["fixed", "depth", "degree", "children"], default="fixed")
    parser.add_argument("--depth-from", choices=["auto", "full", "reduced"], default="auto",
                        help="How to compute hierarchy depth from edges. "
                             "auto: use full edges unless depth collapses; "
                             "full: use all edges; reduced: remove transitive links first")
    parser.add_argument("--min-node-size", type=float, default=12.0)
    parser.add_argument("--max-node-size", type=float, default=90.0)
    parser.add_argument("--out", "-o", default=None)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iter", type=int, default=1000)
    args = parser.parse_args()

    emb, objects = load_embeddings(args.checkpoint)
    print(f"Loaded embeddings shape: {emb.shape}")

    if args.convert_lorentz:
        if emb.shape[1] < 2:
            raise RuntimeError("Embeddings too small to be Lorentz vectors")
        denom = emb[:, 0:1] + 1.0
        denom[denom == 0] = 1e-12
        emb = emb[:, 1:] / denom
        print("Converted Lorentz -> Poincare coordinates; new shape", emb.shape)

    proj_kwargs = {}
    if args.method == "tsne":
        proj_kwargs = {"perplexity": args.tsne_perplexity, "n_iter": args.tsne_iter}

    X = project(emb, method=args.method, **proj_kwargs)
    all_edges = None
    edges = None
    depth_values = None
    hierarchy_edges = None

    if args.edges:
        all_edges = load_edges(args.edges, objects=objects, fmt=args.edge_format)
        if args.edge_direction == "child-parent":
            all_edges = [(b, a) for (a, b) in all_edges]
        # Keep hierarchy/layout based on full graph, but only draw sampled edges
        # for speed/readability.
        edges = sample_edges(all_edges, max_edges=args.max_edges, seed=args.edge_sample_seed)

        hierarchy_edges = all_edges
        if args.depth_from == "reduced":
            hierarchy_edges = reduce_transitive_closure_edges(all_edges)
        elif args.depth_from == "auto":
            roots_tmp, children_tmp = compute_roots_and_children(all_edges, n_nodes=emb.shape[0])
            depth_tmp = compute_depths(roots_tmp, children_tmp)
            finite_tmp = [d for d in depth_tmp.values() if d is not None]
            maxd_tmp = max(finite_tmp) if finite_tmp else 0
            if maxd_tmp <= 1:
                hierarchy_edges = reduce_transitive_closure_edges(all_edges)

    if args.radial:
        if hierarchy_edges is None:
            raise RuntimeError("Radial layout requires --edges")
        roots, children = compute_roots_and_children(hierarchy_edges, n_nodes=emb.shape[0])
        depth = compute_depths(roots, children)
        depth_values = np.array([depth.get(i, 0) for i in range(emb.shape[0])], dtype=float)
        X_radial, nodes_order = assign_radial_positions(roots, children, depth)
        X_full = np.zeros((emb.shape[0], 2), dtype=float)
        for i, nid in enumerate(nodes_order):
            if 0 <= nid < emb.shape[0]:
                X_full[nid] = X_radial[i]
        X = X_full
    elif args.color_by == "depth" and hierarchy_edges is not None:
        roots, children = compute_roots_and_children(hierarchy_edges, n_nodes=emb.shape[0])
        depth = compute_depths(roots, children)
        depth_values = np.array([depth.get(i, 0) for i in range(emb.shape[0])], dtype=float)

    if depth_values is None and args.node_size_by == "depth" and hierarchy_edges is not None:
        roots, children = compute_roots_and_children(hierarchy_edges, n_nodes=emb.shape[0])
        depth = compute_depths(roots, children)
        depth_values = np.array([depth.get(i, 0) for i in range(emb.shape[0])], dtype=float)

    if args.fit_to_disk:
        norms = np.linalg.norm(X, axis=1)
        max_norm = norms.max() if norms.size > 0 else 0.0
        if max_norm > 0.95:
            X = X * (0.95 / max_norm)

    size_values = compute_node_sizes(
        emb.shape[0],
        mode=args.node_size_by,
        depth_values=depth_values,
        edges=all_edges,
        min_size=args.min_node_size,
        max_size=args.max_node_size,
    )
    color_constant = None
    color_values = None
    if args.color_by == "depth":
        color_values = depth_values
    elif args.color_by == "fixed":
        color_constant = args.node_color

    if args.style == "wn":
        plot_wn_style(
            X,
            objects=objects,
            annotate=args.annotate,
            out=args.out,
            edges=edges,
            edge_alpha=args.edge_alpha,
            edge_width=args.edge_width,
            color_values=color_values,
            label_mode=args.label_mode,
            size_values=size_values,
            constant_color=color_constant,
            edge_color=args.edge_color,
            edge_highlight=args.edge_highlight,
            edge_color_by=args.edge_color_by,
            edge_cmap=args.edge_cmap,
            label_fontsize=args.label_fontsize,
        )
    else:
        plot_points(
            X,
            objects=objects,
            annotate=args.annotate,
            out=args.out,
            edges=edges,
            edge_alpha=args.edge_alpha,
            edge_width=args.edge_width,
            color_values=color_values,
            label_mode=args.label_mode,
            size_values=size_values,
            constant_color=color_constant,
            edge_color=args.edge_color,
            edge_highlight=args.edge_highlight,
            edge_color_by=args.edge_color_by,
            edge_cmap=args.edge_cmap,
            label_fontsize=args.label_fontsize,
        )


if __name__ == "__main__":
    main()
