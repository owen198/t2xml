# Figures

## How to read one panel

- Each point is one document's embedding vector, projected from 768 dimensions down to 2 with t-SNE.
- **Blue circle = passage** (a document). **Orange triangle = query.**
- A **gray line** connects a query to the passage it's supposed to retrieve. Short, mostly
  non-crossing lines mean the model tends to place a query near its own document; long tangled lines mean it doesn't.
- **`doc-doc cos` in the title** is the mean pairwise cosine similarity between all passage vectors in that panel —
  computed from the embeddings; **1.000 means every document maps to nearly the same
  vector**, i.e. total collapse. Lower is more spread out / less collapsed.

## Figures

| File | Panels | 
| --- | --- |
| `02_pooling_fix.png` | baseline (first-token pooling, dot product) → mean pooling + dot (`fixB`) → mean pooling + cosine (`fixA`) → no pretraining + `fixA` config, all on `BIKE_7_to_S1000D_spec_sample` |
| `fixA_diff_sources.png` | `fixA_meancos_p1024_s42`'s 3 pretrain sources (BIKE_7, FOSSIG, s1kd-tools-doc), all finetuned on `S1000D_spec_sample` | 
| `orig_vs_anon.png` | `fixA_meancos_p1024_s42` vs. `anonq_meancos_p1024_s42`, both `BIKE_7_to_S1000D_spec_sample` |  

## To visualize models

```bash
cd santa
.venv/bin/python diagnostics/viz_embeddings.py \
  --model "label 1" ../results/<tag>/<pair>/embeddings.npz \
  --model "label 2" ../results/<tag2>/<pair2>/embeddings.npz \
  --method tsne --links --out ../results/figures/my_comparison.png
```

The `embeddings.npz` files are gitignored (local only), so this needs a machine that has run the experiment. Any `results/<tag>/<pair>/embeddings.npz` can be passed to `--model` (label + path) `--links` draws the query→passage lines; drop it for a cleaner plot with many
panels. 
