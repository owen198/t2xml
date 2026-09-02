# SANTA training on t2xml

Adapted from [OpenMatch/SANTA](https://arxiv.org/abs/2305.19912) for t2xml's S1000D
data. `pretrain.sh` runs the joint SDA contrastive + MEP generative pretraining
stage on `data/pretrain.*.jsonl`; `finetune.sh` finetunes on the retrieval
benchmark (`data/finetune.*.jsonl`). `evaluate_xml/` + `shell/index-xml.sh` +
`shell/evaluate_xml.sh` run retrieval eval (see their own docstrings/comments).

Open experiment variables for pretrain/finetune hyperparameters are tracked in
[`datasets/README.md`](../datasets/README.md#stage-3--training-hyperparameters-pretrainfinetune-santa), alongside the dataset-design ones, rather than here.
