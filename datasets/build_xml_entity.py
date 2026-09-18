#!/usr/bin/env python3
"""Stage 2 of the SANTA-style pipeline: build Masked Tag Prediction (MEP/MTP)
examples from the (description, XML document) pairs emitted by preprocess.py.

Mirrors SANTA's own entity-masking scripts in spirit (replace a chosen subset
of "identifier" occurrences with T5 sentinel tokens, emit a span-corruption
label sequence), but the identifier here is the element *tag name*, not an
attribute/text value: for each whole-file structured snippet, a random subset
of element tag occurrences (default 50%) is replaced by a shared sentinel,
and the model must reconstruct the original tag name for each occurrence.

Unlike value-masking (where the same identifier *string* recurring in a
document is masked to one shared sentinel, since repetition signals it's the
same entity), tag masking never dedups by tag name: every selected element
*occurrence* gets its own sentinel/label slot, because repeated tag names
(e.g. five sibling <step> elements) are structurally normal, not an identity
signal. The document root is never masked -- it has no parent/sibling context
to recover it from.
"""
import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PRETRAIN_DIR = REPO_ROOT / "pretrain"

sys.path.insert(0, str(SCRIPT_DIR))
from preprocess import build_element_whitelist, strip_ns  # noqa: E402

# T5's sentinel vocabulary is <extra_id_0> .. <extra_id_99>. One sentinel is
# reserved as the trailing terminator of the label sequence (matching SANTA's
# own convention of appending one extra, content-less sentinel at the end).
SENTINEL_BUDGET = 100
MAX_ENTITIES = SENTINEL_BUDGET - 1
DEFAULT_MASK_RATIO = 0.5


def sentinel(i: int) -> str:
    return f"<extra_id_{i}>"


def placeholder(i: int) -> str:
    # ET.tostring() XML-escapes literal "<"/">" inside attribute values and
    # text, and can't render an arbitrary string as a tag name safely either
    # -- so a masked element's tag is temporarily set to this escaping/
    # XML-safe placeholder during tree mutation, and swapped to the real
    # "<extra_id_N>" string in the serialized text afterward (see
    # build_mtp_example).
    return f"EXTRAIDPLACEHOLDER{i}TAG"


def collect_taggable_elements(root: ET.Element, whitelist: set[str] | None) -> list[ET.Element]:
    """Document-order list of non-root elements eligible for tag masking. The
    root is always excluded. When whitelist is given, only S1000D-whitelisted
    tags are eligible (keeps the objective scoped to genuine S1000D structural
    vocabulary rather than incidental namespace plumbing like xlink/rdf)."""
    out = []

    def walk(elem: ET.Element):
        for child in elem:
            if whitelist is None or strip_ns(child.tag) in whitelist:
                out.append(child)
            walk(child)

    walk(root)
    return out


def build_mtp_example(xml_snippet: str, mask_ratio: float, rng: random.Random,
                       whitelist: set[str] | None):
    try:
        root = ET.fromstring(xml_snippet)
    except ET.ParseError:
        return None, "parse_error"

    candidates = collect_taggable_elements(root, whitelist)
    if not candidates:
        return None, "no_candidates"

    # Ratio + hard cap, mirroring SANTA's own downsampling of identifier
    # occurrences -- without the cap, a huge document could exceed the T5
    # sentinel budget.
    n = len(candidates)
    keep_n = max(1, min(MAX_ENTITIES, round(n * mask_ratio)))
    if keep_n < n:
        chosen_idx = sorted(rng.sample(range(n), keep_n))
    else:
        chosen_idx = list(range(n))

    # Mutate a deep copy so the original `structured` text is never touched.
    masked_root = copy.deepcopy(root)
    masked_candidates = collect_taggable_elements(masked_root, whitelist)

    labels = []
    for out_i, idx in enumerate(chosen_idx):
        orig_tag = strip_ns(candidates[idx].tag)
        # Setting elem.tag on the copy masks both its open and close tag "for
        # free" via ElementTree's own serialization -- no separate open/close
        # bookkeeping needed.
        masked_candidates[idx].tag = placeholder(out_i)
        labels.append(orig_tag)

    masked_root.tail = None
    masked_structured = ET.tostring(masked_root, encoding="unicode")
    # (?!\d): "EXTRAIDPLACEHOLDER1TAG" is a literal substring of
    # "EXTRAIDPLACEHOLDER10TAG", so substituting in ascending index order
    # without this guard would corrupt the latter before its own turn came
    # up. The negative lookahead makes each substitution match only its
    # exact, complete placeholder.
    for i in range(len(labels)):
        masked_structured = re.sub(rf"{placeholder(i)}(?!\d)", f"extra_id_{i}", masked_structured)

    label_parts = []
    for i, tag in enumerate(labels):
        label_parts.append(sentinel(i))
        label_parts.append(tag)
    label_parts.append(sentinel(len(labels)))
    label = " ".join(label_parts)

    return {
        "masked_structured": masked_structured,
        "label": label,
        "num_masked_tags": len(labels),
    }, "ok"


def process_split(in_path: Path, out_path: Path, mask_ratio: float, seed: int,
                   whitelist: set[str] | None, stats: Counter):
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            stats["records"] += 1
            result, status = build_mtp_example(rec["structured"], mask_ratio, rng, whitelist)
            stats[status] += 1
            if result is None:
                continue
            out_rec = {
                "structured": rec["structured"],
                "masked_structured": result["masked_structured"],
                "label": result["label"],
                "num_masked_tags": result["num_masked_tags"],
                "source_file": rec.get("source_file"),
                "dataset": rec.get("dataset"),
                "element": rec.get("element"),
            }
            fout.write(json.dumps(out_rec) + "\n")
            stats["masked_tags_total"] += result["num_masked_tags"]
            stats["emitted"] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Build Masked Tag Prediction (MEP) examples from SDA pairs, per dataset folder."
    )
    parser.add_argument("--pretrain-root", type=str, default=str(PRETRAIN_DIR),
                        help="Directory containing one subdir per dataset folder, each with sda_pairs.{train,dev,test}.jsonl.")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Directory to mirror the per-folder mep_pairs.{train,dev,test}.jsonl into (default: same as --pretrain-root).")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Folder slugs under --pretrain-root to process (default: every subdir containing sda_pairs.*.jsonl).")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--mask-ratio", type=float, default=DEFAULT_MASK_RATIO,
                        help="Fraction of tag occurrences to mask, capped at 99 regardless.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--tag-whitelist", action=argparse.BooleanOptionalAction, default=True,
                        help="Restrict maskable tags to the S1000D schema element whitelist (default: on).")
    args = parser.parse_args()

    whitelist = build_element_whitelist() if args.tag_whitelist else None

    pretrain_root = Path(args.pretrain_root)
    output_root = Path(args.output_root) if args.output_root else pretrain_root

    slugs = args.datasets or sorted(
        p.name for p in pretrain_root.iterdir()
        if p.is_dir() and any(p.glob("sda_pairs.*.jsonl"))
    )
    if not slugs:
        print(f"warning: no dataset folders with sda_pairs.*.jsonl found under {pretrain_root}", file=sys.stderr)

    overall = Counter()
    for slug in slugs:
        in_dir = pretrain_root / slug
        out_dir = output_root / slug
        print(f"=== {slug} ===")
        for split in args.splits:
            in_path = in_dir / f"sda_pairs.{split}.jsonl"
            if not in_path.exists():
                print(f"warning: missing {in_path}, skipping split {split}", file=sys.stderr)
                continue
            out_path = out_dir / f"mep_pairs.{split}.jsonl"
            stats = Counter()
            process_split(in_path, out_path, args.mask_ratio, args.random_seed, whitelist, stats)
            overall.update(stats)

            print(f"--- {split} ---")
            print(f"records read:          {stats['records']}")
            print(f"skipped (parse error): {stats['parse_error']}")
            print(f"skipped (no candidates): {stats['no_candidates']}")
            print(f"emitted:               {stats['emitted']} -> {out_path}")
            if stats["emitted"]:
                print(f"avg masked tags/example: {stats['masked_tags_total'] / stats['emitted']:.2f}")

    print("=== overall ===")
    print(f"records read:          {overall['records']}")
    print(f"emitted:               {overall['emitted']}")
    if overall["emitted"]:
        print(f"avg masked tags/example: {overall['masked_tags_total'] / overall['emitted']:.2f}")


if __name__ == "__main__":
    main()
