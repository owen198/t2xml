#!/usr/bin/env python3
"""Stage 2 of the SANTA-style pipeline: build Masked Entity Prediction (MEP)
examples from the (description, XML snippet) pairs emitted by preprocess.py.

Mirrors SANTA's own entity scripts (OpenMatch/SANTA, processing/Code/build_code_entity.py
and processing/Product/build_product_entity.py): identify "entity" tokens, replace
each distinct entity with a T5 sentinel token (same string -> same sentinel), and
emit a span-corruption-style label sequence for the model to reconstruct.

The key adaptation for XML: SANTA never masks a code function or a product title in
its entirety -- it masks identifier tokens inside the code (variable/function/class
names) or proper nouns inside the product text, leaving keywords/operators/punctuation
(code) or common words (product) untouched as context. The XML analog of an
"identifier" is an attribute value that names/codes something (modelIdentCode,
systemCode, issueNumber, ...) or a proper-noun-ish token inside prose text content --
never the element tag itself. Masking whole tags would remove the structure that
gives the model something to condition on; masking every identifier-like value still
leaves every tag name, attribute name, and piece of markup as context, exactly as
masking every code identifier still leaves every keyword and operator in place.
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

# T5's sentinel vocabulary is <extra_id_0> .. <extra_id_99>. One sentinel is
# reserved as the trailing terminator of the label sequence (matching SANTA's
# own convention of appending one extra, content-less sentinel at the end).
SENTINEL_BUDGET = 100
MAX_ENTITIES = SENTINEL_BUDGET - 1
DEFAULT_MASK_RATIO = 0.5

IDENTIFIER_ATTR_RE = re.compile(r"(?i)(code|ident|number|classification|isocode)")
EXTRA_IDENTIFIER_ATTRS = {"inWork", "year", "month", "day"}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
# The hyphen/slash alternative requires a digit somewhere in the token (the
# lookahead) so it matches real codes like "ICN-C0419-S1000D0382-001-01" but
# not an ordinary hyphenated English word like "cross-reference".
CODE_LIKE_RE = re.compile(r"^(?:[A-Z]{2,}[0-9]*|(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+[-/][A-Za-z0-9/-]+)$")

_NLTK_READY = False
try:
    import nltk
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    nltk.data.find("tokenizers/punkt_tab")
    _NLTK_READY = True
except Exception:
    nltk = None
    _NLTK_READY = False


def sentinel(i: int) -> str:
    return f"<extra_id_{i}>"


def placeholder(i: int) -> str:
    # ET.tostring() XML-escapes literal "<"/">" inside attribute values and
    # text (e.g. "<extra_id_0>" -> "&lt;extra_id_0&gt;"), which would stop the
    # T5 tokenizer from recognizing it as the real sentinel token. So sentinels
    # are stood in for with this escaping-safe placeholder during tree
    # mutation, and swapped to the real "<extra_id_N>" string after
    # serialization (see build_mep_example).
    return f"EXTRA_ID_{i}"


def is_identifier_attr(name: str) -> bool:
    return bool(IDENTIFIER_ATTR_RE.search(name)) or name in EXTRA_IDENTIFIER_ATTRS


def text_entity_tokens(text: str) -> list[str]:
    """Entity-like tokens inside a text node: a capitalization/code-pattern
    heuristic, supplemented by NLTK POS-tagged proper nouns when available.

    Most S1000D text nodes are short title/label fragments (<techName>,
    <infoName>, ...) that use Title Case as a *style convention* -- every
    major word capitalized, regardless of its grammatical role. Checked
    directly: NLTK correctly tags "Management" and "Error" as plain NN (not
    NNP) in fragments like "Service bulletin - Management information",
    because grammatically they *are* common nouns -- Title Case isn't a
    grammatical signal NLTK knows about. So POS tagging alone silently drops
    exactly the label words this heuristic most needs to catch, which is why
    the capitalization/position rule (mirroring SANTA's product-text rule in
    spirit, adapted to this domain's title-fragment style) is the primary
    signal, not NLTK. NLTK is layered on only as a supplement: any NNP/NNPS
    token it finds beyond position 0 that the primary rule missed (e.g. a
    genuine multi-word proper noun in flowing prose) is added, but nothing
    the primary rule already found is ever removed by it."""
    text = text.strip()
    if not text:
        return []

    words = WORD_RE.findall(text)
    tokens = []
    for i, w in enumerate(words):
        if len(w) < 2:
            continue
        if CODE_LIKE_RE.match(w) or (w[0].isupper() and i > 0):
            tokens.append(w)

    if _NLTK_READY:
        try:
            tagged = nltk.pos_tag(nltk.word_tokenize(text))
            seen_lower = {t.lower() for t in tokens}
            for i, (tok, pos) in enumerate(tagged):
                if i == 0 or len(tok) < 2 or tok.lower() in seen_lower:
                    continue
                if pos in ("NNP", "NNPS"):
                    tokens.append(tok)
                    seen_lower.add(tok.lower())
        except Exception:
            pass

    return tokens


def collect_candidates(root: ET.Element):
    """Walk the tree in document order, collecting entity candidates without
    mutating anything yet. Returns:
      attr_hits: list of (elem, attr_name, value)
      text_hits: list of (elem, "text"|"tail", full_string, [entity tokens in it])
    """
    attr_hits = []
    text_hits = []

    def visit_text(elem, which, value):
        if value and value.strip():
            toks = text_entity_tokens(value)
            if toks:
                text_hits.append((elem, which, value, toks))

    def walk(elem: ET.Element):
        for name, value in elem.attrib.items():
            if is_identifier_attr(name) and value.strip():
                attr_hits.append((elem, name, value))
        visit_text(elem, "text", elem.text)
        for child in elem:
            walk(child)
            visit_text(child, "tail", child.tail)

    walk(root)
    return attr_hits, text_hits


def build_mep_example(xml_snippet: str, mask_ratio: float, rng: random.Random):
    try:
        root = ET.fromstring(xml_snippet)
    except ET.ParseError:
        return None, "parse_error"

    attr_hits, text_hits = collect_candidates(root)

    # First-seen order across the whole document, deduped by exact string --
    # same identifier value anywhere in the snippet gets the same sentinel,
    # exactly like SANTA replaces every occurrence of the same code identifier
    # (or the same product entity word) with one shared special token.
    order = []
    seen = set()
    for _, _, value in attr_hits:
        if value not in seen:
            seen.add(value)
            order.append(value)
    for _, _, _, toks in text_hits:
        for t in toks:
            if t not in seen:
                seen.add(t)
                order.append(t)

    if not order:
        return None, "no_entities"

    # Ratio + hard cap, mirroring SANTA's own downsampling of identifier
    # occurrences (they keep ~50%, or 10% for identifier-dense JavaScript) --
    # without this, an attribute-heavy metadata element could have every
    # single value masked out, leaving nothing for the model to condition on.
    keep_n = max(1, min(MAX_ENTITIES, round(len(order) * mask_ratio)))
    if keep_n < len(order):
        kept = set(rng.sample(order, keep_n))
        order = [e for e in order if e in kept]

    # Mutate the tree with escaping-safe placeholders, not the real
    # "<extra_id_N>" sentinel strings directly -- ET.tostring() would
    # XML-escape their "<"/">" into "&lt;"/"&gt;", which the T5 tokenizer
    # would then no longer recognize as the actual sentinel token.
    entity_to_placeholder = {ent: placeholder(i) for i, ent in enumerate(order)}

    masked_root = copy.deepcopy(root)
    m_attr_hits, m_text_hits = collect_candidates(masked_root)

    for elem, name, value in m_attr_hits:
        if value in entity_to_placeholder:
            elem.set(name, entity_to_placeholder[value])

    def mask_string(s: str, toks: list[str]) -> str:
        for t in sorted(set(toks), key=len, reverse=True):
            if t not in entity_to_placeholder:
                continue
            s = re.sub(rf"(?<!\w){re.escape(t)}(?!\w)", entity_to_placeholder[t], s)
        return s

    for elem, which, value, toks in m_text_hits:
        new_value = mask_string(value, toks)
        if which == "text":
            elem.text = new_value
        else:
            elem.tail = new_value

    masked_root.tail = None
    masked_structured = ET.tostring(masked_root, encoding="unicode")
    # (?!\d): "EXTRA_ID_1" is a literal substring of "EXTRA_ID_10", so a plain
    # .replace() done in ascending index order would corrupt the latter before
    # its own turn came up. The negative lookahead makes each substitution
    # match only its exact, complete placeholder.
    for i in range(len(order)):
        masked_structured = re.sub(rf"{placeholder(i)}(?!\d)", sentinel(i), masked_structured)

    label_parts = []
    for i, ent in enumerate(order):
        label_parts.append(sentinel(i))
        label_parts.append(ent)
    label_parts.append(sentinel(len(order)))
    label = " ".join(label_parts)

    return {
        "masked_structured": masked_structured,
        "label": label,
        "num_entities": len(order),
        "num_attr_entities": sum(1 for v in order if v in {v2 for _, _, v2 in attr_hits}),
    }, "ok"


def process_split(in_path: Path, out_path: Path, mask_ratio: float, seed: int, stats: Counter):
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            stats["records"] += 1
            result, status = build_mep_example(rec["structured"], mask_ratio, rng)
            stats[status] += 1
            if result is None:
                continue
            out_rec = {
                "structured": rec["structured"],
                "masked_structured": result["masked_structured"],
                "label": result["label"],
                "num_entities": result["num_entities"],
                "source_file": rec.get("source_file"),
                "element": rec.get("element"),
                "xpath": rec.get("xpath"),
            }
            fout.write(json.dumps(out_rec) + "\n")
            stats["entities_total"] += result["num_entities"]
            stats["attr_entities_total"] += result["num_attr_entities"]
            stats["emitted"] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Build Masked Entity Prediction (MEP) examples from SDA pairs."
    )
    parser.add_argument("--input-dir", type=str, default=str(PRETRAIN_DIR),
                        help="Directory containing sda_pairs.{train,dev,test}.jsonl.")
    parser.add_argument("--output-dir", type=str, default=str(PRETRAIN_DIR),
                        help="Directory to write mep_pairs.{train,dev,test}.jsonl into.")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--mask-ratio", type=float, default=DEFAULT_MASK_RATIO,
                        help="Fraction of unique candidate entities to actually mask, capped at 99 regardless.")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    if not _NLTK_READY:
        print("warning: NLTK POS tagger/tokenizer not available; using regex "
              "capitalization/code-pattern heuristic for text entities", file=sys.stderr)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    overall = Counter()

    for split in args.splits:
        in_path = in_dir / f"sda_pairs.{split}.jsonl"
        if not in_path.exists():
            print(f"warning: missing {in_path}, skipping split {split}", file=sys.stderr)
            continue
        out_path = out_dir / f"mep_pairs.{split}.jsonl"
        stats = Counter()
        process_split(in_path, out_path, args.mask_ratio, args.random_seed, stats)
        overall.update(stats)

        print(f"--- {split} ---")
        print(f"records read:          {stats['records']}")
        print(f"skipped (parse error): {stats['parse_error']}")
        print(f"skipped (no entities): {stats['no_entities']}")
        print(f"emitted:               {stats['emitted']} -> {out_path}")
        if stats["emitted"]:
            print(f"avg entities/example:  {stats['entities_total'] / stats['emitted']:.2f}")

    print("=== overall ===")
    print(f"records read:          {overall['records']}")
    print(f"emitted:               {overall['emitted']}")
    if overall["emitted"]:
        print(f"avg entities/example:  {overall['entities_total'] / overall['emitted']:.2f}")
        print(f"attr-entity share:     {overall['attr_entities_total'] / overall['entities_total']:.1%}")


if __name__ == "__main__":
    main()
