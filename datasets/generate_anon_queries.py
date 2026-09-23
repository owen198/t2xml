#!/usr/bin/env python3
"""Stage 1 of the "new queries" experiment (results/README.md): regenerate S1000D_spec_sample's SDA
descriptions with a prompt that forbids copying verbatim identifiers/dates/codes, so retrieval on them
can't be solved by string matching the way the original queries can (BM25 = 1.0 on them; see
santa/diagnostics/bm25_baseline.py and leakage_split.py).

Does not touch pretrain/, retrieval/, or the original LLM cache. Reads the already-built
`structured` field from pretrain/<folder>/sda_pairs.*.jsonl (so it reuses Stage 1's split assignment and
whitelist-checked whole-file documents) and writes new records with the same shape to
pretrain_anonq/<folder>/sda_pairs.*.jsonl -- a drop-in input for build_retrieval_dataset.py's
--pretrain-root.

Usage:
  python3 generate_anon_queries.py --pilot 5        # sanity-check the prompt on 5 docs, no full run
  python3 generate_anon_queries.py                  # full run over all splits of --folder
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CACHE_PATH = REPO_ROOT / ".preprocess_cache" / "llm_description_cache_anonq.json"
PROMPT_VERSION = "anonq-v1"
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_PROMPT_MAX_CHARS = 16000

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

IDENT = re.compile(r"\b(?:[A-Za-z]*\d{2,}[A-Za-z0-9\-]*|[A-Za-z0-9]+(?:-[A-Za-z0-9]+){2,})\b")


def idents(text):
    return {m.group(0).lower() for m in IDENT.finditer(text)}


def anon_prompt(xml_text: str, root_tag: str) -> str:
    truncated = len(xml_text) > LLM_PROMPT_MAX_CHARS
    snippet = xml_text[:LLM_PROMPT_MAX_CHARS]
    note = "\n\n[Note: document truncated; only the beginning is shown.]" if truncated else ""
    return (
        "You are labeling training data for a dense retrieval model that must learn to "
        "match natural-language descriptions to whole structured S1000D XML documents, "
        "WITHOUT being able to just string-match identifiers.\n"
        "Write ONE concise paragraph (max ~60 words) in plain English describing what the "
        "following XML document is and what it documents or specifies overall -- its "
        "subject, purpose, and principal content. Describe the meaning/content, not the "
        "XML syntax itself.\n\n"
        "Hard rule: do NOT copy any code, number, or date that appears verbatim in the "
        "document -- no dmCode/pmCode values, issue numbers, chapter/section numbers, part "
        "numbers, or dates. Refer to them descriptively instead of quoting them, e.g. "
        "\"the latest issue\" instead of an issue number, \"a mid-level chapter on crew "
        "procedures\" instead of a section number. If the subject matter itself requires a "
        "generic technical term that happens to contain digits (e.g. a real product name), "
        "that is fine -- the rule is about this document's own identifiers, not technical "
        "vocabulary in general.\n\n"
        f"<document root=\"{root_tag}\">\n{snippet}{note}\n</document>"
    )


class AnonDescriber:
    def __init__(self, model, cache_path):
        self.model = model
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self._lock = threading.Lock()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set (checked .env and environment)")
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def _key(self, xml_text):
        return hashlib.sha256((PROMPT_VERSION + xml_text).encode("utf-8")).hexdigest()

    def describe(self, xml_text, root_tag):
        key = self._key(xml_text)
        with self._lock:
            if key in self.cache:
                return self.cache[key], True
        resp = self._client.messages.create(
            model=self.model, max_tokens=180,
            messages=[{"role": "user", "content": anon_prompt(xml_text, root_tag)}],
        )
        text = resp.content[0].text.strip()
        with self._lock:
            self.cache[key] = text
        return text, False

    def save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=0))


_DF_CACHE = {}


def _corpus_df(corpus_dir):
    """Document frequency of identifier-like tokens across the folder's train+dev+test sda_pairs, so a
    leak check can ignore boilerplate shared by (almost) every document, e.g. "s1000d" itself."""
    key = str(corpus_dir)
    if key in _DF_CACHE:
        return _DF_CACHE[key]
    import collections
    df, n = collections.Counter(), 0
    for split in ("train", "dev", "test"):
        p = Path(corpus_dir) / f"sda_pairs.{split}.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            n += 1
            df.update(idents(json.loads(line)["structured"]))
    _DF_CACHE[key] = (df, n)
    return df, n


def leak_check(query, doc_structured, corpus_dir=None, rare_df_frac=0.5):
    """Identifier-like tokens in the query that also appear verbatim in the doc, excluding tokens so
    common across the corpus (>= rare_df_frac of documents) that matching them doesn't help retrieval."""
    qi = idents(query)
    di = idents(doc_structured)
    hit = qi & di
    if corpus_dir is not None and hit:
        df, n = _corpus_df(corpus_dir)
        hit = {t for t in hit if df.get(t, 0) / max(n, 1) < rare_df_frac}
    return sorted(hit)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default="S1000D_spec_sample")
    ap.add_argument("--pretrain-root", default=str(REPO_ROOT / "pretrain"))
    ap.add_argument("--output-root", default=str(REPO_ROOT / "pretrain_anonq"))
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--pilot", type=int, default=0, help="only process the first N docs (across all splits), print leak report, do not write output")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    describer = AnonDescriber(LLM_MODEL, CACHE_PATH)
    src_dir = Path(args.pretrain_root) / args.folder
    out_dir = Path(args.output_root) / args.folder

    records_by_split = {}
    for split in args.splits:
        path = src_dir / f"sda_pairs.{split}.jsonl"
        records_by_split[split] = [json.loads(l) for l in path.open()]

    if args.pilot:
        pool = [r for split in args.splits for r in records_by_split[split]][: args.pilot]
        print(f"pilot: {len(pool)} docs")

        def do_one(r):
            text, cached = describer.describe(r["structured"], r.get("element") or "root")
            leaks = leak_check(text, r["structured"], corpus_dir=src_dir)
            return r["source_file"], text, leaks, cached

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for source_file, text, leaks, cached in ex.map(do_one, pool):
                flag = "LEAK" if leaks else "ok  "
                print(f"[{flag}] {source_file}{' (cached)' if cached else ''}\n  {text}\n  leaked tokens: {leaks or '-'}\n")
        describer.save_cache()
        return

    total_leak = 0
    total_n = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        recs = records_by_split[split]
        results = [None] * len(recs)

        def do_one(i_r):
            i, r = i_r
            text, cached = describer.describe(r["structured"], r.get("element") or "root")
            leaks = leak_check(text, r["structured"], corpus_dir=src_dir)
            return i, text, leaks, cached

        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for i, text, leaks, cached in ex.map(do_one, list(enumerate(recs))):
                results[i] = (text, leaks)

        out_path = out_dir / f"sda_pairs.{split}.jsonl"
        with out_path.open("w") as f:
            for r, (text, leaks) in zip(recs, results):
                total_n += 1
                total_leak += bool(leaks)
                new_r = dict(r)
                new_r["text"] = text
                new_r["generated_by"] = "llm_anonq"
                new_r["_leaked_tokens"] = leaks
                f.write(json.dumps(new_r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(recs)} records -> {out_path}")
        describer.save_cache()

    print(f"\ndone. leak rate: {total_leak}/{total_n} queries still had >=1 verbatim identifier match ({100*total_leak/total_n:.1f}%)")


if __name__ == "__main__":
    main()
