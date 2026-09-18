#!/usr/bin/env python3
"""Stage 1 of the SANTA-style SDA pipeline: build (description, whole XML file) positive pairs from S1000D data modules."""
import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CACHE_DIR = REPO_ROOT / ".preprocess_cache"
WHITELIST_CACHE = CACHE_DIR / "s1000d_element_whitelist.json"
LLM_CACHE = CACHE_DIR / "llm_description_cache.json"

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

DEFAULT_INPUT_DIRS = ["BIKE_7", "FOSSIG", "S1000D spec sample", "s1kd-tools-doc"]
XSD_DIR = SCRIPT_DIR / "xsd"

S1000D_SCHEMA_FILES = [
    "appliccrossreftable.xsd", "brdoc.xsd", "brex.xsd", "checklist.xsd",
    "comment.xsd", "comrep.xsd", "condcrossreftable.xsd", "container.xsd",
    "crew.xsd", "dc.xsd", "ddn.xsd", "descript.xsd", "dml.xsd", "fault.xsd",
    "frontmatter.xsd", "icnmetadata.xsd", "ipd.xsd", "learning.xsd", "pm.xsd",
    "prdcrossreftable.xsd", "proced.xsd", "process.xsd", "rdf.xsd", "sb.xsd",
    "schedul.xsd", "scocontent.xsd", "scormcontentpackage.xsd", "update.xsd",
    "wrngdata.xsd", "wrngflds.xsd", "xcf.xsd", "xlink.xsd",
]

ELEMENT_NAME_RE = re.compile(r'<xs:element\s+[^>]*\bname="([^"]+)"')

NAMESPACES = {
    "dc": "http://www.purl.org/dc/elements/1.1/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xlink": "http://www.w3.org/1999/xlink",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}
for _prefix, _uri in NAMESPACES.items():
    ET.register_namespace(_prefix, _uri)

MIN_DESC_WORDS = 3
LLM_MAX_CALLS_DEFAULT = 600
LLM_PROMPT_MAX_CHARS = 16000
LLM_MODEL_DEFAULT = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "whole-file-v1"


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def git_show(relpath: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:datasets/{relpath}"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def build_element_whitelist(use_cache: bool = True) -> set[str]:
    if use_cache and WHITELIST_CACHE.exists():
        return set(json.loads(WHITELIST_CACHE.read_text()))

    names: set[str] = set()
    missing = []
    for fname in S1000D_SCHEMA_FILES:
        local_path = XSD_DIR / fname
        if local_path.exists():
            content = local_path.read_text()
        else:
            content = git_show(fname)
        if content is None:
            missing.append(fname)
            continue
        names.update(ELEMENT_NAME_RE.findall(content))

    if missing:
        print(f"warning: could not recover {len(missing)} schema file(s): {missing}", file=sys.stderr)
    if not names:
        raise RuntimeError(
            f"failed to build element whitelist: no S1000D schema files found under {XSD_DIR} "
            "or in git history (commit 0095fc1)"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    WHITELIST_CACHE.write_text(json.dumps(sorted(names), indent=2))
    return names


def folder_slug(name: str) -> str:
    return name.replace(" ", "_")


def collect_input_files(input_dirs: list[str]) -> dict[str, list[Path]]:
    """folder name (as passed in --input-dirs) -> sorted list of its *.xml files."""
    files_by_folder: dict[str, list[Path]] = {}
    for d in input_dirs:
        base = SCRIPT_DIR / d
        if not base.is_dir():
            print(f"warning: input dir not found, skipping: {base}", file=sys.stderr)
            continue
        found = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() == ".xml"]
        files_by_folder[d] = sorted(found)
    return files_by_folder


def humanize_tag(tag: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", tag).lower()


def _llm_prompt(xml_text: str, root_tag: str) -> str:
    truncated = len(xml_text) > LLM_PROMPT_MAX_CHARS
    snippet_for_prompt = xml_text[:LLM_PROMPT_MAX_CHARS]
    note = "\n\n[Note: document truncated; only the beginning is shown.]" if truncated else ""
    return (
        "You are labeling training data for a dense retrieval model that must learn to "
        "match natural-language descriptions to whole structured S1000D XML documents.\n"
        "Write ONE concise paragraph (max ~60 words) in plain English describing what the "
        "following XML document is and what it documents or specifies overall -- its "
        "subject, purpose, and principal content. Describe the meaning/content, not the "
        "XML syntax itself.\n\n"
        f"<document root=\"{root_tag}\">\n{snippet_for_prompt}{note}\n</document>"
    )


class LLMDescriber:
    def __init__(self, model: str, max_calls: int, cache_path: Path):
        self.model = model
        self.max_calls = max_calls
        self.cache_path = cache_path
        self.calls_made = 0
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text())
        self._lock = threading.Lock()
        self._client = None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                print("warning: anthropic package not installed; LLM tier will use extractive fallback", file=sys.stderr)
        else:
            print("warning: ANTHROPIC_API_KEY not set; LLM tier will use extractive fallback", file=sys.stderr)

    def _cache_key(self, xml_text: str) -> str:
        return hashlib.sha256((PROMPT_VERSION + xml_text).encode("utf-8")).hexdigest()

    def describe(self, xml_text: str, root_tag: str) -> tuple[str, str]:
        key = self._cache_key(xml_text)
        if key in self.cache:
            return self.cache[key], "llm_cached"

        if self._client is not None and self.calls_made < self.max_calls:
            try:
                text = self._call_api(xml_text, root_tag)
                with self._lock:
                    self.calls_made += 1
                    self.cache[key] = text
                return text, "llm"
            except Exception as exc:
                print(f"warning: LLM call failed ({exc}); falling back to extractive summary", file=sys.stderr)

        return self._extractive_fallback(xml_text, root_tag), "extractive_fallback"

    def _call_api(self, xml_text: str, root_tag: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=180,
            messages=[{"role": "user", "content": _llm_prompt(xml_text, root_tag)}],
        )
        return resp.content[0].text.strip()

    def _extractive_fallback(self, xml_text: str, root_tag: str) -> str:
        try:
            texts = [t.strip() for t in ET.fromstring(xml_text).itertext() if t and t.strip()]
        except ET.ParseError:
            texts = []
        return (" ".join(texts))[:200] or f"{humanize_tag(root_tag)} document."

    def describe_batch(self, text_tag_pairs: list[tuple[str, str]], max_workers: int = 8,
                        save_every: int = 200) -> None:
        """Concurrently warm self.cache for every (xml_text, root_tag) not already
        cached, up to the remaining call budget. Safe to call describe() again
        afterward for each pending item -- it will now resolve as a cache hit (or,
        for anything that didn't get a live call in time, the same deterministic
        fallback)."""
        if self._client is None:
            return

        seen = set()
        to_fetch = []
        for text, tag in text_tag_pairs:
            key = self._cache_key(text)
            if key not in self.cache and key not in seen:
                seen.add(key)
                to_fetch.append((key, text, tag))

        budget = max(0, self.max_calls - self.calls_made)
        to_fetch = to_fetch[:budget]
        if not to_fetch:
            return

        print(f"warming LLM description cache: {len(to_fetch)} live calls, "
              f"{max_workers} concurrent workers", file=sys.stderr)
        completed = 0

        def _fetch_one(item):
            nonlocal completed
            key, text, tag = item
            try:
                desc = self._call_api(text, tag)
            except Exception as exc:
                print(f"warning: LLM call failed ({exc}); will fall back to extractive summary", file=sys.stderr)
                return
            with self._lock:
                self.calls_made += 1
                self.cache[key] = desc
                completed += 1
                if completed % save_every == 0:
                    self.save_cache()
                    print(f"  ...{completed}/{len(to_fetch)} done, cache checkpointed", file=sys.stderr)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_fetch_one, to_fetch))

        self.save_cache()
        print(f"cache warm-up complete: {completed}/{len(to_fetch)} live calls succeeded", file=sys.stderr)

    def save_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2))


def serialize(elem: ET.Element) -> str:
    # elem.tail is text that follows elem in its *parent's* content; it isn't
    # part of elem itself, so it must be dropped or tostring() appends it
    # after the closing tag, producing invalid/misleading output.
    clone = copy.deepcopy(elem)
    clone.tail = None
    return ET.tostring(clone, encoding="unicode")


def content_hash(xml_text: str) -> str:
    normalized = re.sub(r"\s+", " ", xml_text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_for_file(filename: str) -> str:
    # 90/3/7, copying SANTA's own Code Search finetune split ratio
    # (CodeSearchNet/Adv: 251,820/9,604/19,210 train/dev/test ~= 89.7/3.4/6.9%)
    # rather than an arbitrary round number -- see datasets/README.md.
    h = int(hashlib.md5(filename.encode("utf-8")).hexdigest(), 16) % 100
    if h < 90:
        return "train"
    if h < 93:
        return "dev"
    return "test"


def build_sda_record(path: Path, whitelist: set[str], stats: Counter) -> dict | None:
    """Parse one whole file and return a pending SDA record (text=None), or
    None on a parse error."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        print(f"warning: skipping unparsable file {path}: {exc}", file=sys.stderr)
        stats["parse_error"] += 1
        return None

    root_tag = strip_ns(root.tag)
    if root_tag not in whitelist:
        print(f"warning: root element <{root_tag}> in {path.name} not in S1000D whitelist; emitting anyway", file=sys.stderr)
        stats["unwhitelisted_root"] += 1

    return {
        "structured": serialize(root),
        "text": None,
        "source_file": path.name,
        "dataset": None,
        "element": root_tag,
        "tier": "llm",
        "generated_by": None,
    }


def write_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build (description, whole XML file) SDA positive pairs from S1000D data modules."
    )
    parser.add_argument("--input-dirs", nargs="+", default=DEFAULT_INPUT_DIRS,
                         help="Directories under datasets/ to scan for .XML files.")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "pretrain"),
                         help="Directory under which per-folder sda_pairs.{train,dev,test}.jsonl are written.")
    parser.add_argument("--max-llm-calls", type=int, default=LLM_MAX_CALLS_DEFAULT,
                         help="Safety cap on number of live LLM API calls this run.")
    parser.add_argument("--llm-model", type=str, default=LLM_MODEL_DEFAULT)
    parser.add_argument("--llm-concurrency", type=int, default=8,
                         help="Number of concurrent LLM API calls during the cache warm-up pass.")
    parser.add_argument("--no-cache-whitelist", action="store_true",
                         help="Force rebuilding the element whitelist from git history.")
    args = parser.parse_args()

    whitelist = build_element_whitelist(use_cache=not args.no_cache_whitelist)
    print(f"loaded whitelist: {len(whitelist)} S1000D element names")

    files_by_folder = collect_input_files(args.input_dirs)
    total_files = sum(len(v) for v in files_by_folder.values())
    print(f"found {total_files} XML files across {len(files_by_folder)} folder(s): {args.input_dirs}")

    describer = LLMDescriber(args.llm_model, args.max_llm_calls, LLM_CACHE)
    stats = Counter()

    per_folder_records: dict[str, list] = {}
    all_pending = []
    for folder, files in files_by_folder.items():
        slug = folder_slug(folder)
        records = []
        for path in files:
            stats["files"] += 1
            rec = build_sda_record(path, whitelist, stats)
            if rec is None:
                continue
            rec["dataset"] = slug
            records.append(rec)

        # Whole-file exact-duplicate detection is diagnostic only: at this
        # granularity dropping a record discards an entire document from an
        # already-small corpus, so duplicates are logged, never dropped.
        dup_hashes = Counter(content_hash(r["structured"]) for r in records)
        for r in records:
            if dup_hashes[content_hash(r["structured"])] > 1:
                stats["exact_dup_whole_file"] += 1
                print(f"note: {r['source_file']} ({folder}) is byte-identical to another file in this folder", file=sys.stderr)

        per_folder_records[folder] = records
        all_pending.extend(records)

    # All files parsed: warm the description cache concurrently across every
    # folder's pending records (shared LLM call budget), then resolve each
    # pending record from it (a fast cache hit after warm-up, or the same
    # deterministic extractive fallback otherwise).
    unique_pending = {}
    for r in all_pending:
        unique_pending.setdefault(r["structured"], r["element"])
    describer.describe_batch(list(unique_pending.items()), max_workers=args.llm_concurrency)

    for r in all_pending:
        text, generated_by = describer.describe(r["structured"], r["element"])
        if len(text.split()) < MIN_DESC_WORDS:
            stats["dropped_short_desc"] += 1
            r["_drop"] = True
        else:
            r["text"] = text
            r["generated_by"] = generated_by
            stats["emitted_llm"] += 1

    describer.save_cache()

    out_dir = Path(args.output_dir)
    for folder, records in per_folder_records.items():
        slug = folder_slug(folder)
        by_split = {"train": [], "dev": [], "test": []}
        for r in records:
            if r.get("_drop"):
                continue
            by_split[split_for_file(r["source_file"])].append(r)
        for split, recs in by_split.items():
            write_jsonl(recs, out_dir / slug / f"sda_pairs.{split}.jsonl")
        print(f"[{folder}] train: {len(by_split['train'])}  dev: {len(by_split['dev'])}  test: {len(by_split['test'])}")

    print("---")
    print(f"files processed:            {stats['files']}")
    print(f"parse errors:                {stats['parse_error']}")
    print(f"unwhitelisted root (kept):   {stats['unwhitelisted_root']}")
    print(f"whole-file exact duplicates: {stats['exact_dup_whole_file']}")
    print(f"dropped (short desc):       {stats['dropped_short_desc']}")
    print(f"emitted (llm):              {stats['emitted_llm']}")
    print(f"llm API calls made:         {describer.calls_made}")


if __name__ == "__main__":
    main()
