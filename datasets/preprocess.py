#!/usr/bin/env python3
"""Stage 1 of the SANTA-style SDA pipeline: build (description, XML snippet) positive pairs from S1000D data modules."""
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

DEFAULT_INPUT_DIRS = ["BIKE", "FOSSIG", "S1000D spec sample", "s1kd-tools-doc"]
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

PROSE_TEXT_THRESHOLD = 40  # chars of stripped text content -> routes to LLM tier
DEDUP_CAP_DEFAULT = 20
SIBLING_CAP_DEFAULT = 20
MIN_DESC_WORDS = 3
LLM_MAX_CALLS_DEFAULT = 500
LLM_PROMPT_MAX_CHARS = 4000
LLM_MODEL_DEFAULT = "claude-haiku-4-5-20251001"


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


def collect_input_files(input_dirs: list[str]) -> list[Path]:
    files = []
    for d in input_dirs:
        base = SCRIPT_DIR / d
        if not base.is_dir():
            print(f"warning: input dir not found, skipping: {base}", file=sys.stderr)
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".xml":
                files.append(p)
    return sorted(files)


def attr_str(elem: ET.Element, limit: int = 8) -> str:
    items = list(elem.attrib.items())[:limit]
    return ", ".join(f"{k}={v!r}" for k, v in items)


def humanize_tag(tag: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", tag).lower()


def _t_dmCode(e: ET.Element) -> str:
    a = e.attrib
    # Every identity-bearing attribute must appear here -- omitting any of
    # them (as the original version did for systemDiffCode/assyCode/
    # disassyCode/disassyCodeVariant) lets two structurally different dmCode
    # elements render to byte-identical text, which is fatal for a downstream
    # consumer using this text as a unique retrieval query.
    sys_parts = [a.get("systemDiffCode"), a.get("systemCode"), a.get("subSystemCode"), a.get("subSubSystemCode")]
    system = "-".join(p for p in sys_parts if p)
    assy_parts = [a.get("assyCode"), a.get("disassyCode")]
    assy = "-".join(p for p in assy_parts if p)
    return (
        f"Data module code for model {a.get('modelIdentCode', '?')}, system {system}, "
        f"assembly {assy}{a.get('disassyCodeVariant', '')}, "
        f"info code {a.get('infoCode', '?')}{a.get('infoCodeVariant', '')}, "
        f"item location {a.get('itemLocationCode', '?')}."
    )


def _t_language(e: ET.Element) -> str:
    a = e.attrib
    return f"Language: {a.get('languageIsoCode', '?')} ({a.get('countryIsoCode', '?')})."


def _t_issueInfo(e: ET.Element) -> str:
    a = e.attrib
    return f"Issue number {a.get('issueNumber', '?')}, in-work {a.get('inWork', '?')}."


def _t_dmIdent(e: ET.Element) -> str:
    # A pure wrapper: all its identity-bearing content lives in its children
    # (dmCode/language/issueInfo), not its own attributes/text. Composing
    # their descriptions -- rather than falling through to the generic
    # tag+attrs fallback, which sees an empty-attribute element with no text
    # and renders the same "dm ident element." for every instance -- makes
    # the result a genuine function of the full underlying content again.
    code = e.find("dmCode")
    lang = e.find("language")
    issue = e.find("issueInfo")
    parts = [_t_dmCode(code) if code is not None else "unspecified data module code."]
    if lang is not None:
        parts.append(_t_language(lang))
    if issue is not None:
        parts.append(_t_issueInfo(issue))
    return "Data module identity. " + " ".join(parts)


def _t_dmRefIdent(e: ET.Element) -> str:
    # Same wrapper problem as dmIdent, but dmRefIdent only ever wraps a
    # single dmCode identifying some *other*, referenced data module.
    code = e.find("dmCode")
    return "Reference to data module. " + (_t_dmCode(code) if code is not None else "Unspecified target.")


def _t_issueDate(e: ET.Element) -> str:
    a = e.attrib
    return f"Issue date {a.get('year', '?')}-{a.get('month', '?')}-{a.get('day', '?')}."


def _t_security(e: ET.Element) -> str:
    return f"Security classification code {e.attrib.get('securityClassification', '?')}."


def _t_dmTitle(e: ET.Element) -> str:
    tech = (e.findtext("techName") or "").strip()
    info = (e.findtext("infoName") or "").strip()
    title = " — ".join(p for p in (tech, info) if p)
    return f"Data module title: '{title}'."


def _t_dmAddress(e: ET.Element) -> str:
    code = e.find(".//dmCode")
    title = e.find(".//dmTitle")
    date = e.find(".//issueDate")
    parts = []
    if code is not None:
        a = code.attrib
        parts.append(f"model {a.get('modelIdentCode', '?')} system {a.get('systemCode', '?')}")
    if title is not None:
        tech = (title.findtext("techName") or "").strip()
        info = (title.findtext("infoName") or "").strip()
        t = " — ".join(p for p in (tech, info) if p)
        if t:
            parts.append(f"titled '{t}'")
    if date is not None:
        a = date.attrib
        parts.append(f"issued {a.get('year', '?')}-{a.get('month', '?')}-{a.get('day', '?')}")
    return "Address block identifying data module " + ", ".join(parts) + "."


def _t_identAndStatusSection(e: ET.Element) -> str:
    status = e.find("dmStatus")
    title = e.find(".//dmTitle")
    t = ""
    if title is not None:
        tech = (title.findtext("techName") or "").strip()
        info = (title.findtext("infoName") or "").strip()
        t = " — ".join(p for p in (tech, info) if p)
    issue_type = status.attrib.get("issueType", "?") if status is not None else "?"
    return f"Identification and status section for '{t}', status: {issue_type}."


def _t_dmStatus(e: ET.Element) -> str:
    rpc = e.find(".//responsiblePartnerCompany/enterpriseName")
    return (
        f"Data module status block (issue type: {e.attrib.get('issueType', '?')}), "
        f"responsible party: {(rpc.text or '?').strip() if rpc is not None else '?'}."
    )


def _t_responsiblePartnerCompany(e: ET.Element) -> str:
    name = e.findtext("enterpriseName") or "?"
    return f"Responsible partner company: {name.strip()}."


def _t_originator(e: ET.Element) -> str:
    name = e.findtext("enterpriseName") or "?"
    return f"Originator: {name.strip()}."


def _t_qualityAssurance(e: ET.Element) -> str:
    child = next(iter(e), None)
    status = strip_ns(child.tag) if child is not None else "unspecified"
    return f"Quality assurance status: {status}."


def _t_applic(e: ET.Element) -> str:
    text = " ".join(t.strip() for t in e.itertext() if t and t.strip())
    return f"Applicability: {text or 'unspecified'}."


def _t_reasonForUpdate(e: ET.Element) -> str:
    text = " ".join(t.strip() for t in e.itertext() if t and t.strip())
    return f"Reason for update: {text or 'unspecified'}."


def _t_brexDmRef(e: ET.Element) -> str:
    code = e.find(".//dmCode")
    ref = f"model {code.attrib.get('modelIdentCode', '?')} info code {code.attrib.get('infoCode', '?')}" if code is not None else "unspecified"
    return f"Business rules exchange (BREX) reference: {ref}."


TAG_TEMPLATES = {
    "dmCode": _t_dmCode,
    "dmIdent": _t_dmIdent,
    "dmRefIdent": _t_dmRefIdent,
    "language": _t_language,
    "issueInfo": _t_issueInfo,
    "issueDate": _t_issueDate,
    "security": _t_security,
    "dmTitle": _t_dmTitle,
    "dmAddress": _t_dmAddress,
    "identAndStatusSection": _t_identAndStatusSection,
    "dmStatus": _t_dmStatus,
    "responsiblePartnerCompany": _t_responsiblePartnerCompany,
    "originator": _t_originator,
    "qualityAssurance": _t_qualityAssurance,
    "applic": _t_applic,
    "reasonForUpdate": _t_reasonForUpdate,
    "brexDmRef": _t_brexDmRef,
}


def generic_template(elem: ET.Element, tag: str) -> str:
    parts = [f"{humanize_tag(tag)} element"]
    attrs = attr_str(elem)
    if attrs:
        parts.append(f"with {attrs}")
    texts = [t.strip() for t in elem.itertext() if t and t.strip()]
    if texts:
        joined = " ".join(texts)[:200]
        parts.append(f"(text: {joined})")
    return " ".join(parts) + "."


def template_description(elem: ET.Element, tag: str) -> str:
    fn = TAG_TEMPLATES.get(tag)
    if fn is not None:
        try:
            return fn(elem)
        except Exception:
            pass
    return generic_template(elem, tag)


def prose_text_length(elem: ET.Element) -> int:
    return sum(len((t or "").strip()) for t in elem.itertext())


def is_trivial(elem: ET.Element) -> bool:
    # A bare marker element (no attributes, no text anywhere in it, no
    # children) carries ~no information on its own, e.g. <unverified/> inside
    # <qualityAssurance> -- the CSN analog of a stub getter/setter.
    return not elem.attrib and not len(elem) and prose_text_length(elem) == 0


def shape_signature(elem: ET.Element) -> str:
    # Structural fingerprint: tag + attribute *names* (not values) + child
    # shapes, recursively. Two elements get the same signature iff they have
    # the same tag/attribute-name/child-tag structure regardless of the
    # actual values inside -- e.g. every <dmRef> in a reference list has this
    # in common, even though each points at a different module.
    tag = strip_ns(elem.tag)
    attr_names = ",".join(sorted(elem.attrib.keys()))
    children = "".join(shape_signature(c) for c in elem)
    return f"{tag}({attr_names})[{children}]"


def shape_hash(elem: ET.Element) -> str:
    return hashlib.sha256(shape_signature(elem).encode("utf-8")).hexdigest()


def _llm_prompt(xml_snippet: str, tag: str) -> str:
    snippet_for_prompt = xml_snippet[:LLM_PROMPT_MAX_CHARS]
    return (
        "You are labeling training data for a dense retrieval model that must learn to "
        "match natural-language descriptions to structured S1000D XML snippets.\n"
        "Write ONE concise sentence (max ~40 words) in plain English describing what the "
        "following XML snippet documents or specifies. Describe the meaning/content, not "
        "the XML syntax itself.\n\n"
        f"<snippet tag=\"{tag}\">\n{snippet_for_prompt}\n</snippet>"
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

    def describe(self, xml_snippet: str, tag: str) -> tuple[str, str]:
        key = hashlib.sha256(xml_snippet.encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key], "llm_cached"

        if self._client is not None and self.calls_made < self.max_calls:
            try:
                text = self._call_api(xml_snippet, tag)
                with self._lock:
                    self.calls_made += 1
                    self.cache[key] = text
                return text, "llm"
            except Exception as exc:
                print(f"warning: LLM call failed ({exc}); falling back to extractive summary", file=sys.stderr)

        return self._extractive_fallback(xml_snippet, tag), "extractive_fallback"

    def _call_api(self, xml_snippet: str, tag: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=120,
            messages=[{"role": "user", "content": _llm_prompt(xml_snippet, tag)}],
        )
        return resp.content[0].text.strip()

    def _extractive_fallback(self, xml_snippet: str, tag: str) -> str:
        texts = [t.strip() for t in ET.fromstring(xml_snippet).itertext() if t and t.strip()]
        return (" ".join(texts))[:200] or f"{humanize_tag(tag)} content."

    def describe_batch(self, snippet_tag_pairs: list[tuple[str, str]], max_workers: int = 8,
                        save_every: int = 200) -> None:
        """Concurrently warm self.cache for every (snippet, tag) not already cached,
        up to the remaining call budget. Safe to call describe() again afterward for
        each pending item -- it will now resolve as a cache hit (or, for anything
        that didn't get a live call in time, the same deterministic fallback)."""
        if self._client is None:
            return

        seen = set()
        to_fetch = []
        for snippet, tag in snippet_tag_pairs:
            key = hashlib.sha256(snippet.encode("utf-8")).hexdigest()
            if key not in self.cache and key not in seen:
                seen.add(key)
                to_fetch.append((key, snippet, tag))

        budget = max(0, self.max_calls - self.calls_made)
        to_fetch = to_fetch[:budget]
        if not to_fetch:
            return

        print(f"warming LLM description cache: {len(to_fetch)} live calls, "
              f"{max_workers} concurrent workers", file=sys.stderr)
        completed = 0

        def _fetch_one(item):
            nonlocal completed
            key, snippet, tag = item
            try:
                text = self._call_api(snippet, tag)
            except Exception as exc:
                print(f"warning: LLM call failed ({exc}); will fall back to extractive summary", file=sys.stderr)
                return
            with self._lock:
                self.calls_made += 1
                self.cache[key] = text
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
    # elem.tail is text that follows elem in its *parent's* content (e.g. the
    # rest of a paragraph after an inline <emphasis>); it isn't part of elem
    # itself, so it must be dropped or tostring() appends it after the closing
    # tag, producing invalid/misleading output for inline elements.
    clone = copy.deepcopy(elem)
    clone.tail = None
    return ET.tostring(clone, encoding="unicode")


def content_hash(xml_snippet: str) -> str:
    normalized = re.sub(r"\s+", " ", xml_snippet).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_for_file(filename: str) -> str:
    h = int(hashlib.md5(filename.encode("utf-8")).hexdigest(), 16) % 100
    if h < 90:
        return "train"
    if h < 95:
        return "dev"
    return "test"


def walk_file(
    root: ET.Element,
    source_file: str,
    whitelist: set[str],
    exact_dup_counts: Counter,
    sibling_shape_counts: Counter,
    dedup_cap: int,
    sibling_cap: int,
    describer: LLMDescriber,
    stats: Counter,
    records: list,
):
    def process(elem: ET.Element, xpath: str, parent_xpath: str) -> bool:
        """Returns True if elem was dropped as an excess near-duplicate
        sibling, in which case the caller must not recurse into its children
        either -- they're part of the same redundant list entry (e.g. the
        dmCode/dmRefIdent nested inside a capped, 501st <dmRef> in a
        reference list shouldn't be emitted just because each has its own
        uniquely-indexed xpath)."""
        tag = strip_ns(elem.tag)
        skip_children = False
        if tag in whitelist:
            stats["candidates"] += 1
            is_prose = prose_text_length(elem) >= PROSE_TEXT_THRESHOLD

            if not is_prose and is_trivial(elem):
                stats["dropped_trivial"] += 1
            else:
                snippet = serialize(elem)

                # Cross-file exact-duplicate cap: the same boilerplate block
                # (e.g. <security securityClassification="01"/>) recurring
                # byte-for-byte across many files.
                eh = content_hash(snippet)
                exact_dup_counts[eh] += 1
                dropped = False
                if exact_dup_counts[eh] > dedup_cap:
                    stats["dropped_dedup_exact"] += 1
                    dropped = True
                # Within-file near-duplicate cap: many same-shape siblings
                # under the same parent (e.g. a reference list with hundreds
                # of <dmRef> entries, each pointing at a different module but
                # structurally identical). Scoped by (file, parent, shape) so
                # it never touches e.g. one <dmCode> per file under its own
                # <dmAddress> -- those never share a parent context.
                elif not is_prose:
                    sh = (source_file, parent_xpath, shape_hash(elem))
                    sibling_shape_counts[sh] += 1
                    if sibling_shape_counts[sh] > sibling_cap:
                        stats["dropped_dedup_sibling"] += 1
                        dropped = True
                        skip_children = True

                if not dropped:
                    if is_prose:
                        # Deferred: resolving live LLM descriptions one at a time
                        # here, per element, would serialize every API call across
                        # the whole corpus. Instead this is appended as a pending
                        # placeholder (text=None) and resolved in a batch after all
                        # files are walked -- see main()'s cache warm-up step.
                        records.append({
                            "structured": snippet,
                            "text": None,
                            "source_file": source_file,
                            "element": tag,
                            "xpath": xpath,
                            "tier": "llm",
                            "generated_by": None,
                        })
                        stats["pending_llm"] += 1
                    else:
                        text = template_description(elem, tag)
                        if len(text.split()) < MIN_DESC_WORDS:
                            stats["dropped_short_desc"] += 1
                        else:
                            records.append({
                                "structured": snippet,
                                "text": text,
                                "source_file": source_file,
                                "element": tag,
                                "xpath": xpath,
                                "tier": "template",
                                "generated_by": "template",
                            })
                            stats["emitted_template"] += 1

        if skip_children:
            return True

        tag_totals = Counter(strip_ns(c.tag) for c in elem)
        tag_seen = Counter()
        for child in elem:
            ctag = strip_ns(child.tag)
            tag_seen[ctag] += 1
            seg = f"{ctag}[{tag_seen[ctag]}]" if tag_totals[ctag] > 1 else ctag
            process(child, f"{xpath}/{seg}", xpath)
        return False

    root_xpath = f"/{strip_ns(root.tag)}"
    process(root, root_xpath, "")


def write_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build (description, XML snippet) SDA positive pairs from S1000D data modules."
    )
    parser.add_argument("--input-dirs", nargs="+", default=DEFAULT_INPUT_DIRS,
                         help="Directories under datasets/ to scan for .XML files.")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "pretrain"),
                         help="Directory to write sda_pairs.{train,dev,test}.jsonl into.")
    parser.add_argument("--dedup-cap", type=int, default=DEDUP_CAP_DEFAULT,
                         help="Max number of byte-identical snippets to keep, across the whole corpus.")
    parser.add_argument("--sibling-cap", type=int, default=SIBLING_CAP_DEFAULT,
                         help="Max number of same-shape (attrs/text ignored) siblings to keep under one parent in one file, e.g. reference-list entries.")
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

    files = collect_input_files(args.input_dirs)
    print(f"found {len(files)} XML files under {args.input_dirs}")

    describer = LLMDescriber(args.llm_model, args.max_llm_calls, LLM_CACHE)
    exact_dup_counts = Counter()
    sibling_shape_counts = Counter()
    stats = Counter()
    records = []

    for path in files:
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            print(f"warning: skipping unparsable file {path}: {exc}", file=sys.stderr)
            continue
        stats["files"] += 1
        walk_file(
            tree.getroot(), path.name, whitelist,
            exact_dup_counts, sibling_shape_counts, args.dedup_cap, args.sibling_cap,
            describer, stats, records,
        )

    # All files walked: exact-dup/sibling-shape caps are final now, so the full
    # set of pending LLM-tier snippets is known. Warm the description cache
    # concurrently, then resolve each pending record from it (a fast cache hit
    # after warm-up, or the same deterministic extractive fallback otherwise).
    pending = [r for r in records if r["tier"] == "llm" and r["text"] is None]
    unique_pending = {}
    for r in pending:
        unique_pending.setdefault(r["structured"], r["element"])
    describer.describe_batch(list(unique_pending.items()), max_workers=args.llm_concurrency)

    for r in pending:
        text, generated_by = describer.describe(r["structured"], r["element"])
        if len(text.split()) < MIN_DESC_WORDS:
            stats["dropped_short_desc"] += 1
            r["_drop"] = True
        else:
            r["text"] = text
            r["generated_by"] = generated_by
            stats["emitted_llm"] += 1
    records = [r for r in records if not r.get("_drop")]

    describer.save_cache()

    by_split = {"train": [], "dev": [], "test": []}
    for r in records:
        by_split[split_for_file(r["source_file"])].append(r)

    out_dir = Path(args.output_dir)
    for split, recs in by_split.items():
        write_jsonl(recs, out_dir / f"sda_pairs.{split}.jsonl")

    print("---")
    print(f"files processed:            {stats['files']}")
    print(f"candidate snippets:         {stats['candidates']}")
    print(f"dropped (trivial):          {stats['dropped_trivial']}")
    print(f"dropped (exact dup cap):    {stats['dropped_dedup_exact']}")
    print(f"dropped (sibling shape cap):{stats['dropped_dedup_sibling']}")
    print(f"dropped (short desc):       {stats['dropped_short_desc']}")
    print(f"emitted (template):         {stats['emitted_template']}")
    print(f"emitted (llm):              {stats['emitted_llm']}")
    print(f"llm API calls made:         {describer.calls_made}")
    for split, recs in by_split.items():
        print(f"{split}: {len(recs)} pairs -> {out_dir / f'sda_pairs.{split}.jsonl'}")


if __name__ == "__main__":
    main()
