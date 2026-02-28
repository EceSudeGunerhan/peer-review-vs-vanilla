# src/data_prep/build_pairs.py
# Build (paper + ground_truth) pairs from raw ICLR 2017 dataset (robust + clean)

from pathlib import Path
from typing import List, Dict, Any
import random
import re

from src.config import (
    PARSED_PDFS_DIR,
    REVIEWS_DIR,
    PAIRS_JSONL_PATH,
    DEFAULT_PAPER_MAX_CHARS,
    RANDOM_SEED,
    ensure_dirs,
)
from src.utils import read_json, write_jsonl


# --- Quality gates (reduce noisy/generic reviews) ---
MIN_PAPER_CHARS = 1500
MIN_GT_CHARS = 200


def _is_junk_file(p: Path) -> bool:
    return p.name.startswith("._") or p.name.startswith(".")


def _paper_id_from_filename(filename: str) -> str:
    # "304.pdf.json" -> "304"
    return filename.split(".")[0]


def _collect_strings(obj: Any, max_items: int = 2000) -> List[str]:
    """Recursively collect string leaves as a fallback."""
    out: List[str] = []

    def walk(x: Any):
        nonlocal out
        if len(out) >= max_items:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
                if len(out) >= max_items:
                    return
        elif isinstance(x, list):
            for v in x:
                walk(v)
                if len(out) >= max_items:
                    return

    walk(obj)
    return out


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    # collapse huge whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_paper_text(paper_json: Dict[str, Any]) -> str:
    """
    Correct extraction for your parsed_pdfs schema:
      - title: metadata.title
      - abstract: metadata.abstractText
      - sections: metadata.sections[] with {heading, text}
    """
    parts: List[str] = []

    metadata = paper_json.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    # Title
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(f"TITLE: {title.strip()}")

    # Abstract
    abstract = metadata.get("abstractText") or metadata.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        parts.append(f"ABSTRACT: {abstract.strip()}")

    # Sections
    sections = metadata.get("sections")
    if isinstance(sections, list):
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            heading = sec.get("heading")
            text = sec.get("text")

            if isinstance(text, str) and text.strip():
                if isinstance(heading, str) and heading.strip():
                    parts.append(f"{heading.strip()}\n{text.strip()}")
                else:
                    parts.append(text.strip())

    # Fallback
    if not parts:
        parts = _collect_strings(paper_json, max_items=300)

    full_text = _clean_text("\n\n".join(parts))

    # hard cap
    full_text = full_text[:DEFAULT_PAPER_MAX_CHARS]

    # If we only captured a very short stub (e.g. just the title),
    # aggressively fall back to collecting all string leaves from the JSON.
    # This protects against schema drift where sections/abstract are stored
    # under unexpected keys and would otherwise produce title-only "papers".
    if len(full_text) < MIN_PAPER_CHARS:
        fallback_parts = _collect_strings(paper_json, max_items=300)
        fallback_text = _clean_text("\n\n".join(fallback_parts))[:DEFAULT_PAPER_MAX_CHARS]

        # Only overwrite if we actually obtained something longer / more informative
        if len(fallback_text) > len(full_text):
            full_text = fallback_text

    return full_text


def _normalize_review_root(review_obj: Any) -> Any:
    """
    Your review files can start as a LIST like: [ { ... } ].
    Normalize to a dict when possible.
    """
    if isinstance(review_obj, list) and review_obj:
        if isinstance(review_obj[0], dict):
            return review_obj[0]
    return review_obj


def extract_ground_truth(review_obj: Any) -> str:
    """
    Clean extraction for common PeerRead/ICLR review structure.
    After normalization, we try:
      - root['reviews'][i]['review']
    Fallback: string collection.
    """
    review_json = _normalize_review_root(review_obj)

    texts: List[str] = []

    reviews = None
    if isinstance(review_json, dict):
        reviews = review_json.get("reviews")

    if isinstance(reviews, list):
        for r in reviews:
            if isinstance(r, dict):
                txt = r.get("review")
                if isinstance(txt, str) and txt.strip():
                    texts.append(txt.strip())

    if not texts:
        all_strings = _collect_strings(review_obj, max_items=300)
        texts = all_strings[:200]

    gt = _clean_text("\n\n---\n\n".join(texts))
    return gt


def build_pairs() -> List[Dict[str, str]]:
    random.seed(RANDOM_SEED)
    pairs: List[Dict[str, str]] = []

    pdf_files = sorted(PARSED_PDFS_DIR.glob("*.json"))

    missing_review = 0
    empty_paper = 0
    empty_gt = 0
    too_short_paper = 0
    too_short_gt = 0

    for pdf_path in pdf_files:
        if _is_junk_file(pdf_path):
            continue

        paper_id = _paper_id_from_filename(pdf_path.name)
        review_path = REVIEWS_DIR / f"{paper_id}.json"

        if not review_path.exists():
            missing_review += 1
            continue

        paper_json = read_json(pdf_path)
        review_obj = read_json(review_path)

        paper_text = extract_paper_text(paper_json)
        ground_truth = extract_ground_truth(review_obj)

        if not paper_text.strip():
            empty_paper += 1
            continue
        if not ground_truth.strip():
            empty_gt += 1
            continue

        # --- Quality gates ---
        if len(paper_text) < MIN_PAPER_CHARS:
            too_short_paper += 1
            continue
        if len(ground_truth) < MIN_GT_CHARS:
            too_short_gt += 1
            continue

        pairs.append(
            {
                "paper_id": paper_id,
                "paper_text": paper_text,
                "ground_truth": ground_truth,
            }
        )

    print(f"[Debug] PDF files found: {len(pdf_files)}")
    print(f"[Debug] Missing review file: {missing_review}")
    print(f"[Debug] Empty paper_text after extraction: {empty_paper}")
    print(f"[Debug] Empty ground_truth after extraction: {empty_gt}")
    print(f"[Debug] Too-short paper_text (<{MIN_PAPER_CHARS}): {too_short_paper}")
    print(f"[Debug] Too-short ground_truth (<{MIN_GT_CHARS}): {too_short_gt}")

    return pairs


def main() -> None:
    ensure_dirs()

    pairs = build_pairs()
    print(f"Total matched pairs: {len(pairs)}")

    write_jsonl(PAIRS_JSONL_PATH, pairs)
    print(f"Saved to: {PAIRS_JSONL_PATH}")


if __name__ == "__main__":
    main()