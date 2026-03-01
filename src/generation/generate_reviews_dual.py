import json
import logging
from pathlib import Path

from src.config import (
    PAIRS_JSONL_PATH,
    REVIEWS_PEER_JSONL,
    REVIEWS_VANILLA_JSONL,
    GEN_MODEL_NAME,
    GEN_TEMPERATURE,
    GEN_MAX_OUTPUT_TOKENS,
    GEN_PAPER_MAX_CHARS,
    SKILL_PATH,
    PROJECT_ROOT,
    ensure_dirs,
)
from src.generation.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROMPT_PATH_PEER = PROJECT_ROOT / "prompts" / "peer_review_generation.txt"
PROMPT_PATH_VANILLA = PROJECT_ROOT / "prompts" / "vanilla_review.txt"

# If model tries to escape by claiming text is missing, mark as invalid
FORBIDDEN_PHRASES = [
    "only the title",
    "only provided the title",
    "text was not provided",
    "full text was not provided",
    "since you have only provided the title",
    "provided text does not include",
    "cannot review the specific",
    "i cannot review",
    "insufficient information",
    "not provided beyond the title",
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_peer_review_skill_text() -> str:
    """Load skill instructions from peer-review-skills/SKILL.md."""
    with open(SKILL_PATH, "r", encoding="utf-8") as f:
        return f.read()


def looks_invalid(review_text: str) -> bool:
    low = (review_text or "").lower()
    return any(p in low for p in FORBIDDEN_PHRASES)


def smart_truncate(text: str, max_chars: int) -> tuple[str, str]:
    """
    Keep both early (title/abstract/method) and late (experiments/conclusion) parts.
    60% head + 40% tail.
    Returns: (truncated_text, strategy_name)
    """
    text = text or ""
    if len(text) <= max_chars:
        return text, "no_truncation"

    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len

    head = text[:head_len].rstrip()
    tail = text[-tail_len:].lstrip()

    merged = head + "\n\n[...TRUNCATED...]\n\n" + tail
    return merged, "head_tail"


def load_existing_ids(path: Path) -> set:
    """Load paper IDs that already have successful results (for resume)."""
    ids = set()
    if path.exists():
        for row in read_jsonl(path):
            if row.get("error") is None:
                ids.add(row["paper_id"])
    return ids


def main():
    """
    Generate reviews for both conditions (full dataset) with resume support:
      - peer: peer-review skill (SKILL.md) injected prompt
      - vanilla: fair baseline prompt (same 7-section format, no skill)
    Input: data/processed/pairs.jsonl
    Outputs:
      - outputs/generations/reviews_peer.jsonl
      - outputs/generations/reviews_vanilla.jsonl
    """
    ensure_dirs()

    if not PAIRS_JSONL_PATH.exists():
        raise FileNotFoundError(
            f"Pairs file not found: {PAIRS_JSONL_PATH}. "
            f"Run scripts/01_build_pairs.py first."
        )

    # Load all pairs
    all_pairs = list(read_jsonl(PAIRS_JSONL_PATH))
    total = len(all_pairs)
    logger.info(f"Loaded {total} pairs from {PAIRS_JSONL_PATH}")

    # Resume: skip papers already generated successfully
    done_peer = load_existing_ids(REVIEWS_PEER_JSONL)
    done_vanilla = load_existing_ids(REVIEWS_VANILLA_JSONL)
    done_both = done_peer & done_vanilla

    if done_both:
        logger.info(
            f"Resuming: {len(done_both)} papers already complete, "
            f"skipping those."
        )

    client = LLMClient(model_name=GEN_MODEL_NAME)
    template_peer = load_prompt(PROMPT_PATH_PEER)
    template_vanilla = load_prompt(PROMPT_PATH_VANILLA)
    peer_skill_text = load_peer_review_skill_text()

    # Open in append mode for resume support
    mode_p = "a" if done_peer else "w"
    mode_v = "a" if done_vanilla else "w"

    with open(REVIEWS_PEER_JSONL, mode_p, encoding="utf-8") as out_p, \
         open(REVIEWS_VANILLA_JSONL, mode_v, encoding="utf-8") as out_v:

        for idx, row in enumerate(all_pairs, 1):
            paper_id = row["paper_id"]

            if paper_id in done_both:
                continue

            paper_text = row["paper_text"]
            paper_text_trunc, trunc_strategy = smart_truncate(
                paper_text, GEN_PAPER_MAX_CHARS
            )

            # --- Peer / skill condition ---
            if paper_id not in done_peer:
                try:
                    prompt_peer = (
                        template_peer
                        .replace("{peer_review_skill}", peer_skill_text)
                        .replace("{paper_text}", paper_text_trunc)
                    )
                    review_peer = client.generate(
                        prompt_peer,
                        temperature=GEN_TEMPERATURE,
                        max_output_tokens=GEN_MAX_OUTPUT_TOKENS,
                    )

                    if looks_invalid(review_peer):
                        raise RuntimeError(
                            "INVALID: model claimed text was missing."
                        )

                    peer_result = {
                        "paper_id": paper_id,
                        "condition": "peer",
                        "generated_review": review_peer,
                        "error": None,
                        "paper_text_chars": len(paper_text_trunc),
                        "truncation": trunc_strategy,
                        "model": GEN_MODEL_NAME,
                    }
                except Exception as e:
                    peer_result = {
                        "paper_id": paper_id,
                        "condition": "peer",
                        "generated_review": None,
                        "error": str(e),
                        "paper_text_chars": len(paper_text_trunc),
                        "truncation": trunc_strategy,
                        "model": GEN_MODEL_NAME,
                    }

                out_p.write(json.dumps(peer_result, ensure_ascii=False) + "\n")
                out_p.flush()

            # --- Vanilla baseline condition ---
            if paper_id not in done_vanilla:
                try:
                    prompt_vanilla = template_vanilla.replace(
                        "{paper_text}", paper_text_trunc
                    )
                    review_vanilla = client.generate(
                        prompt_vanilla,
                        temperature=GEN_TEMPERATURE,
                        max_output_tokens=GEN_MAX_OUTPUT_TOKENS,
                    )

                    if looks_invalid(review_vanilla):
                        raise RuntimeError(
                            "INVALID: model claimed text was missing."
                        )

                    vanilla_result = {
                        "paper_id": paper_id,
                        "condition": "vanilla",
                        "generated_review": review_vanilla,
                        "error": None,
                        "paper_text_chars": len(paper_text_trunc),
                        "truncation": trunc_strategy,
                        "model": GEN_MODEL_NAME,
                    }
                except Exception as e:
                    vanilla_result = {
                        "paper_id": paper_id,
                        "condition": "vanilla",
                        "generated_review": None,
                        "error": str(e),
                        "paper_text_chars": len(paper_text_trunc),
                        "truncation": trunc_strategy,
                        "model": GEN_MODEL_NAME,
                    }

                out_v.write(
                    json.dumps(vanilla_result, ensure_ascii=False) + "\n"
                )
                out_v.flush()

            logger.info(
                f"[{idx}/{total}] paper_id={paper_id} "
                f"trunc={trunc_strategy}"
            )

    logger.info(f"Done. Peer → {REVIEWS_PEER_JSONL}")
    logger.info(f"Done. Vanilla → {REVIEWS_VANILLA_JSONL}")


if __name__ == "__main__":
    main()
