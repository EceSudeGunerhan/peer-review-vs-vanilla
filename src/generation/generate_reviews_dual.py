import json
from pathlib import Path

from src.config import (
    SAMPLE_PAIRS_JSONL_PATH,
    GENERATIONS_DIR,
    GEN_MODEL_NAME,
    GEN_TEMPERATURE,
    GEN_MAX_OUTPUT_TOKENS,
    ensure_dirs,
)
from src.generation.llm_client import LLMClient


PROMPT_PATH_PEER = Path("prompts/peer_review_generation.txt")
PROMPT_PATH_VANILLA = Path("prompts/vanilla_review.txt")

# Hard limit to avoid context overflow / losing key info
PAPER_TEXT_MAX_CHARS = 8000

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
    # Source of truth for skill instructions (copied from SKILL.md)
    skill_path = Path("prompts/peer_review_skill.md")
    with open(skill_path, "r", encoding="utf-8") as f:
        return f.read()


def looks_invalid(review_text: str) -> bool:
    low = (review_text or "").lower()
    return any(p in low for p in FORBIDDEN_PHRASES)


def smart_truncate(text: str, max_chars: int) -> tuple[str, str]:
    """
    Keep both early (title/abstract/method) and late (experiments/conclusion) parts.
    Returns: (truncated_text, strategy_name)
    """
    text = text or ""
    if len(text) <= max_chars:
        return text, "no_truncation"

    # 60% head + 40% tail (works well for papers)
    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len

    head = text[:head_len].rstrip()
    tail = text[-tail_len:].lstrip()

    merged = head + "\n\n[...TRUNCATED...]\n\n" + tail
    return merged, "head_tail"


def main():
    """
    Generate reviews for both conditions:
      - peer: structured / skill-like prompt
      - vanilla: simple baseline reviewer prompt
    Outputs:
      - outputs/generations/reviews_sample_peer.jsonl
      - outputs/generations/reviews_sample_vanilla.jsonl
    """
    ensure_dirs()

    client = LLMClient(model_name=GEN_MODEL_NAME)
    template_peer = load_prompt(PROMPT_PATH_PEER)
    template_vanilla = load_prompt(PROMPT_PATH_VANILLA)
    peer_skill_text = load_peer_review_skill_text()

    out_peer = GENERATIONS_DIR / "reviews_sample_peer.jsonl"
    out_vanilla = GENERATIONS_DIR / "reviews_sample_vanilla.jsonl"

    with open(out_peer, "w", encoding="utf-8") as out_p, open(
        out_vanilla, "w", encoding="utf-8"
    ) as out_v:
        for row in read_jsonl(SAMPLE_PAIRS_JSONL_PATH):
            paper_id = row["paper_id"]
            paper_text = row["paper_text"]

            paper_text_trunc, trunc_strategy = smart_truncate(
                paper_text, PAPER_TEXT_MAX_CHARS
            )

            # --- Peer / skill-like condition ---
            peer_result: dict
            try:
                prompt_peer = (
                    template_peer.replace("{peer_review_skill}", peer_skill_text)
                    .replace("{paper_text}", paper_text_trunc)
                )
                review_peer = client.generate(
                    prompt_peer,
                    temperature=GEN_TEMPERATURE,
                    max_output_tokens=GEN_MAX_OUTPUT_TOKENS,
                )

                if looks_invalid(review_peer):
                    raise RuntimeError(
                        "INVALID_REVIEW_OUTPUT: model claimed text was missing or wrote a non-faithful generic review."
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

            # --- Vanilla baseline condition ---
            vanilla_result: dict
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
                        "INVALID_REVIEW_OUTPUT: model claimed text was missing or wrote a non-faithful generic review."
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

            out_v.write(json.dumps(vanilla_result, ensure_ascii=False) + "\n")

            print(
                f"Generated peer+vanilla reviews for paper {paper_id} (trunc={trunc_strategy})"
            )

    print(f"Done. Saved peer to: {out_peer}")
    print(f"Done. Saved vanilla to: {out_vanilla}")


if __name__ == "__main__":
    main()

