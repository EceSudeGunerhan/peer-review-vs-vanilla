# src/generation/generate_reviews.py

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


PROMPT_PATH = Path("prompts/generation_prompt.txt")

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


def load_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
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
    ensure_dirs()

    client = LLMClient(model_name=GEN_MODEL_NAME)
    template = load_prompt()

    out_path = GENERATIONS_DIR / "reviews_sample.jsonl"

    with open(out_path, "w", encoding="utf-8") as out:
        for row in read_jsonl(SAMPLE_PAIRS_JSONL_PATH):
            paper_id = row["paper_id"]
            paper_text = row["paper_text"]

            paper_text, trunc_strategy = smart_truncate(paper_text, PAPER_TEXT_MAX_CHARS)

            prompt = template.replace("{paper_text}", paper_text)

            try:
                review = client.generate(
                    prompt,
                    temperature=GEN_TEMPERATURE,          # öneri: 0.2–0.4 arası
                    max_output_tokens=GEN_MAX_OUTPUT_TOKENS,  # öneri: 1000–1600 arası
                )

                if looks_invalid(review):
                    raise RuntimeError(
                        "INVALID_REVIEW_OUTPUT: model claimed text was missing or wrote a non-faithful generic review."
                    )

                result = {
                    "paper_id": paper_id,
                    "generated_review": review,
                    "error": None,
                    "paper_text_chars": len(paper_text),
                    "truncation": trunc_strategy,
                    "model": GEN_MODEL_NAME,
                }

            except Exception as e:
                result = {
                    "paper_id": paper_id,
                    "generated_review": None,
                    "error": str(e),
                    "paper_text_chars": len(paper_text),
                    "truncation": trunc_strategy,
                    "model": GEN_MODEL_NAME,
                }

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Generated review for paper {paper_id} (trunc={trunc_strategy})")

    print(f"Done. Saved to: {out_path}")


if __name__ == "__main__":
    main()