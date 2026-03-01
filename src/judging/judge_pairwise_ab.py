import json
import random
import hashlib
import logging
import argparse
from pathlib import Path

from src.config import (
    PAIRS_JSONL_PATH,
    REVIEWS_PEER_JSONL,
    REVIEWS_VANILLA_JSONL,
    JUDGMENTS_PAIRWISE_JUDGE1_JSONL,
    JUDGMENTS_PAIRWISE_JUDGE2_JSONL,
    JUDGE_MODEL_NAME,
    JUDGE_MODEL_NAME_2,
    JUDGE_TEMPERATURE,
    JUDGE_MAX_OUTPUT_TOKENS,
    JUDGE_PAPER_MAX_CHARS,
    JUDGE_GT_MAX_CHARS,
    RANDOM_SEED,
    PROJECT_ROOT,
    ensure_dirs,
)
from src.generation.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PROMPT_PATH = PROJECT_ROOT / "prompts" / "judge_pairwise_ab.txt"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def truncate_for_judge(text: str, max_chars: int) -> str:
    """Truncate text for judge prompt to fit context window."""
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars].rstrip() + "\n\n[...TRUNCATED...]"


def load_existing_ids(path: Path) -> set:
    """Load paper IDs already judged (for resume)."""
    ids = set()
    if path.exists():
        for row in read_jsonl(path):
            ids.add(row["paper_id"])
    return ids


def main():
    """
    Pairwise LLM-as-a-Judge with multi-judge support and resume.

    Usage:
      python scripts/03_judge_pairwise_ab.py               # Judge 1 (Claude, primary)
      python scripts/03_judge_pairwise_ab.py --judge 2      # Judge 2 (GPT, secondary)
    """
    parser = argparse.ArgumentParser(description="Pairwise LLM Judge")
    parser.add_argument(
        "--judge", type=int, default=1, choices=[1, 2],
        help="Judge number: 1=primary (Claude), 2=secondary (GPT)"
    )
    args = parser.parse_args()

    ensure_dirs()
    random.seed(RANDOM_SEED)

    # Select judge model and output path
    if args.judge == 1:
        judge_model = JUDGE_MODEL_NAME
        out_path = JUDGMENTS_PAIRWISE_JUDGE1_JSONL
        judge_label = "judge1_claude"
    else:
        judge_model = JUDGE_MODEL_NAME_2
        out_path = JUDGMENTS_PAIRWISE_JUDGE2_JSONL
        judge_label = "judge2_gpt"

    logger.info(f"Judge: {judge_label} ({judge_model})")
    logger.info(f"Output: {out_path}")

    if not PAIRS_JSONL_PATH.exists():
        raise FileNotFoundError(
            f"Pairs file not found: {PAIRS_JSONL_PATH}. "
            f"Run scripts/01_build_pairs.py first."
        )

    template = load_prompt()
    client = LLMClient(model_name=judge_model)

    # Load data
    pairs_by_id = {
        row["paper_id"]: row for row in read_jsonl(PAIRS_JSONL_PATH)
    }

    peer_by_id = {}
    if REVIEWS_PEER_JSONL.exists():
        for row in read_jsonl(REVIEWS_PEER_JSONL):
            if row.get("error") is None:
                peer_by_id[row["paper_id"]] = row

    vanilla_by_id = {}
    if REVIEWS_VANILLA_JSONL.exists():
        for row in read_jsonl(REVIEWS_VANILLA_JSONL):
            if row.get("error") is None:
                vanilla_by_id[row["paper_id"]] = row

    common_ids = sorted(
        set(pairs_by_id.keys()) & set(peer_by_id.keys()) & set(vanilla_by_id.keys())
    )
    total = len(common_ids)
    logger.info(f"Papers to judge: {total}")

    # Resume: skip already-judged papers
    done_ids = load_existing_ids(out_path)
    if done_ids:
        logger.info(f"Resuming: {len(done_ids)} already judged, skipping.")

    mode = "a" if done_ids else "w"

    with open(out_path, mode, encoding="utf-8") as out:
        for idx, paper_id in enumerate(common_ids, 1):
            if paper_id in done_ids:
                continue

            pair = pairs_by_id[paper_id]
            peer_row = peer_by_id[paper_id]
            vanilla_row = vanilla_by_id[paper_id]

            paper_text = truncate_for_judge(
                pair["paper_text"], JUDGE_PAPER_MAX_CHARS
            )
            ground_truth = truncate_for_judge(
                pair["ground_truth"], JUDGE_GT_MAX_CHARS
            )
            peer_review = peer_row["generated_review"]
            vanilla_review = vanilla_row["generated_review"]

            # Randomly decide mapping to A/B (blind)
            # Use deterministic hash so both judges get SAME assignment
            seed = int(hashlib.md5(paper_id.encode()).hexdigest()[:8], 16)
            rng = random.Random(RANDOM_SEED + seed)
            if rng.random() < 0.5:
                cond_A, cond_B = "peer", "vanilla"
                review_A, review_B = peer_review, vanilla_review
            else:
                cond_A, cond_B = "vanilla", "peer"
                review_A, review_B = vanilla_review, peer_review

            prompt = (
                template.replace("{paper_text}", paper_text)
                .replace("{ground_truth}", ground_truth)
                .replace("{review_A}", review_A)
                .replace("{review_B}", review_B)
            )

            try:
                judge_out = client.generate(
                    prompt,
                    temperature=JUDGE_TEMPERATURE,
                    max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                )
                # Strip markdown code fences if present
                cleaned = judge_out.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned.rsplit("```", 1)[0]
                parsed = json.loads(cleaned.strip())
            except Exception as e:
                parsed = {
                    "winner": "tie",
                    "reasoning": f"ERROR: {str(e)[:200]}",
                }

            winner = parsed.get("winner")
            reasoning = parsed.get("reasoning")

            result = {
                "paper_id": paper_id,
                "cond_A": cond_A,
                "cond_B": cond_B,
                "winner": winner,
                "reasoning": reasoning,
                "judge_model": judge_model,
            }

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

            logger.info(
                f"[{idx}/{total}] paper={paper_id} "
                f"A={cond_A} B={cond_B} winner={winner}"
            )

    logger.info(f"Judging complete ({judge_label}). Saved to: {out_path}")


if __name__ == "__main__":
    main()
