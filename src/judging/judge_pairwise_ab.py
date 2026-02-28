import json
import random
from pathlib import Path

from src.config import (
    PAIRS_JSONL_PATH,
    REVIEWS_PEER_JSONL,
    REVIEWS_VANILLA_JSONL,
    JUDGMENTS_PAIRWISE_JSONL,
    JUDGE_MODEL_NAME,
    JUDGE_TEMPERATURE,
    JUDGE_MAX_OUTPUT_TOKENS,
    JUDGE_PAPER_MAX_CHARS,
    JUDGE_GT_MAX_CHARS,
    RANDOM_SEED,
    ensure_dirs,
)
from src.generation.llm_client import LLMClient


PROMPT_PATH = Path("prompts/judge_pairwise_ab.txt")


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


def main():
    """
    Pairwise LLM-as-a-Judge (full dataset):
      - Uses pairs.jsonl for (paper_text, ground_truth)
      - Uses reviews_peer.jsonl and reviews_vanilla.jsonl
      - Truncates paper/ground_truth for judge context limits
      - Randomly assigns which condition is A/B per paper (blind)
    Output: outputs/judgments/judgments_pairwise.jsonl
    """
    ensure_dirs()
    random.seed(RANDOM_SEED)

    if not PAIRS_JSONL_PATH.exists():
        raise FileNotFoundError(
            f"Pairs file not found: {PAIRS_JSONL_PATH}. Run scripts/01_build_pairs.py first."
        )

    template = load_prompt()
    client = LLMClient(model_name=JUDGE_MODEL_NAME)

    pairs_by_id = {
        row["paper_id"]: row for row in read_jsonl(PAIRS_JSONL_PATH)
    }

    peer_path = REVIEWS_PEER_JSONL
    vanilla_path = REVIEWS_VANILLA_JSONL

    peer_by_id = {}
    if peer_path.exists():
        for row in read_jsonl(peer_path):
            if row.get("error"):
                continue
            peer_by_id[row["paper_id"]] = row

    vanilla_by_id = {}
    if vanilla_path.exists():
        for row in read_jsonl(vanilla_path):
            if row.get("error"):
                continue
            vanilla_by_id[row["paper_id"]] = row

    common_ids = sorted(
        set(pairs_by_id.keys()) & set(peer_by_id.keys()) & set(vanilla_by_id.keys())
    )

    out_path = JUDGMENTS_PAIRWISE_JSONL

    with open(out_path, "w", encoding="utf-8") as out:
        for paper_id in common_ids:
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
            if random.random() < 0.5:
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
                parsed = json.loads(judge_out)
            except Exception as e:
                parsed = {
                    "winner": "tie",
                    "reasoning": f"ERROR: {str(e)}",
                }

            winner = parsed.get("winner")
            reasoning = parsed.get("reasoning")

            result = {
                "paper_id": paper_id,
                "cond_A": cond_A,
                "cond_B": cond_B,
                "winner": winner,
                "reasoning": reasoning,
            }

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Pairwise judged paper {paper_id} (A={cond_A}, B={cond_B}, winner={winner})")

    print(f"Pairwise judging complete. Saved to: {out_path}")


if __name__ == "__main__":
    main()

