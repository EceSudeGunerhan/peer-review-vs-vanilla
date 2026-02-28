# src/judging/judge_pairwise.py

import json
import random
from pathlib import Path

from src.config import (
    SAMPLE_PAIRS_JSONL_PATH,
    GENERATIONS_DIR,
    JUDGMENTS_DIR,
    JUDGE_MODEL_NAME,
    JUDGE_TEMPERATURE,
    JUDGE_MAX_OUTPUT_TOKENS,
    RANDOM_SEED,
    ensure_dirs,
)
from src.generation.llm_client import LLMClient


PROMPT_PATH = Path("prompts/judge_pairwise.txt")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def main():
    ensure_dirs()
    random.seed(RANDOM_SEED)

    client = LLMClient(model_name=JUDGE_MODEL_NAME)
    template = load_prompt()

    gen_path = GENERATIONS_DIR / "reviews_sample.jsonl"
    out_path = JUDGMENTS_DIR / "judgments_sample.jsonl"

    with open(out_path, "w", encoding="utf-8") as out:

        for row in read_jsonl(gen_path):

            if row.get("error"):
                continue

            paper_id = row["paper_id"]
            generated_review = row["generated_review"]

            prompt = template.replace("{evaluation}", generated_review)

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

            result = {
                "paper_id": paper_id,
                "winner": parsed.get("winner"),
                "reasoning": parsed.get("reasoning"),
            }

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Judged paper {paper_id}")

    print("Judging complete.")


if __name__ == "__main__":
    main()