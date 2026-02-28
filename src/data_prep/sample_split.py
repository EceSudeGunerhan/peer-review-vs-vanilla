# src/data_prep/sample_split.py
# Create a small sample dataset from pairs.jsonl for quick testing

import random
from typing import List, Dict

from src.config import (
    PAIRS_JSONL_PATH,
    SAMPLE_PAIRS_JSONL_PATH,
    RANDOM_SEED,
    ensure_dirs,
)
from src.utils import read_jsonl, write_jsonl


def make_sample(sample_size: int = 20) -> List[Dict]:
    """
    Read full pairs.jsonl and write a smaller sample JSONL.
    Default: 20 examples.
    """

    random.seed(RANDOM_SEED)

    pairs = list(read_jsonl(PAIRS_JSONL_PATH))
    if not pairs:
        raise ValueError(
            f"No data found in {PAIRS_JSONL_PATH}. "
            "Run scripts/01_build_pairs.py first."
        )

    if sample_size >= len(pairs):
        return pairs

    sampled = random.sample(pairs, sample_size)
    return sampled


def main(sample_size: int = 20):
    ensure_dirs()

    sampled = make_sample(sample_size=sample_size)
    write_jsonl(SAMPLE_PAIRS_JSONL_PATH, sampled)

    print(f"Sample size: {len(sampled)}")
    print(f"Saved to: {SAMPLE_PAIRS_JSONL_PATH}")


if __name__ == "__main__":
    main()