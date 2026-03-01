# Run: python scripts/03_judge_pairwise_ab.py
# Usage:
#   python scripts/03_judge_pairwise_ab.py          → Judge 1 (Claude, primary)
#   python scripts/03_judge_pairwise_ab.py --judge 2 → Judge 2 (GPT, secondary)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.judging.judge_pairwise_ab import main

if __name__ == "__main__":
    main()
