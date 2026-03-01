# Run: python scripts/03b_judge_pairwise_ab_secondary.py
# Secondary judge pass using GPT-5.2

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = [sys.argv[0], "--judge", "2"]  # Force judge 2
from src.judging.judge_pairwise_ab import main

if __name__ == "__main__":
    main()
