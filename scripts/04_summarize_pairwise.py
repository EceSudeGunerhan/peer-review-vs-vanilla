# Run: python scripts/04_summarize_pairwise.py
# Usage:
#   python scripts/04_summarize_pairwise.py          → Summarize Judge 1 (Claude)
#   python scripts/04_summarize_pairwise.py --judge 2 → Summarize Judge 2 (GPT)
#   python scripts/04_summarize_pairwise.py --all     → Summarize both judges

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reports.summarize_pairwise import main

if __name__ == "__main__":
    main()
