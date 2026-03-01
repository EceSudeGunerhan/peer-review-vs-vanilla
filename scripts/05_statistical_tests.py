# Run: python scripts/05_statistical_tests.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reports.statistical_tests import main

if __name__ == "__main__":
    main()
