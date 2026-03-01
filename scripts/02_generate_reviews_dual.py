# Run: python scripts/02_generate_reviews_dual.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.generate_reviews_dual import main

if __name__ == "__main__":
    main()
