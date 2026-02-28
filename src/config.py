# src/config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ICLR2017_DIR = RAW_DIR / "iclr_2017"
PARSED_PDFS_DIR = ICLR2017_DIR / "parsed_pdfs"
REVIEWS_DIR = ICLR2017_DIR / "reviews"

PAIRS_JSONL_PATH = PROCESSED_DIR / "pairs.jsonl"
SAMPLE_PAIRS_JSONL_PATH = PROCESSED_DIR / "sample_pairs.jsonl"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GENERATIONS_DIR = OUTPUTS_DIR / "generations"
JUDGMENTS_DIR = OUTPUTS_DIR / "judgments"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Full-dataset output filenames (no "sample" suffix)
REVIEWS_PEER_JSONL = GENERATIONS_DIR / "reviews_peer.jsonl"
REVIEWS_VANILLA_JSONL = GENERATIONS_DIR / "reviews_vanilla.jsonl"
JUDGMENTS_PAIRWISE_JSONL = JUDGMENTS_DIR / "judgments_pairwise.jsonl"

# -------- Data limits --------
DEFAULT_PAPER_MAX_CHARS = 50000

# -------- Judge context limits (for pairwise prompt) --------
JUDGE_PAPER_MAX_CHARS = 8000
JUDGE_GT_MAX_CHARS = 6000

# -------- SAMPLE (used only by 01b_make_sample) --------
SAMPLE_SIZE = 10
RANDOM_SEED = 42

# -------- OpenRouter --------
LLM_PROVIDER = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Generation Model
GEN_MODEL_NAME = "openai/gpt-4o"

# Judge Model
JUDGE_MODEL_NAME = "anthropic/claude-3.5-sonnet"

GEN_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.0

GEN_MAX_OUTPUT_TOKENS = 1500
JUDGE_MAX_OUTPUT_TOKENS = 800


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)