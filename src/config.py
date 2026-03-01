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

# -------- Multi-judge output filenames --------
JUDGMENTS_PAIRWISE_JUDGE1_JSONL = JUDGMENTS_DIR / "judgments_pairwise_claude.jsonl"
JUDGMENTS_PAIRWISE_JUDGE2_JSONL = JUDGMENTS_DIR / "judgments_pairwise_gpt.jsonl"
# Legacy alias (kept for backward compat with summarize)
JUDGMENTS_PAIRWISE_JSONL = JUDGMENTS_PAIRWISE_JUDGE1_JSONL

# -------- Peer-review skill --------
SKILL_PATH = PROJECT_ROOT / "peer-review-skills" / "REVIEW_SKILL.md"

# -------- Data limits --------
DEFAULT_PAPER_MAX_CHARS = 50000

# -------- Generation: NO truncation --------
# All papers fit within context (max ~22K tokens, models support 128K+)
GEN_PAPER_MAX_CHARS = 60000  # Above max paper size (50K) — effectively no truncation

# -------- Judge context limits --------
# Full paper + full GT + 2 reviews still fits (max ~29K tokens)
JUDGE_PAPER_MAX_CHARS = 60000  # No truncation
JUDGE_GT_MAX_CHARS = 30000    # No truncation (max GT is ~28K chars)

# -------- SAMPLE (used only by 01b_make_sample) --------
SAMPLE_SIZE = 10
RANDOM_SEED = 42

# -------- OpenRouter --------
LLM_PROVIDER = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# -------- Generation Model --------
GEN_MODEL_NAME = "openai/gpt-5.2"

# -------- Judge Models (multi-judge design) --------
JUDGE_MODEL_NAME = "anthropic/claude-opus-4.6"    # Primary (different family → no self-preference)
JUDGE_MODEL_NAME_2 = "openai/gpt-5.2"             # Secondary (same family → tests self-preference)

# -------- Temperature --------
GEN_TEMPERATURE = 0.4   # Slightly higher for natural, diverse reviews
JUDGE_TEMPERATURE = 0.0  # Deterministic judging (standard in LLM-judge literature)

# -------- Token limits --------
GEN_MAX_OUTPUT_TOKENS = 3000   # Was 1500 — truncated 7-section skill reviews (critical fix)
JUDGE_MAX_OUTPUT_TOKENS = 1000

# -------- Retry / robustness --------
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # Base delay in seconds (exponential backoff)


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)