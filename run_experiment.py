#!/usr/bin/env python3
"""
Full experiment runner with checkpointing and graceful shutdown.

Usage:
    python run_experiment.py              # Run from beginning (or resume)
    python run_experiment.py --from 3     # Resume from step 3
    python run_experiment.py --step 2     # Run only step 2
    python run_experiment.py --status     # Show checkpoint status

Steps:
    0. Build pairs — PRE-EXPERIMENT, only if pairs.jsonl missing
    1. Generate reviews (peer + vanilla)
    2. Judge — primary (Claude)
    3. Judge — secondary (GPT)
    4. Summarize + statistical tests

Ctrl+C is safe — each step writes results line-by-line with flush.
Resume will skip already-processed papers automatically.
"""

import os
import sys
import json
import time
import signal
import logging
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PAIRS_JSONL_PATH,
    REVIEWS_PEER_JSONL,
    REVIEWS_VANILLA_JSONL,
    JUDGMENTS_PAIRWISE_JUDGE1_JSONL,
    JUDGMENTS_PAIRWISE_JUDGE2_JSONL,
    REPORTS_DIR,
    ensure_dirs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "experiment.log", mode="a"),
    ],
)
logger = logging.getLogger("experiment")

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoint.json"

# --- Graceful shutdown ---
_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("Force quit (second Ctrl+C). Exiting immediately.")
        sys.exit(1)
    _shutdown_requested = True
    logger.warning(
        "\n⚠️  Shutdown requested (Ctrl+C). "
        "Finishing current paper, then saving checkpoint..."
    )

signal.signal(signal.SIGINT, _signal_handler)


def count_jsonl(path: Path) -> int:
    """Count lines in a JSONL file."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def count_successful(path: Path) -> int:
    """Count successful (non-error) entries in a JSONL file."""
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("error") is None:
                    count += 1
    return count


def save_checkpoint(step: int, status: str, details: dict = None):
    """Save current progress to checkpoint file."""
    ensure_dirs()
    checkpoint = {
        "last_completed_step": step,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": details or {},
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved: step {step} — {status}")


def load_checkpoint() -> dict:
    """Load checkpoint if it exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"last_completed_step": 0, "status": "not_started"}


def get_status() -> dict:
    """Get current experiment status."""
    cp = load_checkpoint()
    return {
        "checkpoint": cp,
        "pairs": count_jsonl(PAIRS_JSONL_PATH),
        "reviews_peer": count_successful(REVIEWS_PEER_JSONL),
        "reviews_vanilla": count_successful(REVIEWS_VANILLA_JSONL),
        "judgments_claude": count_jsonl(JUDGMENTS_PAIRWISE_JUDGE1_JSONL),
        "judgments_gpt": count_jsonl(JUDGMENTS_PAIRWISE_JUDGE2_JSONL),
        "reports_exist": REPORTS_DIR.exists() and any(REPORTS_DIR.iterdir()) if REPORTS_DIR.exists() else False,
    }


def print_status():
    """Pretty-print experiment status."""
    s = get_status()
    print("\n" + "=" * 60)
    print("EXPERIMENT STATUS")
    print("=" * 60)
    print(f"  Checkpoint:        Step {s['checkpoint']['last_completed_step']} — {s['checkpoint']['status']}")
    print(f"  Last updated:      {s['checkpoint'].get('timestamp', 'N/A')}")
    print(f"  Pairs built:       {s['pairs']}")
    print(f"  Reviews (peer):    {s['reviews_peer']}")
    print(f"  Reviews (vanilla): {s['reviews_vanilla']}")
    print(f"  Judgments (Claude): {s['judgments_claude']}")
    print(f"  Judgments (GPT):    {s['judgments_gpt']}")
    print(f"  Reports:           {'✅' if s['reports_exist'] else '❌'}")
    print("=" * 60 + "\n")


# ===== STEP FUNCTIONS =====

def step_0_build_pairs():
    """Step 0 (pre-experiment): Build paper-review pairs from raw data."""
    logger.info("STEP 0: Building paper-review pairs (pre-experiment)")

    if PAIRS_JSONL_PATH.exists():
        n = count_jsonl(PAIRS_JSONL_PATH)
        logger.info(f"pairs.jsonl already exists ({n} pairs). Skipping.")
        return

    from src.data_prep.build_pairs import main as build_main
    build_main()

    n = count_jsonl(PAIRS_JSONL_PATH)
    logger.info(f"Step 0 complete: {n} pairs built.")


def step_1_generate_reviews():
    """Step 1: Generate peer + vanilla reviews for all papers."""
    logger.info("STEP 1: Generating reviews (peer + vanilla)")

    n_pairs = count_jsonl(PAIRS_JSONL_PATH)
    n_peer = count_successful(REVIEWS_PEER_JSONL)
    n_vanilla = count_successful(REVIEWS_VANILLA_JSONL)

    if n_peer >= n_pairs and n_vanilla >= n_pairs:
        logger.info(
            f"All reviews already generated "
            f"(peer={n_peer}, vanilla={n_vanilla}). Skipping."
        )
        return

    logger.info(
        f"Progress: peer={n_peer}/{n_pairs}, vanilla={n_vanilla}/{n_pairs}. "
        f"Resuming..."
    )

    from src.generation.generate_reviews_dual import main as gen_main
    gen_main()

    n_peer = count_successful(REVIEWS_PEER_JSONL)
    n_vanilla = count_successful(REVIEWS_VANILLA_JSONL)
    logger.info(f"Step 1 complete: peer={n_peer}, vanilla={n_vanilla}")
    save_checkpoint(1, "reviews_generated", {
        "peer": n_peer, "vanilla": n_vanilla
    })


def step_2_judge_primary():
    """Step 2: Run primary judge (Claude)."""
    logger.info("STEP 2: Judging — Primary (Claude)")

    # Override sys.argv for argparse inside the module
    sys.argv = ["judge", "--judge", "1"]
    from src.judging.judge_pairwise_ab import main as judge_main
    judge_main()

    n = count_jsonl(JUDGMENTS_PAIRWISE_JUDGE1_JSONL)
    logger.info(f"Step 2 complete: {n} judgments (Claude)")
    save_checkpoint(2, "judge1_complete", {"judgments_claude": n})


def step_3_judge_secondary():
    """Step 3: Run secondary judge (GPT)."""
    logger.info("STEP 3: Judging — Secondary (GPT)")

    sys.argv = ["judge", "--judge", "2"]
    # Re-import to reset argparse state
    import importlib
    import src.judging.judge_pairwise_ab as judge_mod
    importlib.reload(judge_mod)
    judge_mod.main()

    n = count_jsonl(JUDGMENTS_PAIRWISE_JUDGE2_JSONL)
    logger.info(f"Step 3 complete: {n} judgments (GPT)")
    save_checkpoint(3, "judge2_complete", {"judgments_gpt": n})


def step_4_summarize_and_stats():
    """Step 4: Summarize results + run statistical tests."""
    logger.info("STEP 4: Summarizing + Statistical Tests")

    # Summarize both judges
    sys.argv = ["summarize", "--all"]
    from src.reports.summarize_pairwise import main as sum_main
    sum_main()

    # Statistical tests
    from src.reports.statistical_tests import main as stat_main
    stat_main()

    logger.info("Step 4 complete: reports generated")
    save_checkpoint(4, "experiment_complete")

    # Print final results
    stat_path = REPORTS_DIR / "statistical_tests.md"
    if stat_path.exists():
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(stat_path.read_text())


# ===== MAIN =====

STEPS = {
    1: ("Generate reviews", step_1_generate_reviews),
    2: ("Judge — Claude", step_2_judge_primary),
    3: ("Judge — GPT", step_3_judge_secondary),
    4: ("Summarize + Stats", step_4_summarize_and_stats),
}
MAX_STEP = 4


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument(
        "--from", type=int, dest="from_step", default=None,
        help="Resume from this step (1-4)"
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Run only this step (1-4)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current experiment status and exit"
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.status:
        print_status()
        return

    # Verify API key
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY not set. Export it first:")
        logger.error('  export OPENROUTER_API_KEY="sk-or-..."')
        sys.exit(1)

    # Ensure pairs exist (pre-experiment step)
    if not PAIRS_JSONL_PATH.exists():
        logger.info("pairs.jsonl not found — running pre-experiment build...")
        step_0_build_pairs()
    else:
        logger.info(f"pairs.jsonl ready ({count_jsonl(PAIRS_JSONL_PATH)} pairs)")

    # Determine which steps to run
    if args.step:
        steps_to_run = [args.step]
    elif args.from_step:
        steps_to_run = list(range(args.from_step, MAX_STEP + 1))
    else:
        # Auto-resume from checkpoint
        cp = load_checkpoint()
        start = cp["last_completed_step"] + 1
        if start > MAX_STEP:
            logger.info("Experiment already complete! Use --from to re-run steps.")
            print_status()
            return
        steps_to_run = list(range(start, MAX_STEP + 1))

    logger.info(f"Steps to run: {steps_to_run}")
    t0 = time.time()

    for step_num in steps_to_run:
        if _shutdown_requested:
            logger.warning(f"Shutdown requested. Stopping before step {step_num}.")
            break

        name, func = STEPS[step_num]
        logger.info(f"\n{'='*60}")
        logger.info(f"▶ Starting Step {step_num}/{MAX_STEP}: {name}")
        logger.info(f"{'='*60}")

        try:
            func()
        except KeyboardInterrupt:
            logger.warning(f"Interrupted during step {step_num}. Progress saved.")
            save_checkpoint(step_num - 1, f"interrupted_at_step_{step_num}")
            break
        except Exception as e:
            logger.error(f"Error in step {step_num}: {e}", exc_info=True)
            save_checkpoint(step_num - 1, f"error_at_step_{step_num}: {str(e)[:200]}")
            raise

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    print_status()


if __name__ == "__main__":
    main()
