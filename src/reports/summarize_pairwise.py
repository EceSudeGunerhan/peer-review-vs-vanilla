import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

from src.config import (
    JUDGMENTS_PAIRWISE_JUDGE1_JSONL,
    JUDGMENTS_PAIRWISE_JUDGE2_JSONL,
    JUDGE_MODEL_NAME,
    JUDGE_MODEL_NAME_2,
    GEN_MODEL_NAME,
    GEN_TEMPERATURE,
    GEN_MAX_OUTPUT_TOKENS,
    JUDGE_TEMPERATURE,
    REPORTS_DIR,
    ensure_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_judgments(in_path: Path, judge_label: str, judge_model: str):
    """Summarize a single judge's results to JSON, CSV, and MD."""
    if not in_path.exists():
        raise FileNotFoundError(f"Judgments not found: {in_path}")

    total = 0
    peer_wins = 0
    vanilla_wins = 0
    ties = 0

    for row in read_jsonl(in_path):
        total += 1
        winner = (row.get("winner") or "").lower().strip()
        cond_A = row.get("cond_A")
        cond_B = row.get("cond_B")

        if winner == "tie":
            ties += 1
        elif winner == "a":
            if cond_A == "peer":
                peer_wins += 1
            elif cond_A == "vanilla":
                vanilla_wins += 1
        elif winner == "b":
            if cond_B == "peer":
                peer_wins += 1
            elif cond_B == "vanilla":
                vanilla_wins += 1

    non_tie = max(total - ties, 1)

    summary = {
        "judge": judge_label,
        "judge_model": judge_model,
        "gen_model": GEN_MODEL_NAME,
        "gen_temperature": GEN_TEMPERATURE,
        "gen_max_tokens": GEN_MAX_OUTPUT_TOKENS,
        "judge_temperature": JUDGE_TEMPERATURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_examples": total,
        "peer_wins": peer_wins,
        "vanilla_wins": vanilla_wins,
        "ties": ties,
        "peer_win_rate_total": peer_wins / total if total else 0.0,
        "vanilla_win_rate_total": vanilla_wins / total if total else 0.0,
        "tie_rate_total": ties / total if total else 0.0,
        "peer_win_rate_non_tie": peer_wins / non_tie if non_tie else 0.0,
        "vanilla_win_rate_non_tie": vanilla_wins / non_tie if non_tie else 0.0,
    }

    suffix = f"_{judge_label}"

    # JSON
    json_path = REPORTS_DIR / f"pairwise_summary{suffix}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV
    csv_path = REPORTS_DIR / f"pairwise_summary{suffix}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("condition,wins,win_rate_total,win_rate_non_tie\n")
        f.write(
            f"peer,{peer_wins},{summary['peer_win_rate_total']},"
            f"{summary['peer_win_rate_non_tie']}\n"
        )
        f.write(
            f"vanilla,{vanilla_wins},{summary['vanilla_win_rate_total']},"
            f"{summary['vanilla_win_rate_non_tie']}\n"
        )
        f.write(f"tie,{ties},{summary['tie_rate_total']},\n")

    # Markdown
    md_path = REPORTS_DIR / f"pairwise_summary{suffix}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Pairwise Results — {judge_label}\n\n")
        f.write(f"**Judge**: {judge_model}  \n")
        f.write(f"**Generator**: {GEN_MODEL_NAME}  \n")
        f.write(f"**Timestamp**: {summary['timestamp']}  \n\n")
        f.write(f"- Total examples: {total}\n")
        f.write(f"- Peer wins: {peer_wins}\n")
        f.write(f"- Vanilla wins: {vanilla_wins}\n")
        f.write(f"- Ties: {ties}\n\n")
        f.write("## Win rates (over all examples)\n\n")
        f.write(f"- Peer: {summary['peer_win_rate_total']:.3f}\n")
        f.write(f"- Vanilla: {summary['vanilla_win_rate_total']:.3f}\n")
        f.write(f"- Tie: {summary['tie_rate_total']:.3f}\n\n")
        f.write("## Win rates (excluding ties)\n\n")
        f.write(f"- Peer: {summary['peer_win_rate_non_tie']:.3f}\n")
        f.write(f"- Vanilla: {summary['vanilla_win_rate_non_tie']:.3f}\n")

    logger.info(f"Wrote: {json_path}")
    logger.info(f"Wrote: {csv_path}")
    logger.info(f"Wrote: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize pairwise results")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--judge", type=int, default=None, choices=[1, 2],
        help="Summarize specific judge (1=Claude, 2=GPT)"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Summarize all available judges"
    )
    args = parser.parse_args()

    ensure_dirs()

    judges = []
    if args.all:
        if JUDGMENTS_PAIRWISE_JUDGE1_JSONL.exists():
            judges.append((JUDGMENTS_PAIRWISE_JUDGE1_JSONL, "judge1_claude", JUDGE_MODEL_NAME))
        if JUDGMENTS_PAIRWISE_JUDGE2_JSONL.exists():
            judges.append((JUDGMENTS_PAIRWISE_JUDGE2_JSONL, "judge2_gpt", JUDGE_MODEL_NAME_2))
    elif args.judge == 2:
        judges.append((JUDGMENTS_PAIRWISE_JUDGE2_JSONL, "judge2_gpt", JUDGE_MODEL_NAME_2))
    else:
        # Default: judge 1
        judges.append((JUDGMENTS_PAIRWISE_JUDGE1_JSONL, "judge1_claude", JUDGE_MODEL_NAME))

    for in_path, label, model in judges:
        summarize_judgments(in_path, label, model)


if __name__ == "__main__":
    main()
