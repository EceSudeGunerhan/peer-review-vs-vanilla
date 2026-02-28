import json
from pathlib import Path

from src.config import JUDGMENTS_DIR, REPORTS_DIR, ensure_dirs


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    """
    Summarize pairwise LLM-as-a-Judge results.

    Input:
      - outputs/judgments/judgments_pairwise_sample.jsonl
        Each line:
          {
            "paper_id": ...,
            "cond_A": "peer" | "vanilla",
            "cond_B": "peer" | "vanilla",
            "winner": "A" | "B" | "tie",
            "reasoning": "..."
          }

    Outputs (in outputs/reports/):
      - pairwise_summary.json
      - pairwise_summary.csv
      - pairwise_summary.md
    """
    ensure_dirs()

    in_path = JUDGMENTS_DIR / "judgments_pairwise_sample.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Judgments file not found: {in_path}")

    total = 0
    peer_wins = 0
    vanilla_wins = 0
    ties = 0

    for row in read_jsonl(in_path):
        total += 1
        winner = (row.get("winner") or "").lower()
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

    # JSON summary
    json_path = REPORTS_DIR / "pairwise_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV summary (simple 3-row format)
    csv_path = REPORTS_DIR / "pairwise_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("condition,wins,win_rate_total,win_rate_non_tie\n")
        f.write(
            f"peer,{peer_wins},{summary['peer_win_rate_total']},{summary['peer_win_rate_non_tie']}\n"
        )
        f.write(
            f"vanilla,{vanilla_wins},{summary['vanilla_win_rate_total']},{summary['vanilla_win_rate_non_tie']}\n"
        )
        f.write(f"tie,{ties},{summary['tie_rate_total']},\n")

    # Markdown summary
    md_path = REPORTS_DIR / "pairwise_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Pairwise Peer vs Vanilla Results\n\n")
        f.write(f"- Total examples: {total}\n")
        f.write(f"- Peer wins: {peer_wins}\n")
        f.write(f"- Vanilla wins: {vanilla_wins}\n")
        f.write(f"- Ties: {ties}\n\n")
        f.write("## Win rates (over all examples)\n\n")
        f.write(
            f"- Peer win rate: {summary['peer_win_rate_total']:.3f}\n"
        )
        f.write(
            f"- Vanilla win rate: {summary['vanilla_win_rate_total']:.3f}\n"
        )
        f.write(f"- Tie rate: {summary['tie_rate_total']:.3f}\n\n")
        f.write("## Win rates (excluding ties)\n\n")
        f.write(
            f"- Peer win rate (non-tie): {summary['peer_win_rate_non_tie']:.3f}\n"
        )
        f.write(
            f"- Vanilla win rate (non-tie): {summary['vanilla_win_rate_non_tie']:.3f}\n"
        )

    print(f"Wrote JSON summary to: {json_path}")
    print(f"Wrote CSV summary to: {csv_path}")
    print(f"Wrote Markdown summary to: {md_path}")


if __name__ == "__main__":
    main()

