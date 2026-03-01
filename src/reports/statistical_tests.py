# src/reports/statistical_tests.py
"""
Statistical significance tests for pairwise LLM-as-a-Judge results.

Tests:
  1. Binomial test — H₀: peer win rate = 0.5
  2. Sign test — non-parametric pairwise
  3. 95% Wilson CI — confidence interval for peer win rate
  4. Cohen's h — effect size for proportion differences
  5. Cohen's κ — inter-judge agreement (when two judge files available)
"""

import json
import math
import argparse
from pathlib import Path
from collections import Counter

from src.config import (
    JUDGMENTS_PAIRWISE_JUDGE1_JSONL,
    JUDGMENTS_PAIRWISE_JUDGE2_JSONL,
    REPORTS_DIR,
    ensure_dirs,
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_winner(row: dict) -> str:
    """Map A/B winner back to peer/vanilla/tie."""
    winner = (row.get("winner") or "").lower().strip()
    cond_A = row.get("cond_A")
    cond_B = row.get("cond_B")

    if winner == "tie":
        return "tie"
    elif winner == "a":
        return cond_A
    elif winner == "b":
        return cond_B
    else:
        return "tie"  # Parse errors → tie (conservative)


def load_outcomes(path: Path) -> list[str]:
    """Load judgment file → list of 'peer'/'vanilla'/'tie'."""
    return [resolve_winner(row) for row in read_jsonl(path)]


# ---- Statistical functions (no scipy dependency) ----

def _binomial_cdf(k: int, n: int, p: float) -> float:
    """Exact binomial CDF using log-space for numerical stability."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0

    total = 0.0
    for i in range(k + 1):
        log_prob = (
            _log_comb(n, i) + i * math.log(p) + (n - i) * math.log(1 - p)
        )
        total += math.exp(log_prob)
    return min(total, 1.0)


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient."""
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    )


def binomial_test_two_sided(successes: int, trials: int, p0: float = 0.5) -> float:
    """
    Two-sided binomial test.
    H₀: true proportion = p0.
    Returns p-value.
    """
    if trials == 0:
        return 1.0

    # P(X >= successes) for upper tail
    p_upper = 1.0 - _binomial_cdf(successes - 1, trials, p0)
    # P(X <= successes) for lower tail
    p_lower = _binomial_cdf(successes, trials, p0)

    # Two-sided: 2 * min(lower, upper)
    return min(2.0 * min(p_upper, p_lower), 1.0)


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.
    Default z=1.96 for 95% CI.
    """
    if trials == 0:
        return 0.0, 1.0

    p_hat = successes / trials
    denom = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denom
    spread = z * math.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials
    ) / denom

    return max(0.0, center - spread), min(1.0, center + spread)


def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for difference between two proportions.
    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    |h| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large
    """
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def interpret_effect_size(h: float) -> str:
    """Interpret Cohen's h magnitude."""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "negligible"
    elif abs_h < 0.5:
        return "small"
    elif abs_h < 0.8:
        return "medium"
    else:
        return "large"


def cohens_kappa(outcomes_1: list[str], outcomes_2: list[str]) -> float:
    """
    Cohen's kappa for inter-judge agreement.
    Both lists must have same length and same paper order.
    Categories: 'peer', 'vanilla', 'tie'
    """
    n = len(outcomes_1)
    if n == 0:
        return 0.0

    categories = sorted(set(outcomes_1) | set(outcomes_2))

    # Observed agreement
    agree = sum(1 for a, b in zip(outcomes_1, outcomes_2) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    p_e = 0.0
    for cat in categories:
        f1 = sum(1 for x in outcomes_1 if x == cat) / n
        f2 = sum(1 for x in outcomes_2 if x == cat) / n
        p_e += f1 * f2

    if p_e >= 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa


def interpret_kappa(k: float) -> str:
    """Landis-Koch scale for kappa."""
    if k < 0.0:
        return "poor"
    elif k < 0.20:
        return "slight"
    elif k < 0.40:
        return "fair"
    elif k < 0.60:
        return "moderate"
    elif k < 0.80:
        return "substantial"
    else:
        return "almost_perfect"


def analyze_single_judge(outcomes: list[str], judge_name: str) -> dict:
    """Run all statistical tests for a single judge's outcomes."""
    counts = Counter(outcomes)
    total = len(outcomes)
    peer_wins = counts.get("peer", 0)
    vanilla_wins = counts.get("vanilla", 0)
    ties = counts.get("tie", 0)
    non_tie = peer_wins + vanilla_wins

    # Binomial test (excluding ties)
    p_val = binomial_test_two_sided(peer_wins, non_tie, 0.5) if non_tie > 0 else 1.0

    # Wilson CI for peer win rate (excluding ties)
    ci_low, ci_high = wilson_ci(peer_wins, non_tie) if non_tie > 0 else (0.0, 1.0)

    # Cohen's h (peer rate vs 0.5)
    peer_rate = peer_wins / non_tie if non_tie > 0 else 0.5
    h = cohens_h(peer_rate, 0.5) if non_tie > 0 else 0.0

    return {
        "judge": judge_name,
        "total_examples": total,
        "peer_wins": peer_wins,
        "vanilla_wins": vanilla_wins,
        "ties": ties,
        "peer_win_rate_total": peer_wins / total if total else 0.0,
        "vanilla_win_rate_total": vanilla_wins / total if total else 0.0,
        "tie_rate": ties / total if total else 0.0,
        "peer_win_rate_non_tie": peer_rate,
        "binomial_p_value": p_val,
        "significant_at_005": p_val < 0.05,
        "ci_95_lower": ci_low,
        "ci_95_upper": ci_high,
        "cohens_h": h,
        "effect_size": interpret_effect_size(h),
    }


def run_tests() -> dict:
    """Run all statistical tests across available judge files."""
    ensure_dirs()
    results = {"judges": []}

    # Judge 1
    j1_path = JUDGMENTS_PAIRWISE_JUDGE1_JSONL
    if j1_path.exists():
        outcomes_1 = load_outcomes(j1_path)
        stats_1 = analyze_single_judge(outcomes_1, "judge1_claude")
        results["judges"].append(stats_1)
    else:
        outcomes_1 = None

    # Judge 2
    j2_path = JUDGMENTS_PAIRWISE_JUDGE2_JSONL
    if j2_path.exists():
        outcomes_2 = load_outcomes(j2_path)
        stats_2 = analyze_single_judge(outcomes_2, "judge2_gpt")
        results["judges"].append(stats_2)
    else:
        outcomes_2 = None

    # Inter-judge agreement
    if outcomes_1 is not None and outcomes_2 is not None:
        # Align by paper_id order (both should be sorted same way)
        kappa = cohens_kappa(outcomes_1, outcomes_2)
        results["inter_judge"] = {
            "cohens_kappa": kappa,
            "agreement_level": interpret_kappa(kappa),
            "n_papers_compared": min(len(outcomes_1), len(outcomes_2)),
        }

    return results


def write_results(results: dict):
    """Write statistical test results to JSON and Markdown."""
    # JSON
    json_path = REPORTS_DIR / "statistical_tests.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Markdown
    md_path = REPORTS_DIR / "statistical_tests.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Statistical Significance Tests\n\n")

        for judge in results.get("judges", []):
            f.write(f"## {judge['judge']}\n\n")
            f.write(f"- Total examples: {judge['total_examples']}\n")
            f.write(f"- Peer wins: {judge['peer_wins']}\n")
            f.write(f"- Vanilla wins: {judge['vanilla_wins']}\n")
            f.write(f"- Ties: {judge['ties']}\n\n")
            f.write(f"### Win Rate (excluding ties)\n\n")
            f.write(f"- Peer: {judge['peer_win_rate_non_tie']:.3f}\n")
            f.write(f"- 95% CI: [{judge['ci_95_lower']:.3f}, {judge['ci_95_upper']:.3f}]\n\n")
            f.write(f"### Significance\n\n")
            f.write(f"- Binomial test p-value: {judge['binomial_p_value']:.6f}\n")
            f.write(f"- Significant at α=0.05: {'✅ Yes' if judge['significant_at_005'] else '❌ No'}\n")
            f.write(f"- Cohen's h: {judge['cohens_h']:.3f} ({judge['effect_size']})\n\n")
            f.write("---\n\n")

        if "inter_judge" in results:
            ij = results["inter_judge"]
            f.write("## Inter-Judge Agreement\n\n")
            f.write(f"- Cohen's κ: {ij['cohens_kappa']:.3f}\n")
            f.write(f"- Agreement level: {ij['agreement_level']}\n")
            f.write(f"- Papers compared: {ij['n_papers_compared']}\n")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


def main():
    results = run_tests()
    write_results(results)


if __name__ == "__main__":
    main()
