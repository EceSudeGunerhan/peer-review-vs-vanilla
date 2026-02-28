# Peer-Review Skill vs Vanilla: LLM-Generated Scientific Review Quality

This project compares automatically generated scientific reviews produced using a **peer-review skill** (defined in `SKILL.md`) with reviews generated using a **vanilla prompt** (without explicit skill injection).

Quality is evaluated using **LLM-as-a-Judge** with blind pairwise comparison.

The pipeline runs on the **full ICLR 2017 subset** of PeerRead (all paper–review pairs that pass quality filters).

---

## Research Question

> Does the use of a peer-review skill improve the quality of scientific reviews compared to a vanilla prompt?

For each paper, an LLM judge answers:

> Which review is closer to the ground truth — the skill-based output or the vanilla output?

Results are reported as:

- **Peer win rate (%)**
- **Vanilla win rate (%)**
- **Tie rate (%)**

---

## Requirements

- Python 3.10+
- `requests` (for OpenRouter API access)

```bash
pip install -r requirements.txt
```

---

## Environment Variables

An OpenRouter API key is required:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

---

## Dataset

This project uses the **ICLR 2017 subset of the PeerRead dataset**, containing parsed paper texts and corresponding human-written reviews.

The evaluation is conducted on **all** paper–review pairs that satisfy predefined quality constraints:

- `paper_text` ≥ 1500 characters  
- `ground_truth` ≥ 200 characters  

The dataset is not included in this repository due to size and licensing considerations.  
Users must obtain the dataset independently and run the preprocessing scripts locally.

---

## Execution Pipeline (Full Dataset)

### 1. Build paper–review pairs

```bash
python scripts/01_build_pairs.py
```

Creates:

```
data/processed/pairs.jsonl
```

This file contains all paper–review pairs that pass the quality filters (316 papers in the final experiment).

---

### 2. Generate reviews (peer + vanilla)

```bash
python scripts/02_generate_reviews_dual.py
```

Creates:

```
outputs/generations/reviews_peer.jsonl
outputs/generations/reviews_vanilla.jsonl
```

Each paper in `pairs.jsonl` is processed. Paper text is truncated to 8000 characters (head + tail) for generation.

---

### 3. Blind A/B pairwise judging

```bash
python scripts/03_judge_pairwise_ab.py
```

Creates:

```
outputs/judgments/judgments_pairwise.jsonl
```

For each paper, the judge receives:

- Paper text  
- Ground-truth review  
- Two generated reviews (A and B)

Assignment of peer/vanilla to A/B is randomized per paper (blind comparison).

---

### 4. Compute win-rate summary

```bash
python scripts/04_summarize_pairwise.py
```

Creates:

```
outputs/reports/pairwise_summary.json
outputs/reports/pairwise_summary.csv
outputs/reports/pairwise_summary.md
```

---

# Experimental Results (Full Dataset, n = 316)

## Pairwise Summary (JSON)

```json
{
  "num_examples": 316,
  "peer_wins": 203,
  "vanilla_wins": 109,
  "ties": 4,
  "peer_win_rate_total": 0.6424050632911392,
  "vanilla_win_rate_total": 0.3449367088607595,
  "tie_rate_total": 0.012658227848101266,
  "peer_win_rate_non_tie": 0.6506410256410257,
  "vanilla_win_rate_non_tie": 0.34935897435897434
}
```

---

## Pairwise Summary (CSV)

```csv
condition,wins,win_rate_total,win_rate_non_tie
peer,203,0.6424050632911392,0.6506410256410257
vanilla,109,0.3449367088607595,0.34935897435897434
tie,4,0.012658227848101266,
```

---

## Human-Readable Summary

```
Total examples: 316
Peer wins: 203
Vanilla wins: 109
Ties: 4

Win rates (over all examples):
- Peer: 0.642
- Vanilla: 0.345
- Tie: 0.013

Win rates (excluding ties):
- Peer: 0.651
- Vanilla: 0.349
```

---

## Interpretation

- The peer-review skill outperforms the vanilla prompt on the full dataset.
- Peer wins in **64.2%** of all comparisons.
- Excluding ties, peer wins **65.1%** of the time.
- Tie rate is very low (**1.3%**), indicating strong separation between conditions.

These results suggest that structured peer-review prompting improves alignment with human-written ground-truth reviews.

---

## Example Outputs

### Example Judgment Entry

```json
{
  "paper_id": "305",
  "cond_A": "vanilla",
  "cond_B": "peer",
  "winner": "B",
  "reasoning": "Review B more closely matches the ground-truth review's style and content coverage..."
}
```

### Example Peer Review Output (Truncated)

```json
{
  "paper_id": "305",
  "condition": "peer",
  "model": "openai/gpt-4o",
  "truncation": "head_tail"
}
```

### Example Vanilla Review Output (Truncated)

```json
{
  "paper_id": "305",
  "condition": "vanilla",
  "model": "openai/gpt-4o",
  "truncation": "head_tail"
}
```

---

## Configuration

Defined in `src/config.py`:

| Parameter | Default |
|------------|----------|
| `GEN_MODEL_NAME` | `openai/gpt-4o` |
| `JUDGE_MODEL_NAME` | `anthropic/claude-3.5-sonnet` |
| `GEN_TEMPERATURE` | 0.3 |
| `JUDGE_TEMPERATURE` | 0.0 |
| `GEN_MAX_OUTPUT_TOKENS` | 1500 |
| `JUDGE_MAX_OUTPUT_TOKENS` | 800 |
| `JUDGE_PAPER_MAX_CHARS` | 8000 |
| `JUDGE_GT_MAX_CHARS` | 6000 |
| `RANDOM_SEED` | 42 |

