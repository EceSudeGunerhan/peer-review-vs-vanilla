# Peer-Review Skill vs Vanilla: LLM-Generated Scientific Review Quality

This project compares automatically generated scientific reviews produced using a **peer-review skill** (defined in `SKILL.md`) with reviews generated using a **vanilla prompt** (without explicit skill injection).

Quality is evaluated using **LLM-as-a-Judge** with blind pairwise comparison.

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

This project uses the ICLR 2017 subset of the PeerRead dataset, which contains parsed paper texts and corresponding human-written reviews.

The evaluation is conducted only on paper–review pairs that satisfy predefined quality constraints:

- paper_text ≥ 1500 characters  
- ground_truth ≥ 200 characters  

The dataset is not included in this repository due to size and licensing considerations.  
Users must obtain the dataset independently and run the preprocessing scripts locally.

---

## Quality Filters

To ensure meaningful evaluation:

- `paper_text` must be at least **1500 characters**
- `ground_truth` must be at least **200 characters**

---

## Execution Pipeline

### 1. Build paper-review pairs

```bash
python scripts/01_build_pairs.py
```

Creates:

```
data/processed/pairs.jsonl
```

---

### 2. Create sample subset (default: 10 papers)

```bash
python scripts/01b_make_sample.py
```

Creates:

```
data/processed/sample_pairs.jsonl
```

---

### 3. Generate reviews (peer + vanilla)

```bash
python scripts/02_generate_reviews_dual.py
```

Creates:

```
outputs/generations/reviews_sample_peer.jsonl
outputs/generations/reviews_sample_vanilla.jsonl
```

---

### 4. Blind A/B pairwise judging

```bash
python scripts/03_judge_pairwise_ab.py
```

Creates:

```
outputs/judgments/judgments_pairwise_sample.jsonl
```

---

### 5. Compute win-rate summary

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

## Output Format

### `judgments_pairwise_sample.jsonl`

Each line contains:

```json
{
  "paper_id": "762",
  "cond_A": "peer",
  "cond_B": "vanilla",
  "winner": "A",
  "reasoning": "Review A better matches the ground truth..."
}
```

- `winner`: `"A"`, `"B"`, or `"tie"`
- `cond_A` / `cond_B`: Randomly assigned for blind comparison

---

### `pairwise_summary.json`

```json
{
  "num_examples": 10,
  "peer_wins": 6,
  "vanilla_wins": 2,
  "ties": 2,
  "peer_win_rate_total": 0.6,
  "vanilla_win_rate_total": 0.2,
  "tie_rate_total": 0.2,
  "peer_win_rate_non_tie": 0.75,
  "vanilla_win_rate_non_tie": 0.25
}
```

---

## Example Interpretation (10-Sample Run)

| Metric | Value |
|--------|-------|
| Total examples | 10 |
| Peer wins | 6 |
| Vanilla wins | 2 |
| Ties | 2 |
| Peer win rate (total) | 60% |
| Vanilla win rate (total) | 20% |
| Tie rate | 20% |
| Peer win rate (excluding ties) | 75% |
| Vanilla win rate (excluding ties) | 25% |

Interpretation:

- The peer-review skill produces outputs that are closer to the ground truth compared to the vanilla prompt.
- A 10-example sample is small; increase `SAMPLE_SIZE` in `src/config.py` for more reliable results.

---

## Configuration

Defined in `src/config.py`:

- `SAMPLE_SIZE`
- `GEN_MODEL_NAME`
- `JUDGE_MODEL_NAME`
- `GEN_TEMPERATURE`
- `JUDGE_TEMPERATURE`
- `GEN_MAX_OUTPUT_TOKENS`
- `JUDGE_MAX_OUTPUT_TOKENS`

