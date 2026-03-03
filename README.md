# Peer-Review Skill vs Vanilla: LLM-Generated Scientific Review Quality

This project investigates whether injecting a **peer-review skill** (`peer-review-skills/REVIEW_SKILL_ML.md`) into an LLM improves the quality of automated scientific reviews compared to a **vanilla prompt** (same output format, no skill injection).

Quality is measured using **LLM-as-a-Judge pairwise comparison** against real ICLR 2017 human reviews, with a **multi-judge design** (two independent judges) and **statistical significance testing**.

---

## Research Question

> Does the use of a peer-review skill improve the quality of scientific reviews compared to a vanilla prompt?

---

## Quick Start

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."

python run_experiment.py          # Run full pipeline (auto-resumes)
python run_experiment.py --status # Check progress
```

---

## Experiment Design

### Two Conditions (Same 7-Section Output Format)

| | Peer-Review Skill | Vanilla Baseline |
|---|---|---|
| **Prompt** | Paper + `REVIEW_SKILL_ML.md` criteria injected | Paper only |
| **Output format** | 7 sections (Summary → Recommendation) | 7 sections (identical) |
| **Max output tokens** | 5000 | 3000 |
| **Only difference** | ML conference skill knowledge injected | No skill knowledge |

Both conditions use `openai/gpt-5.2` as the generator with temperature 0.4. Invalid responses (e.g. model claims "text was missing") are retried up to 3 times before recording an error.

### Multi-Judge Design

| Role | Model |
|------|-------|
| Generator | `openai/gpt-5.2` |
| Judge 1 (primary) | `anthropic/claude-opus-4.6` |
| Judge 2 (secondary) | `openai/gpt-5.2` |

- **Judge 1** is from a different model family → no self-preference bias
- **Judge 2** is from the same family → tests for self-preference bias
- **Cohen's κ** measures inter-judge agreement

### Judge Evaluation Criteria (5 Dimensions)

**Primary criterion:** Alignment with ground truth (dimension 4). "Closer" means overlap of main criticisms/strengths with the human review, similar significance assessment, and similar recommendation direction. Format differences are ignored.

1. **Content Coverage** — Does the review address the paper's key contributions?
2. **Critical Depth** — Does it identify strengths AND weaknesses with reasoning?
3. **Specificity** — Does it reference specific paper elements (methods, equations)?
4. **Alignment with Ground Truth** — How closely do criticisms, strengths, and methodological comments match the human review? (PRIMARY)
5. **Actionability** — Are suggestions constructive and revision-worthy?

**Forced choice (win/loss only):** No ties anywhere. The judge must pick A or B. Invalid/tie responses are retried up to 3 times; if still invalid, that paper is skipped. Summaries and statistical tests use only win/loss outcomes. Multi-reviewer ground truth: prefer the machine review that best matches the overall consensus.

### Statistical Tests

- **Binomial test** (H₀: peer win rate = 0.5)
- **95% Wilson CI** for peer win rate
- **Cohen's h** for effect size
- **Cohen's κ** for inter-judge agreement

---

## Pipeline (4 Steps)

```
python run_experiment.py
```

| Step | Script | Output |
|------|--------|--------|
| 0* | Pre-experiment: build pairs | `data/processed/pairs.jsonl` |
| 1 | Generate reviews (peer + vanilla) | `outputs/generations/reviews_{peer,vanilla}.jsonl` |
| 2 | Judge — Claude (primary) | `outputs/judgments/judgments_pairwise_claude.jsonl` |
| 3 | Judge — GPT (secondary) | `outputs/judgments/judgments_pairwise_gpt.jsonl` |
| 4 | Summarize + statistical tests | `outputs/reports/` |

*Step 0 runs automatically only if `pairs.jsonl` is missing.

All steps support **resume** — on restart, already-processed papers are skipped. Ctrl+C is safe. Generation retries invalid responses up to 3× per paper; error rows are cleaned at the end so failed papers are retried on the next run.

---

## Dataset

**ICLR 2017 subset of PeerRead** — 349 matched paper–review pairs.

- Papers: `data/raw/iclr_2017/parsed_pdfs/*.pdf.json`
- Reviews: `data/raw/iclr_2017/reviews/*.json`
- Quality filters: `paper_text ≥ 1500 chars`, `ground_truth ≥ 200 chars`
- Ground truth: actual peer reviews only (meta-reviews and committee decisions filtered out)
- **Full paper text sent** — no truncation (all papers fit within model context windows)

---

## Configuration

| Parameter | Value |
|------------|--------|
| `GEN_MODEL_NAME` | `openai/gpt-5.2` |
| `JUDGE_MODEL_NAME` | `anthropic/claude-opus-4.6` |
| `JUDGE_MODEL_NAME_2` | `openai/gpt-5.2` |
| `GEN_TEMPERATURE` | 0.4 |
| `JUDGE_TEMPERATURE` | 0.0 |
| `GEN_MAX_OUTPUT_TOKENS` | 3000 (vanilla) |
| `GEN_MAX_OUTPUT_TOKENS_PEER` | 5000 (peer) |
| `JUDGE_MAX_OUTPUT_TOKENS` | 1000 |
| `GEN_MAX_RETRIES_INVALID` | 3 (retries when model claims text missing) |
| `JUDGE_MAX_RETRIES_TIE` | 2 (retries when judge returns tie; forced A/B choice) |
| `RANDOM_SEED` | 42 |
| Paper truncation | None (full text) |
| A/B assignment | Deterministic per paper (hashlib.md5) |

---

## Project Structure

```
peer-review-vs-vanilla/
├── run_experiment.py               # Master experiment runner
├── peer-review-skills/
│   └── REVIEW_SKILL_ML.md          # ML conference skill (used for peer condition)
├── prompts/
│   ├── peer_review_generation.txt  # Skill-injected prompt
│   ├── vanilla_review.txt          # Fair baseline (same 7-section format)
│   └── judge_pairwise_ab.txt       # Multi-criteria judge prompt
├── scripts/
│   ├── 01_build_pairs.py
│   ├── 02_generate_reviews_dual.py
│   ├── 03_judge_pairwise_ab.py
│   ├── 03b_judge_pairwise_ab_secondary.py
│   ├── 04_summarize_pairwise.py
│   └── 05_statistical_tests.py
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── data_prep/build_pairs.py
│   ├── generation/
│   │   ├── generate_reviews_dual.py
│   │   └── llm_client.py
│   ├── judging/judge_pairwise_ab.py
│   └── reports/
│       ├── summarize_pairwise.py
│       └── statistical_tests.py
├── data/
│   ├── raw/iclr_2017/
│   └── processed/pairs.jsonl
└── outputs/
    ├── generations/
    ├── judgments/
    └── reports/
```
