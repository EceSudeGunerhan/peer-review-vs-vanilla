# Results and Discussion

## Experimental Setup (Summary)

We compare two conditions for generating peer reviews of machine learning conference papers:

- **Vanilla:** A capable LLM (GPT-5.2) with a standard prompt asking for a thorough review in a fixed 7-section format (summary, strengths, weaknesses, major/minor comments, questions, recommendation).
- **Peer:** The same model and format, with an ML-specific peer-review skill injected. The skill provides recommendation calibration (avoid over-rejection, match human consensus direction), ML evaluation criteria (novelty, ablations, reproducibility, related work), tone guidance (conversational, collegial), and parsing-aware advice (don’t penalize missing tables from PDF extraction).

The prompts are aligned so that **peer = vanilla + skill**; the only difference is the skill content.

**Dataset:** ICLR 2017 papers with ground-truth human reviews. Each paper has one peer and one vanilla machine-generated review.

**Evaluation:** Pairwise LLM-as-a-Judge. Two judges (Claude Opus 4.6, GPT-5.2) independently compare each peer–vanilla pair and select which review is closer to the human ground truth. The judge prompt defines five dimensions (Content Coverage, Critical Depth, Specificity, Alignment with Ground Truth, Actionability) and treats **Alignment with Ground Truth** as primary, especially recommendation direction (accept vs reject).

---

## Results

### Primary Outcomes: Win Rates

| Judge | Model | Peer Wins | Vanilla Wins | Peer Win Rate | 95% CI | Binomial p | Cohen's h |
|-------|-------|-----------|--------------|---------------|--------|------------|-----------|
| Primary | Claude Opus 4.6 | 229 | 119 | **65.8%** | [0.607, 0.706] | <0.001 | 0.32 (small) |
| Secondary | GPT-5.2 | 199 | 150 | **57.0%** | [0.518, 0.621] | 0.010 | 0.14 (negligible) |

Both judges favor peer over vanilla. The effect is statistically significant at α = 0.05 for both judges (binomial test against a null of 50% peer win rate).

- **Claude (primary):** Peer wins 65.8% of comparisons (229/348). Effect size is small (Cohen's h ≈ 0.32).
- **GPT (secondary):** Peer wins 57.0% of comparisons (199/349). Effect size is negligible (Cohen's h ≈ 0.14).

Claude’s preference for peer is stronger than GPT’s, but both point in the same direction: the skill improves alignment with human reviews.

### Inter-Judge Agreement

- **Cohen's κ = 0.57** (moderate agreement, Landis–Koch scale)
- **Agreement:** 276 / 348 papers (79.3%)
- **Disagreement:** 72 / 348 papers (20.7%)

When judges agree:
- Both pick peer: 178 papers (51.2%)
- Both pick vanilla: 98 papers (28.2%)

When judges disagree:
- Claude picks peer, GPT picks vanilla: 51 papers
- Claude picks vanilla, GPT picks peer: 21 papers

In disagreements, Claude favors peer more often than GPT (51 vs 21). This suggests Claude is more sensitive to the skill’s calibration and tone, or that GPT as judge may have a slight bias toward its own (vanilla) style when it is also the generator.

### Recommendation Direction as Primary Differentiator

Across sampled judgments, the main deciding factor is **Dimension 4: Alignment with Ground Truth**, especially recommendation direction. Representative reasoning:

> *"The ground-truth human reviews are clearly positive about the paper... Review A recommends 'Weak Accept,' which aligns with this positive consensus, while Review B recommends 'Weak Reject,' which contradicts the human reviewers' overall positive assessment... the primary dimension—alignment with ground truth recommendation direction—clearly favors Review A."*

The skill’s calibration guidance (avoid over-rejection, match human consensus direction) appears to reduce cases where the model recommends Weak Reject when human reviewers lean accept. Vanilla reviews tend to be more reject-leaning; peer reviews more often match the human consensus.

---

## Discussion

### Does the Skill Add Value?

Yes. With peer = vanilla + skill, peer reviews are preferred over vanilla by both judges. The skill adds value beyond a capable model with the same format.

The skill contributes:

1. **Recommendation calibration:** Guidance to avoid systematic over-rejection and to align with human consensus direction. This directly targets the primary judge dimension (Alignment).
2. **ML-specific criteria:** Novelty, ablations, reproducibility, related work—the kinds of concerns human ML reviewers typically raise.
3. **Tone and style:** Conversational, collegial phrasing that matches ICLR-style reviews.
4. **Parsing-aware advice:** Not penalizing missing tables/appendices when they are absent from extracted text.

The strongest signal is recommendation direction: peer reviews more often match the accept/reject stance of human reviewers.

### Judge Differences: Claude vs GPT

Claude favors peer more strongly (65.8%) than GPT (57.0%). Possible explanations:

- **Different sensitivity to calibration:** Claude may weight the skill’s calibration guidance more heavily when judging alignment.
- **Self-preference (GPT):** GPT generates both reviews and judges them. It may slightly favor its default (vanilla) style when judging.
- **Interpretation of criteria:** The two judges may apply the five dimensions differently, even with the same prompt.

The primary judge (Claude) is from a different model family than the generator (GPT), which reduces generator–judge self-preference. The secondary judge (GPT) tests whether the effect holds when judge and generator share a family.

### Effect Sizes: Modest but Consistent

Effect sizes are modest: small for Claude (h ≈ 0.32), negligible for GPT (h ≈ 0.14). The skill does not produce a large shift, but the effect is:

- **Consistent** across both judges
- **Statistically significant** for both
- **Directionally aligned** with the skill’s design (better alignment with human consensus)

For deployment, this suggests the skill is a meaningful improvement but not a dramatic one. Further gains may require additional skill refinements or different evaluation setups.

### Inter-Judge Reliability

κ = 0.57 indicates moderate agreement. About 20% of papers receive different judgments across judges. This is typical for subjective evaluation tasks and suggests:

- The evaluation is reasonably reliable but not perfect.
- Disagreements are informative: they highlight edge cases where alignment is ambiguous.
- Using multiple judges and reporting both individual and aggregate results is appropriate.

### Implications for Skill Design

The results support several design choices in the skill:

- **Recommendation calibration** is central; it targets the main failure mode (over-rejection) and the primary judge dimension.
- **ML-specific criteria** help focus reviews on what human ML reviewers care about.
- **Tone guidance** may improve perceived alignment, though this is harder to isolate.
- **Parsing-aware advice** likely reduces spurious criticism of missing content.

Future work could test ablations (e.g., skill without calibration, or without ML criteria) to quantify the contribution of each component.

---

## Conclusion

We compared peer reviews generated with an ML-specific skill to vanilla reviews from the same model and format. Both judges (Claude and GPT) prefer peer over vanilla, with statistically significant effects. The primary differentiator is alignment with human recommendation direction: the skill’s calibration guidance reduces over-rejection and improves match with human consensus. Effect sizes are modest but consistent. Inter-judge agreement is moderate (κ = 0.57). The results indicate that the skill adds value for ML conference peer review, with recommendation calibration as the most impactful component.
