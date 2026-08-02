# Working notes outline — CEUR, due 20 September 2026

Working-notes quality is an explicit **one third** of the winning decision (alongside
best-run macro-F1 and novelty), and submitting them is a *precondition* for being declared
a winner at all. This is not a formality to write the night before.

**Venue:** CEUR-WS, FIRE 2026 proceedings. Use the CEUR-ART single-column template
(`ceurart.cls`). Typical FIRE working note: 6–10 pages.

**Draft title**

> SETU: Bridging English Climate-Stance Resources to Indic YouTube Comments via a
> Culturally-Localised Argument Taxonomy and Transductive Committee Distillation

---

## The framing decision

Lead with the **diagnosis**, not the pipeline. A working note that says "we fine-tuned
XLM-R on translated data and ensembled three runs" is indistinguishable from every other
submission. A working note that says:

> *Cross-lingual stance transfer for climate discourse fails less because of language
> mismatch than because of **argument-inventory mismatch**: the permitted English corpora
> encode a US contrarian inventory (hoax, Al Gore, liberal agenda), while Indian YouTube
> commenters reject the same claim through an entirely different inventory (the West
> polluted first, development before environment, the monsoon was always erratic, yuga
> cycles, TRP drama). Translating the English data yields fluent Hindi sentences about Al
> Gore.*

…states a **falsifiable claim about the task**, and everything else becomes evidence for
it. That is what makes a shared-task paper citable rather than archival.

---

## Section plan

### 1. Introduction
Task, the English-train/Indic-test asymmetry, the three simultaneous shifts (language,
domain/genre, argument inventory). State the diagnosis and the four contributions.
Close with headline results.

### 2. Task and data
- Target claim, three labels, 500 + 500 test comments.
- Permitted training pool: GWSD (2 300 sentences × 8 crowd votes) + SemEval-2016 Task 6
  target *Climate Change is a Real Concern* (395 tweets, `FAVOR 212 / NONE 168 / AGAINST 15`).
- Report the pooled statistics from `artifacts/train_en.stats.json`: **2 442 rows after
  dedup, only 15.0 % `Against`**, mean GWSD crowd agreement 0.72.
- **Make the metric argument explicitly**: under macro-F1 the ~15 % `Against` class carries
  a full third of the score, so a system that collapses it is capped near 0.45 regardless
  of how well it does elsewhere. This paragraph justifies half the paper.

### 3. The SETU argument taxonomy *(contribution C1 — the most citable part)*
- Extension of CARDS (Coan et al., *Sci. Rep.* 2021) with an India-specific contrarian
  branch, a mirrored pro-climate branch grounded in Indian lived experience, and an
  explicit `None` inventory.
- **Include the full taxonomy as a table**: 25 nodes — 12 `Against` (5 CARDS + 7 India),
  7 `Favour`, 6 `None`; 204 cue phrases across en/hi/bn. Generate from `src/taxonomy.py`.
- Argue why it is a *resource*, not an appendix: it is used three ways (generative schema,
  auxiliary supervision, committee reasoning scaffold).
- Discuss the two annotation traps that motivate the `None` inventory:
  *sentiment ≠ stance* ("shame on this government for the pollution" = **Favour**) and
  *video praise ≠ claim agreement* ("very informative video sir" = **None**).

### 4. Method
- **4.1 Script-aware preprocessing** — romanised code-mix prevalence in the test set
  (report the actual figure from `normalize.py`; it is a dataset observation worth
  publishing), script routing, transliteration.
- **4.2 Taxonomy-conditioned synthesis (C2)** — the cell product
  `node × lang × script × register`, diversity guards, dedup rate, final class balance.
  Report `artifacts/synth_train.stats.json`.
- **4.3 Argument-aware multi-task encoder (C3)** — MuRIL/XLM-R, stance head + argument-node
  head, soft-label KL over GWSD crowd votes, class-weighted loss. Say why class weighting
  rather than oversampling (oversampling distorts soft targets).
- **4.4 Transductive committee distillation (C4)** — 5 heterogeneous LLMs, argument-before-
  stance prompting, in-language prompts, agreement-weighted soft labels, self-training with
  per-class quotas and committee veto. **Be explicit that this is transduction on unlabelled
  test data, not label leakage.**
- **4.5 Claim-conditioned NLI channel** — training-free, error-decorrelated.
- **4.6 Macro-F1-optimal decisions** — temperature scaling, Saerens et al. EM prior
  correction, per-class weight search, greedy stage acceptance. Note the EM collapse
  failure mode and the floor/damping fix; this is an honest negative result worth a
  paragraph.

### 5. Experimental setup
Backbones, hyperparameters, CPU-only compute, seeds (`RANDOM_SEED=20260502`), the exact
`run_all.sh` stages. **Describe the hand-annotated dev set**: ~150 comments,
disagreement-stratified sampling, the codebook, and the human-vs-committee agreement number
from `dev_gold.stats.json`. Acknowledge its size as a limitation.

### 6. Results
- **Table 1** — the three submitted runs, macro-F1 and per-class F1, per language and pooled.
- **Table 2** — channel-wise dev results (`artifacts/eval_report_*.json`): encoder,
  encoder+calibration, committee, NLI, hard vote, fused. The hard-vote-vs-soft-fusion delta
  is worth its own sentence.
- **Table 3 — ablations.** Each row removes one thing:
  | ablation | what it tests |
  |---|---|
  | − taxonomy synthesis (translate-train only) | is C2 doing the work? |
  | − India-specific branch (CARDS only) | is *cultural localisation* the active ingredient, or just more data? |
  | − auxiliary argument head (`--aux-weight 0`) | does argument supervision transfer? |
  | − soft labels (`--soft-alpha 0`) | do crowd distributions help the minority class? |
  | − self-training rounds | value of transduction |
  | − calibration | value of metric-aware decisions |
  | XLM-R instead of MuRIL | does transliteration-aware pretraining matter? |

  The **− India-specific branch** row is the single most important experiment in the paper:
  it isolates the central claim. Run it even if time is short elsewhere.
- **Table 4** — committee behaviour: per-member coverage, pairwise agreement, Fleiss' κ,
  most frequent argument nodes per language. The node distribution is a genuine finding
  about Indian climate discourse, not just a diagnostic.

### 7. Error analysis
From `artifacts/errors_*.csv`. Organise by taxonomy node and by trap:
sentiment-vs-stance, video-praise, sarcasm, whataboutism, romanised code-mix, very short
comments. **Quote real examples with glosses** — reviewers remember examples.
Report which nodes the system systematically misses.

### 8. Limitations
Dev-set size; synthetic data is LLM-generated and may encode LLM stereotypes of Indian
discourse rather than the real thing (say this plainly — it is the honest caveat and
pre-empts the obvious criticism); committee members are closed models, hurting exact
reproducibility, which is why `model2` is API-free; single annotator on the dev set.

### 9. Conclusion & future work
Release the taxonomy. Propose a properly annotated Indic climate-stance corpus organised by
argument node.

---

## Reproducibility appendix

The organizers reserve the right to request the classifier plus a README. Ship:
- `Codes/README.md` (already written for this purpose),
- `artifacts/*/metrics.json`, `selftrain_history.json`, `calib_*.json`, `fuse_report_*.json`,
- `artifacts/submission/submission_report.json` — the run→slot mapping. **Record which
  system was `model1`/`model2`/`model3` on submission day**; three months later the
  mapping is not recoverable from memory, and the working notes need it.

---

## Pre-submission checklist (31 July 2026)

- [ ] Test data received and in `Dataset/Testing_Data/`
- [ ] `run_all.sh stage0` complete (synthetic corpus + translations cached)
- [ ] Committee run, `committee_{hi,bn}.csv` present, Fleiss' κ recorded
- [ ] **`dev_gold.csv` hand-annotated** (≥ 120 usable rows)
- [ ] Encoder trained; gold-dev macro-F1 beats the committee alone
- [ ] Calibration fitted per language; gains recorded (reject stages that do not help)
- [ ] Fusion weights searched on dev; fused beats every single channel
- [ ] `make_submission.py` run with **strict** validation (no `--no-strict`)
- [ ] ZIP opened and inspected: header exactly `id,model1_label,model2_label,model3_label`,
      500 rows per file, ids identical to the organizers' files in the original order
- [ ] No column predicts a single class for every comment
- [ ] `submission_report.json` archived with the run→slot mapping
- [ ] ZIP uploaded via the organizers' form **before 31 July**
