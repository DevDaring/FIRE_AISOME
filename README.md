# SETU — AISoMe 2026 stance detection (Hindi & Bengali YouTube comments)

**Track:** [AISoMe 2026 @ FIRE 2026](https://aisome2026.my.canva.site/), ISI Kolkata, 17–20 Dec 2026
**Task:** label a YouTube comment's stance toward *"climate change and global warming is a
serious concern"* as **Favour / Against / None**
**Metric:** macro-F1. **The catch:** training data is English-only; the test set is Hindi + Bengali.

**SETU** — *सेतु* / *সেতু*, "bridge" in both target languages; **S**tance via
**E**vidence-**T**axonomy **U**nification. The design rationale and the full method are
written up in the CEUR working note for the track.

---

## Repository layout

```
README.md              this file
Codes/
  src/                 all pipeline code
  results/             submitted runs + validation report
  run_all.sh           stage driver
  requirements.txt
```

**Every command below is run from inside `Codes/`.** Paths in this README are relative to
that directory.

### Submitted runs

`Codes/results/` holds what was actually sent to the organisers:
`Nirnay_hindi.csv`, `Nirnay_bengali.csv` (columns `id,model1_label,model2_label,
model3_label`), the packaged `Nirnay_AISoMe2026_submission.zip`, and
`submission_report.json` with the format-validation output.

| Column | System | dev macro-F1 (hi / bn) |
|---|---|---|
| `model1` | DeBERTa-v3-large, English pivot, calibrated | 0.922 / 0.915 |
| `model2` | DeBERTa-v3-large, English-only pool | 0.886 / 0.877 |
| `model3` | XLM-R-large, native script | 0.836 / 0.812 |

Those figures are measured against held-out **LLM-judge-panel** labels, not human gold —
see the working note's limitations section.

---

## ⚠️ Deadlines & submission format

| Date | Item |
|---|---|
| **31 July 2026** | **Run submission deadline** (extended from 30 July) |
| 31 Aug | Track results declared |
| 20 Sep | Working notes due (CEUR — **required to win**) |
| 31 Oct | Camera-ready working notes |

Winners are the **top 2 teams**, decided on *(a)* best-run macro-F1, *(b)* working-notes
quality, and *(c)* **novelty of the approach**.

**The submission format is not one-file-per-run.** From the organizers' email:

```
id,model1_label,model2_label,model3_label
1,Favour,Favour,None
2,None,None,None
```

* **one file per language** — Hindi (500 comments) and Bengali (500) **separately**
* `.csv` or `.xlsx`, both files in a **single ZIP**
* max **3 classifiers**, side by side as three columns of the same file
* the organizers may ask for the classifier + a README to reproduce a run

`src/make_submission.py` produces exactly this and refuses to write a ZIP unless the ids,
row counts and label strings all validate.

### Test data

Put the organizers' two files in **`Dataset/Testing_Data/`** (`.csv`/`.tsv`/`.txt`/`.xlsx`).
Files named `*hindi*` / `*bengali*` are picked up automatically; otherwise the dominant
script is sniffed. They are emailed to registered teams — if they have not arrived, chase
**aisome.fire2026@gmail.com**.

---

## Setup

The ML stack on this machine lives in **`python3.12`** (plain `python3` is 3.10 and has
nothing installed).

```bash
cd Codes
python3.12 -m pip install -r requirements.txt
python3.12 src/llm.py          # verify API keys — should print 5 committee members
```

`.env` (never committed) holds several keys per provider — `OPENROUTER_API_KEY_1..2`,
`DEEPSEEK_API_KEY_1..2`. `common.env_keys()` rotates across all of them, and evicts a key
for the rest of the process once a provider reports it dead. The final pipeline uses
DeepSeek and paid OpenRouter models only.

---

## Running it

```bash
bash run_all.sh stage0     # no test data needed — START HERE (long-pole API work)
bash run_all.sh stage1     # normalise the test files + run the LLM committee
bash run_all.sh annotate   # draw the dev sheet, then hand-label it (see below)
bash run_all.sh stage2     # train + self-train the argument-aware encoder
bash run_all.sh stage3     # NLI channel + calibration + fusion + scoreboard
bash run_all.sh submit     # build and validate the submission ZIP
```

Knobs are environment variables: `BACKBONE`, `EPOCHS`, `BATCH`, `AUX_WEIGHT`, `SOFT_ALPHA`,
`PER_CELL`, `GENERATORS`, `MEMBERS`, `SELFTRAIN_ROUNDS`, `TRANSLIT`, `TEAM`.

### 🔴 The one manual step that matters most

There is **no labelled Hindi/Bengali validation data anywhere**. Without a dev set we
cannot calibrate, cannot tune fusion weights, and cannot choose between the three runs —
we would be shipping guesses.

```bash
bash run_all.sh annotate                    # writes artifacts/dev_to_annotate.csv
# open it, read artifacts/dev_to_annotate_CODEBOOK.txt, fill `gold` (Favour|Against|None)
python3.12 src/annotate_dev.py finalise --input artifacts/dev_to_annotate.csv \
    --out artifacts/dev_gold.csv
```

~150 comments, roughly two hours, sampled where the LLM committee *disagrees* so the
effort lands on the decision boundary. It is worth more than any additional model, and it
gives the working notes a real human-agreement number.

---

## Architecture

Three heterogeneous channels, calibrated then fused. Errors decorrelate because the
channels share neither training data nor priors.

```
  Dataset/Training_Data/              GWSD (2300, 8 crowd votes) + SemEval-2016 CC (395)
        │                                          ↓ 2442 rows, only 15 % Against
        ├─ prepare_data.py ──────────────→ train_en.csv  (+ crowd soft labels)
        ├─ translate.py ─────────────────→ train_en.to-{hi,bn}.csv   (batched, 20/call)
        └─ synth_generate.py ────────────→ synth_train.csv
             └─ taxonomy.py: 25 argument nodes × {hi,bn} × {native,roman} × 5 registers
                             ↳ class-balanced BY CONSTRUCTION, India-specific arguments

  Dataset/Testing_Data/  ──normalize.py──→ test_{hi,bn}.csv   (script routing, romanised
                                                               Indic detection, emoji hints)
        │
   ┌────┴─────────────────────────────────────────────────────────────┐
   │ A  train_transformer.py + selftrain.py                           │  probs_setu_*.csv
   │    MuRIL/XLM-R, stance head + argument-node auxiliary head,      │
   │    soft-label KL, committee distillation, self-training          │
   │ B  llm_committee.py — 5 LLMs, argument-before-stance, in-language│  committee_*.csv
   │ C  nli_zeroshot.py  — claim-conditioned mDeBERTa-XNLI, no training│ probs_nli_*.csv
   └────┬─────────────────────────────────────────────────────────────┘
        ├─ calibrate.py   temperature → EM prior shift → per-class weights (greedy)
        ├─ fuse.py        weighted geometric mean, weights searched on dev
        └─ make_submission.py → validated ZIP
```

### The three runs (an ablation triple by design)

| Slot | System | Why it is in the submission |
|---|---|---|
| `model1` | **SETU-Full** — calibrated fusion of A + B + C | flagship, expected best |
| `model2` | **SETU-Encoder** — A alone, calibrated, *no API at inference* | the reproducible artefact the organizers can re-run |
| `model3` | **SETU-Committee** — B alone | LLM-reasoning reference point |

Only the **best** of the three counts for ranking, so this costs nothing on the
leaderboard and hands the working notes its headline table.

---

## Files

| File | Role |
|---|---|
| `results/` | the three submitted runs, the packaged ZIP, and the validation report |
| `src/common.py` | paths, labels, `.env` key rotation, metrics, **`read_csv`** |
| `src/taxonomy.py` | **27-node culturally-localised climate-argument taxonomy** + codebook |
| `src/llm.py` | multi-provider client: key rotation, disk cache, concurrency, JSON coercion |
| `src/normalize.py` | script detection, romanised-Indic routing, transliteration, cleaning |
| `src/prepare_data.py` | GWSD + SemEval → English pool with crowd soft labels |
| `src/translate.py` | batched stance-preserving MT (+ offline IndicTrans2 fallback) |
| `src/synth_generate.py` | taxonomy-conditioned synthetic Indic corpus |
| `src/llm_committee.py` | transductive committee, Fleiss' κ, soft labels, disagreement ranking |
| `src/nli_zeroshot.py` | training-free claim-conditioned NLI channel |
| `src/train_transformer.py` | multi-task encoder (stance + argument node, soft-label KL) |
| `src/train_baseline.py` | LaBSE + LogReg reference baseline |
| `src/selftrain.py` | agreement-weighted self-training with per-class quotas + committee veto |
| `src/predict.py` | inference → probability frame (never hard labels) |
| `src/calibrate.py` | temperature / EM prior shift / class weights, **greedily selected** |
| `src/fuse.py` | geometric soft fusion, dev-searched weights |
| `src/ensemble.py` | hard majority vote — **ablation baseline only** |
| `src/annotate_dev.py` | disagreement-stratified dev sampling + finalisation |
| `src/evaluate.py` | scoreboard + error analysis dump |
| `src/make_submission.py` | official-format ZIP with hard validation |

---

## Two traps worth knowing about

**1. `"None"` is a label, and pandas eats it.** One of the three class labels is the
literal string `None`, which `pandas.read_csv` treats as a missing-value marker. Reading a
labelled CSV normally turns every `None` row into `NaN` and silently drops **1092 of 2442**
training rows — it looks like a two-class problem, macro-F1 caps near 0.45, and nothing
errors. Every CSV read in this repo goes through `common.read_csv()`, which passes
`keep_default_na=False, na_values=[""]`. Do not reintroduce a bare `pd.read_csv`.

**2. `argmax` does not maximise macro-F1.** With `Against` at ~10–15 %, the metric-optimal
decision rule is argmax over *reweighted* probabilities. On a deliberately
Against-suppressing model this was worth **+0.22 macro-F1** in testing — which is why
`predict.py` emits distributions rather than labels, and why `calibrate.py` exists.
Calibration stages are accepted greedily (kept only if dev macro-F1 does not drop), because
EM prior correction can misfire badly and drive a minority class to zero.

---

## Timings (CPU, 32 cores — no local GPU)

| Step | Cost |
|---|---|
| `prepare_data.py` | seconds |
| `synth_generate.py` (per-cell 6 ≈ 1 500 rows) | ~15–30 min of API calls, cached |
| `translate.py` (2 442 × 2 languages, batched) | ~10–20 min, cached |
| `llm_committee.py` (5 members × 1 000 comments) | ~20–40 min, cached |
| `nli_zeroshot.py` (mDeBERTa, 1 000 comments × 3 hypotheses) | ~10–15 min |
| `train_transformer.py` (MuRIL, ~10 k rows, 3 epochs) | ~60–120 min |
| `selftrain.py` (3 rounds) | ~3–6 h |

Every LLM call is cached in `artifacts/cache/llm_*.jsonl`, so interrupting and resuming is
free. A `VAST_AI_API_KEY` is in `.env` if a rented GPU becomes worthwhile — on a T4
everything above is ~10× faster.

### Smoke test before committing to a full run

```bash
python3.12 src/taxonomy.py                                    # 27 nodes, 228 cue phrases
python3.12 src/llm.py                                         # provider connectivity
python3.12 src/synth_generate.py --dry-run                    # cell plan, zero API calls
python3.12 src/synth_generate.py --per-cell 2 --nodes B1_climate_colonialism \
    --out artifacts/synth_smoke.csv                           # one cell, real call
python3.12 src/train_transformer.py --train artifacts/train_en.csv \
    --model google/muril-base-cased --out /tmp/smoke --max-train 200 --epochs 1
```

---

## Data sources

* **GWStance / GWSD** — [Luo et al., Findings of EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.296/) → `Dataset/Training_Data/GWSD.tsv` ✓
* **SemEval-2016 Task 6**, target *"Climate Change is a Real Concern"* only (tweet IDs 614–1008) → `Dataset/Training_Data/semeval2016-task6-trainingdata.txt` ✓
* **CARDS taxonomy** of contrarian climate claims — Coan et al., *Scientific Reports* 11:22320 (2021); extended with an India-specific branch in `src/taxonomy.py`
* Methodology reference suggested by the organizers: **Das 2025**, [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S2212420925004972) (paywalled — worth reading before the working notes are written)
