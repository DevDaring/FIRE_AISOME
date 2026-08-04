# Required actions — things only you can do

**Team Nirnay · AISoMe 2026 @ FIRE · updated 4 Aug 2026**

The modelling work is **finished**. Everything below is either yours to do or a
decision only you can make. Nothing is running; nothing is billing.

---

## 🔴 DO THIS FIRST — submit the runs

### 1. Upload the ZIP to the organizers

**File:** `Submission/runs/Nirnay_AISoMe2026_submission.zip`
(contains `Nirnay_hindi.csv` and `Nirnay_bengali.csv`)

Validated on every check: 500 rows per file, header exactly
`id,model1_label,model2_label,model3_label`, ids byte-identical to your `.xlsx`
files **in the original order**, no blanks, no single-class column.

**Deadline 7 August 2026.** Not submitting is the only guaranteed loss.

### 2. Confirm two things with the organizers (same email)

Email **aisome.fire2026@gmail.com**:

- **The exact deadline hour and timezone.** Still unconfirmed. The original
  date was 31 July; the test data arrived 1 August.
- **CSV or XLSX?** We produced CSV, which their email permits. If they want
  XLSX, tell me and I will regenerate — it is a one-flag change.

---

## 🟡 THE BIGGEST REMAINING RISK — and it is cheap to fix

### 3. Blind-label 50 comments (~2 hours)

Every score we report — **hi 0.922 / bn 0.915** — measures agreement with our
**LLM judge panel**, not with ground truth. The panel agrees with itself well
(Fleiss' κ 0.859) and you reviewed 15 labels and endorsed them, but that review
was **not blind**, so it yields no independent number.

This matters concretely. `None` is our weakest class (0.870 / 0.863) and it is
exactly where careful human annotators disagree — is *"very informative video,
thank you"* `None` or `Favour`? If the organizers' annotators drew that line
differently from our judges, the official score will be lower than reported, and
our calibration (fitted to the panel's class priors) may transfer poorly.

**What to do:** label 50 comments *before* looking at any prediction.

```bash
cd /home/Debz/Hackathon/AISOME/Codes
python3.12 src/annotate_dev.py sample --committee artifacts/committee_hi.csv \
    artifacts/committee_bn.csv --n 50 --out artifacts/blind50.csv
# fill the `gold` column, then:
python3.12 src/annotate_dev.py finalise --input artifacts/blind50.csv \
    --out artifacts/blind50_gold.csv
```

Then tell me and I will score the judges against it. Interpretation:

| Agreement with your 50 | What it means |
|---|---|
| ≥ 90 % | The reported figures are trustworthy. Strong position. |
| 75–90 % | Real but somewhat inflated. The paper must say so. |
| < 75 % | Recalibrate on your labels instead of the panel's. |

Either way the number belongs in the paper. *"We used LLM judges and measured
them against blind human annotation at 0.89"* is a contribution;
*"we used LLM judges"* is a reviewer's objection.

**This is the single highest-value two hours left in the project.**

---

## 🟢 THE WORKING NOTES — drafted, needs your details

**File:** `Submission/working_notes.tex` — a complete CEURART paper, due
**20 September 2026**. Submitting it (and presenting at FIRE in December) is a
*precondition* for being declared a winner, not optional.

### What you must fill in

1. **Co-authors.** I listed only you (IIIT Kalyani / Accenture). Add anyone else
   with their affiliation — see the `\author` block and the `%% TODO` markers.
2. **Your ORCID** in the `\author` options.
3. **The Das 2025 citation.** The organizers pointed at it as methodological
   guidance; it is paywalled on ScienceDirect and I could not read it. Fetch the
   full reference and, ideally, cite it properly in §2.
4. **Compile it.** Needs `ceurart.cls` and the Libertinus fonts:
   - Overleaf: search "CEURART" and paste the `.tex` in, or
   - download `CEURART.zip` from <https://ceur-ws.org/HOWTOSUBMIT.html>

### Read the Generative AI declaration before submitting

CEUR **mandates** a "Declaration on Generative AI". I have written a full and
honest one: LLMs generated the synthetic corpus, produced the distilled
pseudo-labels and development labels, and did the translation; and an AI
assistant helped with code and drafting. **Read it and confirm it matches how
you want your involvement described** — it is your name on the paper. Adjust the
wording freely; do not weaken the factual content, because the LLM role here is
substantial and reviewers will check it against the method section.

---

## 🔵 DECISIONS FOR YOU

### 4. GitHub repo is PRIVATE — flip it after results

I set `DevDaring/FIRE_AISOME` private because `STRATEGY.md` spells out the
test-set discovery and the competition was open. **Make it public once results
are declared (31 Aug)** — good practice for the CEUR paper, and the code is a
genuine artifact. Say the word and I will flip it.

### 5. HuggingFace repos are PRIVATE — publish with the paper

- `Debk/nirnay-aisome2026-setu` — both adopted checkpoints + model card
- `Debk/nirnay-aisome2026-data` — taxonomy, synthetic corpus, judge labels, dev split

The **taxonomy is the most citable thing here.** Publishing it alongside the
paper is how it gets used by others.

### 6. Two dead API keys — replace or delete

`GEMINI_API_KEY_2` and `Link_Gemini_Cheap_API_Key` both return *"API key not
valid"*. Harmless now (nothing uses Google any more, per your instruction), but
worth cleaning out of `.env`.

---

## ✅ DONE — no action needed

| Item | State |
|---|---|
| Submission built and validated | `Submission/runs/` — all checks pass |
| Best system | DeBERTa-v3-large on back-translated English, calibrated |
| Dev macro-F1 | **hi 0.922 / bn 0.915** (vs 0.810 / 0.779 pre-pivot) |
| GitHub | 20 commits, private, secrets and organizer data excluded |
| HuggingFace | models + dataset uploaded with cards |
| GPU | **destroyed — 0 instances, total spend $1.11**, $7.21 credit left |
| Background jobs | none running |
| Working notes | drafted at `Submission/working_notes.tex` |

### What is in the submission

| Column | System | dev hi | dev bn |
|---|---|---|---|
| `model1` | DeBERTa-v3-large, English pivot, calibrated | **0.922** | **0.915** |
| `model2` | DeBERTa-v3-large, English-only pool | 0.886 | 0.877 |
| `model3` | XLM-R-large, native script | 0.836 | 0.812 |

All three are classifiers we trained — no API call at inference. Note that
`model1` consumes the back-translated English view, so reproducing it needs the
shipped `test_hi_en.csv` / `test_bn_en.csv` (both in `Submission/runs/`) or a
re-translation step.

---

## If you ever hand this over

Everything reproduces from the repo. The one thing deliberately not in git is
`.env` — back it up somewhere safe and never commit it. If any key in it has
been pasted anywhere public, rotate it.
