# Required actions — things only you can do

**Team Nirnay · AISoMe 2026 @ FIRE · updated 2 Sep 2026**

**Runs are submitted.** The organisers have confirmed receipt and report **more
than 80 runs** across the track, with results delayed while they evaluate. The
modelling work is finished and nothing is running or billing.

The remaining work is the **working note, due 20 September** — and one
measurement that is worth more than anything else left.

---

## 🔴 BLOCKING — paste the official test scores into the paper

The organisers reviewed the working note and asked for **the evaluation result
on test data**. The paper now has a section for exactly that
(§8.2 *Results on the evaluation data*, Table 5), and everything in it that I
could compute myself is already filled in. What I **cannot** supply is the
official macro-F1, because those numbers exist only in the organisers' results
e-mail — I will not invent them.

**Send me the scores and I will fill them in**, or edit them yourself at the
top of `Submission/AISOME.tex`:

```latex
\newcommand{\offOneHi}{0.71}   % model1 (English pivot, calibrated), Hindi
\newcommand{\offOneBn}{0.69}   % model1, Bengali
\newcommand{\offTwoHi}{...}    % model2 (English-only pool), Hindi
\newcommand{\offTwoBn}{...}    % model2, Bengali
\newcommand{\offThreeHi}{...}  % model3 (XLM-R-large, native), Hindi
\newcommand{\offThreeBn}{...}  % model3, Bengali
\newcommand{\offRank}{3rd of 87 runs}   % overall standing
```

Until you do, the table prints **[FILL]** in bold in seven places. That is
deliberate — a loud marker cannot be submitted by accident, whereas a
plausible-looking placeholder number could. **Do not submit with [FILL] still
in the PDF.**

Two sentences will likely need a small revision once the real numbers are in,
and I will handle both if you send the scores:

- §11 Limitations still says our official score *"will be lower than reported
  here"* — written before any official number existed. If the official score
  is close to the development figure, that sentence becomes a validated
  prediction and should say so; if it is far below, the gap is worth a
  sentence of its own, because it directly measures how far our LLM judge
  panel's `None` boundary sits from the organisers' guidelines.
- §8.2 currently describes the dev-vs-official gap in the abstract. With
  numbers it should state the gap.

### What is already in that section without needing you

Computed from the submitted run files, so these are real and verifiable:

| Run | hi F/A/N | bn F/A/N |
|---|---|---|
| `model1` | 39.0 / 44.0 / 17.0 | 38.8 / 40.2 / 21.0 |
| `model2` | 38.6 / 42.4 / 19.0 | 35.4 / 42.4 / 22.2 |
| `model3` | 35.2 / 44.4 / 20.4 | 32.2 / 43.2 / 24.6 |

All three runs rank `Against` first and `None` last in both languages, which
independently corroborates the class-imbalance finding in §4 from the model
side. Three-run agreement is 87.2 % (hi) and 86.4 % (bn) — I corrected the
paper, which previously rounded both to "87 % in both languages".

---

## 🟡 THE BIGGEST REMAINING RISK — and it is cheap to fix

### 1. Blind-label 50 comments (~2 hours)

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

## 🟢 THE WORKING NOTE — written and compiling

**File:** `Submission/AISOME.tex` → builds to `AISOME.pdf`, **10 pages**
(the limit is 6–10 including references), 18 references, 3 figures.
Due **20 September 2026**. Submitting it *and* presenting at FIRE in December is
a precondition for being declared a winner.

Three things in the 19 Aug email changed what I had drafted, and are now fixed:

- **Title no longer names the team or track.** The old draft was literally
  *"Nirnay at AISoMe 2026: …"* — the exact pattern the organisers call out as
  discouraged. It is now *"Diagnosing the Evaluation Distribution:
  Argument-Taxonomy Distillation and an English Pivot for Climate Stance
  Detection in Hindi and Bengali"*.
- **Single column.** The template's own comment forbids `twocolumn` for CEUR-WS;
  my earlier draft had it.
- **Only one working note per track**, so the superseded `working_notes.tex` has
  been deleted. `AISOME.tex` is the one.

### Authorship — done

Single author, exactly as you specified: **Koushik Deb**, ORCID
0000-0001-7902-4387, `koushik_phd21@iiitkalyani.ac.in`, sole affiliation
*Indian Institute of Information Technology Kalyani, Kalyani, West Bengal,
741235, India*. The Accenture affiliation and the gmail address are gone, and
the name carries no "Dr."/"Prof." prefix, as the organisers require.

One implementation note in case you edit this in Overleaf: the underscore in
your email is written `koushik\string_phd21@...`. A plain `\_` compiles fine in
body text but **fails** inside the `email=` key, because hyperref writes that
value into a `mailto:` URL where `\_` expands to `\T1\textunderscore`. Leave
`\string_` as it is. Verified: the footer prints the address correctly and the
`mailto:` and `https://orcid.org/…` links both resolve.

### The one thing still to fill in

**The Das 2025 citation.** The organisers pointed at it as methodological
guidance; it is paywalled and I could not read it. If you can get it, add it to
`referenences.bib` and cite it in the Related Work section. The paper is
complete and submittable without it — this is an improvement, not a blocker.

### Read the Generative AI declaration before you submit

CEUR **mandates** it, and ours is substantial rather than boilerplate: LLMs
generated the synthetic corpus, produced the pseudo-labels distilled into the
submitted classifiers, produced the development labels used for calibration, and
did all the translation. **Read that section and confirm it describes your
involvement the way you want** — it is your name on the paper. Adjust the wording
freely, but do not weaken the factual content; reviewers will check it against
the method section.

### Rendering

It compiles locally (0 errors, 0 undefined citations). On Overleaf use the
template the organisers linked, then upload `AISOME.tex`, `referenences.bib`
and the `figures/` folder. `ceurart.cls` and `elsarticle-num-names.bst` are in
the repo if you need them.

**One copyright form per author** is required — the organisers will send it.

---

## 🔵 DECISIONS FOR YOU

### 2. GitHub repo is PRIVATE — flip it after results

I set `DevDaring/FIRE_AISOME` private because `STRATEGY.md` spells out the
test-set discovery and the competition was open. **Make it public once results are declared** — good practice for the CEUR paper, and the code is a
genuine artifact. Say the word and I will flip it.

### 3. HuggingFace repos are PRIVATE — publish with the paper

- `Debk/nirnay-aisome2026-setu` — both adopted checkpoints + model card
- `Debk/nirnay-aisome2026-data` — taxonomy, synthetic corpus, judge labels, dev split

The **taxonomy is the most citable thing here.** Publishing it alongside the
paper is how it gets used by others.

### 4. Two dead API keys — replace or delete

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
| Runs submitted | organisers confirmed receipt; 80+ runs in the track |
| GitHub | 22 commits, private, secrets and organiser data excluded |
| HuggingFace | models + dataset uploaded with cards |
| GPU | **destroyed — 0 instances, total spend $1.11**, $7.21 credit left |
| Background jobs | none running |
| Working note | `Submission/AISOME.tex` — compiles to 10 pages, 3 figures, 18 refs |

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
