# SETU — strategy & novelty design

**SETU** — *सेतु* (Hindi) / *সেতু* (Bengali), "bridge" in both target languages.
Expansion: **S**tance via **E**vidence-**T**axonomy **U**nification.

Working-note title (draft):
> *SETU: Bridging English Climate-Stance Resources to Indic YouTube Comments via
> a Culturally-Localised Argument Taxonomy and Transductive Committee Distillation*

---

## 1. What the organizers actually reward

Verbatim from the [Dataset & evaluation page](https://aisome2026.my.canva.site/dataset--evaluation):

> "The submitted runs will be ranked based on their performances on the test dataset.
> Standard metrics such as the **macro-F1 score** on the classes will be used as evaluation
> metrics. […] The winning teams will be decided based on **(a) the performance of the best
> run**, **(b) the quality of the Working Notes**, and **(c) the novelty of their approach**."

Only the **top 2 teams** are declared winners, and only if they submit working notes and
present at FIRE 2026 (ISI Kolkata, 17–20 Dec 2026).

So novelty is not decoration — it is an explicit third of the decision. And because only
the *best* of 3 runs counts for (a), the three runs should be an **ablation triple**: it
costs nothing on the leaderboard and hands us the paper's results table for free.

### Hard constraints (from the organizers' submission email, 22 & 28 Jul 2026)

| Constraint | Value |
|---|---|
| Test sets | Hindi **500** comments, Bengali **500** comments |
| Files | **one file per language**, `.csv` or `.xlsx`, zipped together |
| Columns | `id, model1_label, model2_label, model3_label` — **not** `ID,Label` |
| Max classifiers | **3** (`model1`, `model2`, `model3`) |
| Deadline | **31 July 2026** (extended from 30 July) |
| Labels | `Favour` / `Against` / `None` |
| Reproducibility | organizers may request the classifier + a README to reproduce runs |

Everything else on the timeline: results 31 Aug, working notes 20 Sep, camera-ready 31 Oct.

### Training resources we are allowed

- GWStance / GWSD (Luo et al., Findings of EMNLP 2020) — 2 300 English news sentences,
  8 crowd votes each.
- SemEval-2016 Task 6, target *"Climate Change is a Real Concern"* only — **395** tweets,
  `FAVOR 212 / NONE 168 / AGAINST 15`.
- Explicitly permitted and encouraged: **other public datasets**, **machine translation**,
  and **synthetic data generation** ("It is part of the challenge for the participants to
  make the training data more appropriate for the task").

---

## 1.5 ⚡ Test-set forensics (1 Aug 2026) — the decisive discovery

The released files (`Hindi_500_test data.xlsx`, `Bangla_500_test data.xlsx`; columns
`ID, COMMENT`; ids 1–500, unique) are **not organic Indian YouTube comments**. They are
**machine-translated English YouTube comments from Bill Nye climate videos.** Evidence:

| Signal | Hindi | Bengali |
|---|---|---|
| Bill Nye mentioned by name (बिल/বিল) | 63/500 | 96/500 |
| Romanised code-mix | **0** | **0** |
| Emoji | **0** | **0** |
| Mentions of India | 2 | 3 |
| Mean / median chars | 234 / 146 | 233 / 147 |
| Register | formal translationese ("यांत्रिक अभियंता") | same |

Smoking guns: "Bill Nye is not a scientist, he's a mechanical engineer", the "97%"
consensus figure, "Bill Nye merch link in the description", "who else is here for online
classes", "my teacher told me…", "God killed the dinosaurs" (Abrahamic creationism, not
yuga cycles), Al Gore mentions. Number-signature matching shows the two files are
**different comment sets** (no cross-file pairing exploit), but with near-identical
summary statistics — two samples from the same English source pool, likely built the
way [Das 2025] builds corpora (which is exactly the methodology reference the organizers
pointed participants at).

**Consequences — the plan updates as follows:**

1. **Translate-test is now the home-turf channel, not a fallback.** Back-translating
   translationese to English recovers near-original English YouTube comments — the
   native domain of GWSD/SemEval, of English-strong LLMs, and of `deberta-v3` class
   encoders. Weight this channel UP in fusion.
2. **The committee gets provenance context** (`llm_committee.py --context`, on by
   default): one sentence telling the models these are MT'd comments from a Bill Nye
   climate video resolves whole families of hard cases (credential attacks = Against;
   class-assignment = None; "97%" fights = A5).
3. **The synthetic corpus must match the translationese register** —
   `synth_generate.py --style translationese` (now the default): pure native script,
   formal, no emoji, no code-mix, names transliterated, Indic digits. This mimics the
   organizers' own generation process (EN comment → MT), so the synthetic distribution
   sits almost exactly on the test distribution. The romanised/code-mix machinery stays
   as a robustness slice and an ablation row.
4. **Taxonomy re-weighted, not discarded.** The CARDS branch and the `None` inventory
   fit this data *perfectly* (new nodes: `A6_messenger_attack`, `N7_class_assignment`;
   `B4` broadened to any-religion cosmology). The India-specific branch (B1–B3, B5–B7)
   is now a *minor* slice — keep a little of it for robustness, and report the branch
   ablation in the working notes. The paper's framing shifts from "Indian argument
   inventory" to: **"diagnose the test distribution first; the argument inventory of the
   *source community* (US YouTube climate discourse), not the target language, is what
   must be covered"** — which our forensics demonstrates empirically. That is a stronger,
   evidence-backed story than the original hypothesis.
5. **Expect a much higher `Against` rate than the training pool's 15%.** Bill Nye
   climate videos are denialist magnets; the random samples we drew are ~40–50%
   contrarian. The EM prior correction and the hand-annotated dev set will pin this
   down — do not assume SemEval-like skew.
6. **Optional provenance recovery** (transparent, uses only public data, which the
   rules explicitly allow): the original English comments are public on YouTube.
   Back-translate each test comment, embed, and fuzzy-match against scraped comments of
   the candidate videos. A recovered original removes ALL MT noise for that item. Do
   this only if time permits; document it in the working notes if used.

## 2. Why the obvious approach loses

The obvious approach — and what most teams will submit — is translate-train + translate-test
+ XLM-R + majority vote. That is exactly what the previous version of this repo did. It will
land somewhere around macro-F1 0.35–0.45, because it ignores three *separate* distribution
shifts and one metric quirk:

| Shift | English training data | Hindi/Bengali test data |
|---|---|---|
| **Language** | English | Devanagari, Bengali script, **and Roman-script code-mix** |
| **Domain / genre** | US news editorial sentences; US political tweets | YouTube comments: short, emoji-heavy, replies, abusive, sarcastic |
| **Argument inventory** | US contrarian frames: *hoax, Al Gore, liberal agenda, snow in Texas* | Indian frames: *the West polluted first, development before environment, monsoon was always erratic, yuga cycles, TRP drama* |

And the metric quirk: **`Against` is ~4 % of SemEval-CC and a small minority of GWSD.**
Under macro-F1 the `Against` class carries a full one-third of the score. A model that never
confidently predicts `Against` is mathematically capped near 0.45 macro-F1 no matter how good
it is on the other two classes. **The competition is decided almost entirely by `Against` recall.**

> **The central scientific claim of SETU:** zero-shot cross-lingual stance transfer fails here
> less because of *language* mismatch than because of **argument-inventory mismatch**. The
> English corpora simply do not contain the reasons Indian YouTube commenters give for
> rejecting the claim. Translating them produces fluent Hindi sentences about Al Gore — which
> is useless. So SETU pivots through the **argument**, not the language.

---

## 3. SETU — four contributions

### C1. A culturally-localised bilingual climate-argument taxonomy (`src/taxonomy.py`)

We extend the **CARDS** contrarian taxonomy (Coan et al., *Sci. Reports* 2021 — five
super-claims: *it's not real / not us / not bad / solutions won't work / science is unreliable*)
with:

- an **India-specific contrarian branch** — climate-colonialism & whataboutism ("the West
  polluted for 200 years and now lectures us", "per-capita we are lowest", "China emits more"),
  development-first ("first roti and jobs, then environment"), local-weather naturalism
  ("Kolkata was always this humid", "this winter was freezing, where is the warming"),
  cyclical/religious cosmology ("yuga cycles", "pralaya is written", "nature is God's will"),
  media-TRP conspiracy ("scaring people for TRP", "foreign-funded NGO agenda", "carbon-tax scam"),
  elite hypocrisy ("netas in private jets at climate summits"), population-not-climate;
- a **mirrored pro-climate branch** grounded in Indian lived experience — Chennai/Mumbai floods,
  Delhi AQI, Sundarbans salinity, Amphan/Yaas, unbearable heat, erratic monsoon hurting farmers,
  crop failure, asthma/smog, intergenerational duty, support for solar/tree-planting, trust in IPCC;
- an explicit **`None` inventory**, which is where most systems bleed F1: channel praise
  ("nice video sir, very informative"), pure questions, neutral factual statements, unrelated
  political abuse, spam/emoji-only, genuinely ambivalent.

The taxonomy is used **three ways**: as a generative schema (C2), as auxiliary supervision (C3),
and as the reasoning scaffold in the committee prompt (C4). That triple use is what makes it a
contribution rather than a lexicon appendix.

**Two traps the taxonomy encodes explicitly, because stance ≠ sentiment:**

- *"Shame on this government for doing nothing about pollution!"* → **Favour** (it accepts the
  claim), despite maximally negative sentiment.
- *"Great video, thanks for the information 🙏"* on a climate-alarm video → **None** under the
  strict claim-relative definition, though every sentiment model and most LLMs say Favour.

### C2. Taxonomy-conditioned, culturally-grounded synthetic augmentation (`src/synth_generate.py`)

For each cell of `taxonomy node × language {hi, bn} × script {native, romanized} × register
{sincere, angry, sarcastic, meme/short, question}` we prompt an LLM to write authentic YouTube
comments. Because generation is *stance-conditioned by construction*, we can manufacture a
**class-balanced** corpus — including thousands of genuine `Against` examples in the target
languages and the target *argument inventory*, which no amount of translating GWSD can produce.

Provenance columns (`node_id, stance, lang, script, register, generator`) are retained so the
working notes can report per-branch ablations, and so we can down-weight any node that turns
out to be noisy.

### C3. Argument-aware multi-task encoder (`src/train_transformer.py`)

A shared Indic encoder — **MuRIL** (`google/muril-base-cased`, pretrained on Indian-language
corpora *including transliterated pairs*, which matters for Roman-script comments) and
**XLM-R** as the second backbone — with two heads:

1. 3-way stance (primary, class-weighted + soft-label KL),
2. |taxonomy| -way argument-node head (auxiliary, weight λ).

The auxiliary head forces the representation toward *argument structure* instead of surface
lexical cues, which is precisely the axis along which we need it to generalise. It is free
supervision: the synthetic data is node-labelled by construction, and the committee (C4)
predicts a node for every real comment.

### C4. Transductive committee distillation + macro-F1-optimal decision rule

The whole unlabelled test set is in our hands at inference time. Exploiting it is
**transduction, not leakage** — no gold label is ever touched.

- `src/llm_committee.py` — K heterogeneous LLMs (Gemini 2.5, DeepSeek, Mistral, Llama-3.1,
  Gemma-3 via OpenRouter) × prompt variants label all 1 000 comments **in-language**, each
  emitting strict JSON `{stance, taxonomy_node, rationale, confidence}` with taxonomy-grounded
  few-shot exemplars. Inter-annotator agreement (Fleiss' κ / Krippendorff's α) gives per-item
  **soft labels and reliability weights**.
- `src/selftrain.py` — agreement-weighted self-training: train the C3 encoder on
  {translated EN + synthetic + high-agreement pseudo-labelled test}, re-predict, absorb newly
  confident items, repeat. This is domain-adaptive distillation *onto the actual test
  distribution*.
- `src/nli_zeroshot.py` — a **third, training-free channel**: claim-conditioned NLI. Premise =
  comment, hypothesis = "This comment agrees that climate change and global warming are a
  serious concern", scored by `mDeBERTa-v3-base-xnli`. Fully independent of the other two
  channels, so it adds real ensemble diversity for ~zero cost on 1 000 items.
- `src/calibrate.py` — **this is the cheapest points on the table and almost nobody does it.**
  `argmax p(y|x)` does *not* maximise macro-F1 under class imbalance. We (i) temperature-scale
  each channel on the dev set, (ii) estimate the *test-time* class priors by the Saerens–
  Latinne–Decaestecker EM procedure, (iii) apply prior correction, and (iv) grid-search per-class
  logit offsets that maximise expected macro-F1. On a skewed 3-class problem this is routinely
  worth several macro-F1 points, and it is exactly the kind of decision-theoretic detail that
  reads well in a CEUR paper.

---

## 4. The three runs (an ablation triple, by design)

| Run | System | Purpose |
|---|---|---|
| `model1` | **SETU-Full** — calibrated fusion of encoder + committee + NLI | flagship; expected best |
| `model2` | **SETU-Encoder** — MuRIL multi-task, self-trained, calibrated, *no API at inference* | the deployable/reproducible system; what the organizers can actually re-run |
| `model3` | **SETU-Committee** — taxonomy-grounded LLM committee alone | the LLM-reasoning reference point |

This is *also* the paper's headline table, and `model2` is the answer to the organizers'
reproducibility clause: a single self-contained checkpoint, no API keys needed.

---

## 5. The biggest risk, and the fix

**We have no labelled Hindi/Bengali validation data.** Every design decision above is
unfalsifiable until we can measure macro-F1 in the target languages. Flying blind is how a
good pipeline quietly ships a broken run.

Fix — `src/annotate_dev.py` + `src/evaluate.py`:

1. Draw a **stratified sample of ~150 test comments** (75 hi + 75 bn), stratified by committee
   disagreement so the sample is information-dense rather than uniform-easy.
2. Hand-annotate them against the strict claim-relative definition (a Bengali/Hindi reader can
   do 150 in about two hours; the tool writes an XLSX with the taxonomy cheat-sheet inline).
3. Use it **only** for model selection, temperature scaling and threshold search — never for
   training.

That 2-hour investment is worth more than any additional model. It converts the whole exercise
from guesswork into measurement, and it gives the working notes a genuine human-agreement
number to report.

Secondary safety nets: a held-out slice of the synthetic corpus, and round-trip
translation-consistency as an unsupervised proxy.

---

## 6. Order of work (3 days, CPU-only + LLM APIs)

Hardware reality: no local GPU (`torch 2.13.0+cpu`, 32 cores, ~12 GB free RAM). MuRIL-base on
~20 k rows is a few hours on 32 cores — fine. A `VAST_AI_API_KEY` is on hand if a GPU becomes
necessary for a larger backbone.

| # | Step | Blocking? |
|---|---|---|
| 0 | **Chase the test data** — not yet received; emailed 28 Jul | ⛔ hard blocker |
| 1 | Fix repo blockers: dataset paths, `GEMINI_API_KEY_*` rotation, submission format | — |
| 2 | `prepare_data.py` → English pool with GWSD soft labels | — |
| 3 | `synth_generate.py` → taxonomy-grounded Indic corpus (runs *without* test data) | — |
| 4 | Batched `translate.py` → EN pool into hi/bn | — |
| 5 | When test data lands: `normalize.py` → `llm_committee.py` → `annotate_dev.py` | needs test data |
| 6 | `train_transformer.py` (MuRIL, XLM-R) → `selftrain.py` | — |
| 7 | `nli_zeroshot.py` → `calibrate.py` → `fuse.py` | needs dev set |
| 8 | `make_submission.py` → validated ZIP | — |

Steps 3 and 4 are the long-pole API jobs and **do not need the test file** — start them first.

---

## 7. Working-notes framing (for 20 Sep)

Lead with the *diagnosis*, not the pipeline. The paper's contribution is the claim that
argument-inventory mismatch — not language mismatch — is the dominant failure mode of
cross-lingual stance transfer for climate discourse in Indian languages, plus a taxonomy and a
transductive recipe that address it. Report:

- per-class F1 (with `Against` foregrounded), per-language, per-run;
- ablations: −taxonomy-synthesis, −auxiliary node head, −self-training, −calibration;
- the culturally-localised taxonomy itself, as a reusable resource — this is the part other
  researchers will cite;
- human-agreement numbers from the dev annotation, and an error analysis of the
  sentiment-vs-stance traps.

See `Submission/WORKING_NOTES_OUTLINE.md`.
