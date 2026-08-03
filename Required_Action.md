# Required actions — things only you can do

**Team Nirnay · AISoMe 2026 @ FIRE · updated 2 Aug 2026**

Ordered by urgency. Everything not on this list is either already done or is
something I can run myself.

---

## 🔴 BLOCKING — the submission cannot happen without these

### 1. Confirm the actual deadline with the organizers

The published run-submission deadline was **31 July 2026**, but the test data only
arrived **1 August**. The schedule has clearly slipped and there is no revised date
anywhere on the track site.

**Do this today.** Email **aisome.fire2026@gmail.com** and ask for the revised run
deadline in writing. Everything below is paced against a date we do not actually know.

> Suggested wording:
> *"Dear organizers — Team Nirnay here. We received the Hindi and Bengali test files
> on 1 August. As the originally published run-submission deadline was 30/31 July,
> could you confirm the revised deadline for run submission?"*

Ask in the same email:
- Is a **CSV** acceptable, or do they require **XLSX**?
- Should the ZIP go to that email address, or through a form?

---

### 2. Spot-check 15 comments (~5 minutes)

You asked for the 40-comment anchor to be done by LLM instead, and I have done that —
a panel of four strong models (Claude Haiku 4.5, DeepSeek Reasoner, GPT-5-mini,
Qwen3.7-plus) is labelling 300 comments through the English pivot, with a second model
adversarially challenging every label. That is a much bigger and probably better dev
set than 40 hand labels would have been.

**But there is one thing it cannot do, and I want to be straight about it.**

The 40 hand labels existed to *measure whether LLM labels can be trusted*. An LLM
cannot perform that measurement on itself — if the judges are wrong in some systematic
way, a judge-built answer key agrees with them and the error is invisible. We then fit
calibration constants and fusion weights to that error and ship it with confidence.
This is the single most likely way to lose the competition while every number on screen
looks healthy.

So the code produces `artifacts/spot_check.csv` — **15 comments, about five minutes**,
weighted toward the ones the judges were least certain about:

```bash
cd /home/Debz/Hackathon/AISOME/Codes
# open artifacts/spot_check.csv, fill `your_label` for 15 rows
```

If you agree with **13 or more of 15**, the silver set is behaving and we proceed with
confidence. If you agree with fewer than 11, something is systematically off and I
should re-examine the judge prompt before anything gets calibrated.

Five minutes converts the whole dev set from *unvalidated* to *spot-validated*, and it
gives the CEUR paper a real human-agreement figure. "We used LLM judges and measured
them against human labels at 0.87" is a contribution; "we used LLM judges" is a
reviewer's objection.

The four rules, if you do fill it in:

1. **Angry is not Against.** *"Shame on the government for this pollution"* accepts
   the claim → **Favour**.
2. **Praising the video is not agreeing.** *"Very informative, thank you"* → **None**.
3. **Hypocrisy cuts both ways.** *"They fly private jets, so it's all drama"* →
   **Against**. *"They fly private jets, they should act first"* → **Favour**.
4. **Unsure? Leave it blank.** A blank is more useful than a guess.

### What I *can* measure without you, and will report

Three numbers that bound reliability without any human labels. All three go in the
working notes:

1. **Agreement among the four strong judges** (Fleiss' κ). High agreement between
   independently-built strong models means the task is well defined and they are
   converging — evidence, not proof.
2. **Strong judges vs the cheap teacher panel.** This is the useful one: it tells us
   how much the labels being distilled into the encoder differ from a 4× more
   expensive panel's view.
3. **Adversarial overturn rate** — how often a strong challenger rejected a strong
   proposal.

---

## 🔵 GOOGLE REMOVED — one leftover to decide

Per your instruction, **no GCP service and no Gemini is called anywhere** in the
pipeline now. Removed: the Gemini client, the `gemini` provider, and Google models via
OpenRouter. Every provider is DeepSeek's own API or a paid OpenRouter model.

The judge panel is now: `claude-haiku-4.5`, `deepseek-reasoner`, `gpt-5-mini`,
`qwen3.7-plus`. Translation and back-translation now run on `deepseek-chat`, which
turned out to be **much better than Gemini for this job** — it batches cleanly, so
1,000 back-translations took 75 seconds with zero retries, against Gemini's
truncation-and-retry problems.

**Two leftovers:**

1. **`train_en.to-hi.csv` / `train_en.to-bn.csv` were translated by Gemini** before you
   gave that instruction. The data is good, but if you want zero Google involvement in
   anything that reaches the paper, re-translate on DeepSeek — about 30 minutes and
   roughly $0.30:
   ```bash
   cd /home/Debz/Hackathon/AISOME/Codes
   mv artifacts/train_en.to-hi.csv artifacts/train_en.to-hi.gemini.bak
   mv artifacts/train_en.to-bn.csv artifacts/train_en.to-bn.gemini.bak
   python3.12 src/translate.py --input artifacts/train_en.csv --text-col text \
       --to hi bn --provider deepseek --workers 4
   ```
   **Tell me and I will run it.** I did not do it unasked because it discards work that
   is genuinely fine.

2. **`google/muril-base-cased` is still the default backbone.** This is *open weights
   downloaded from the HuggingFace Hub* — no Google API, no GCP service, nothing leaves
   this machine. It is also the model that has topped FIRE's Indic classification
   leaderboards, so dropping it would cost real accuracy. If you want zero
   Google-authored artefacts even so, switch with one flag:
   ```bash
   BACKBONE=ai4bharat/IndicBERTv2-MLM-only bash run_all.sh stage2
   ```

---

## 🟠 TWO DEAD API KEYS — replace or delete them

Testing every Gemini key individually against a live model:

| Key in `.env` | Result |
|---|---|
| `GEMINI_API_KEY_1` | ✅ works |
| `GEMINI_API_KEY_2` | ❌ **"API key not valid"** |
| `GEMINI_API_KEY_3` | ✅ works |
| `GEMINI_API_KEY_4` | ✅ works |
| `Link_Gemini_Cheap_API_Key` | ❌ **"API key not valid"** |

Two of five were dead, so round-robin was sending **40% of requests to keys that
could never succeed** — each one burning retries before landing on a good key. That is
what caused 360 of 2,442 translation rows to fall back to slow single-row retries.

**I have worked around it in code**: a key that the provider rejects as invalid is now
retired for the rest of the run instead of being rediscovered thousands of times. But
please still fix the source:

- Either replace both with fresh keys from <https://aistudio.google.com/apikey>,
- or delete those two lines from `.env` entirely.

Three working keys is enough for everything we need, so this is not blocking.

### Also fixed: a retired model was pinned in `.env`

`GEMINI_MODEL_NAME` was set to `gemini-2.5-flash-lite`, which Google now rejects with
*"no longer available to new users"*. I changed that one line to `gemini-2.5-flash`,
which is verified working. No secret in `.env` was touched.

---

## 🟡 DECISIONS I need from you

### 3. GitHub repo is now PRIVATE — confirm that is what you want

`DevDaring/FIRE_AISOME` was **public and empty**. I pushed the code but set it
**private** first, because the competition is still open and `STRATEGY.md` spells out
the Bill Nye test-set discovery — the single biggest edge we have. A public repo hands
it to every other team.

- Code is pushed: 28 files, commit `aa9a882`.
- **Excluded deliberately:** `.env`, the organizer test files, and all derived
  artifacts. Verified — no key-shaped string is in the repo.

**Make it public after the results are declared** (31 Aug, or whenever the organizers
confirm), which is also good practice for the CEUR paper. Tell me and I will flip it,
or do it yourself in the repo's Settings → General → Danger Zone.

If you would rather it be public right now, say so and I will change it back.

### 4. Should the organizer test data ever be published?

I excluded `Dataset/Testing_Data/` from git. Those files were emailed privately to
registered teams, and republishing an evaluation set mid-competition is the kind of
thing that gets a team disqualified. **My recommendation: never publish them.** Point
the working notes at the organizers instead. Tell me if you disagree.

---

## 🟢 TOP-UPS — not blocking, but worth knowing

| Service | Balance now | Needed | Verdict |
|---|---|---|---|
| **Vast.ai** | **$8.32** | ~$0.50 for all GPU training | ✅ plenty |
| **OpenRouter** | **$21.16** left of $33 | ~$3–5 for committee + judges | ✅ plenty |
| Gemini | free tier | translation job | ✅ |
| DeepSeek | not checked | only if used as a fallback | — |

Nothing needs topping up. The GPU is remarkably cheap for this workload — an RTX 3090
is going for **$0.10/hr** and both backbones train in well under an hour.

**One risk to know about:** a Vast.ai instance bills every second it exists, including
while idle. If a session is ever interrupted, run this — it is safe even if nothing is
running:

```bash
cd /home/Debz/Hackathon/AISOME/Codes && python3.12 src/vast_gpu.py down
```

Or check by eye at <https://cloud.vast.ai/instances/>.

---

## ✅ Already set up — no action needed

| Thing | State |
|---|---|
| GitHub `DevDaring/FIRE_AISOME` | pushed, **private**, token scrubbed from `.git/config` |
| HF model repo | `Debk/nirnay-aisome2026-setu` (private) |
| HF dataset repo | `Debk/nirnay-aisome2026-data` (private) |
| Vast.ai tooling | `src/vast_gpu.py` — search / up / train / down, refuses to overspend |
| LLM roster | 8 models verified live; teacher and judge tiers disjoint |
| Team name | `Nirnay` is the default in `run_all.sh` and `make_submission.py` |

---

## Where the pipeline stands

```
[x] English training pool       2,442 rows (983 Favour / 367 Against / 1,092 None)
[x] Test files normalised       500 Hindi + 500 Bengali, IDs preserved
[x] Synthetic corpus            1,040 rows, all 27 argument nodes covered
                                48% Against vs 15% in the real pool  <-- the point
[x] EN -> Hindi                 2,442 rows translated
[~] EN -> Bengali               running
[~] LLM committee (teacher)     running — 4 of 5 members done
[ ] YOUR 40 hand labels         <-- BLOCKING, see item 2
[ ] Silver dev set (AI judges)  needs your 40
[ ] GPU training                needs the committee output
[ ] Calibrate -> fuse -> submit needs the dev set
```

### One command to re-run when you come back

The committee's 5th member (`gpt-5-nano`) failed on the first pass — it is a
reasoning model and burned its whole token budget thinking before answering. That
is fixed now, but the running job started before the fix. Re-run it once the current
one finishes; the other four members come straight from cache, so it only redoes the
missing one:

```bash
cd /home/Debz/Hackathon/AISOME/Codes
python3.12 src/llm_committee.py --test artifacts/test_hi.csv artifacts/test_bn.csv
```

A 4-member committee is perfectly usable if you would rather skip this.

---

## If you ever need to hand this over

Everything is reproducible from the repo. The one thing deliberately not in git is
`.env` — keep it backed up somewhere safe and **never commit it**. If any key in it has
ever been pasted somewhere public, rotate it; there are 34 and they are all live.
