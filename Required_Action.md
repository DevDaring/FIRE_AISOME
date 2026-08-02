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

### 2. Hand-label 40 comments (~25 minutes) — the anchor set

This is the one modelling-critical step a machine cannot do for you, and it gates
calibration, fusion weights and run selection.

```bash
cd /home/Debz/Hackathon/AISOME/Codes
bash run_all.sh seed
# open artifacts/seed_to_annotate.csv
# read  artifacts/seed_to_annotate_CODEBOOK.txt
# fill the `gold` column with:  Favour | Against | None     ('?' if truly unsure)
python3.12 src/annotate_dev.py finalise \
    --input artifacts/seed_to_annotate.csv --out artifacts/seed_gold.csv
```

The four rules that resolve most hard cases:

1. **Angry is not Against.** *"Shame on the government for this pollution"* accepts
   the claim → **Favour**.
2. **Praising the video is not agreeing.** *"Very informative, thank you"* → **None**.
3. **Hypocrisy cuts both ways.** *"They fly private jets, so it's all drama"* →
   **Against**. *"They fly private jets, they should act first"* → **Favour**.
4. **Unsure? Type `?`.** It is dropped and counted, which is more useful than a guess.

Without this, the AI judges that build the rest of the dev set have **no measured
reliability**, and we would be tuning on labels we cannot vouch for.

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
[~] Synthetic corpus            running — 270 cells, ~1,600 rows
[~] EN -> Hindi/Bengali         running — 2,442 rows x 2 languages
[ ] LLM committee (teacher)     next, once the two above finish
[ ] YOUR 40 hand labels         <-- BLOCKING, see item 2
[ ] Silver dev set (AI judges)  needs your 40
[ ] GPU training                needs the committee output
[ ] Calibrate -> fuse -> submit needs the dev set
```

---

## If you ever need to hand this over

Everything is reproducible from the repo. The one thing deliberately not in git is
`.env` — keep it backed up somewhere safe and **never commit it**. If any key in it has
ever been pasted somewhere public, rotate it; there are 34 and they are all live.
