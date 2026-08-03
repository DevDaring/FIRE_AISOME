"""Build a SILVER dev set with LLM judges, anchored to a small human seed.

The problem
-----------
Calibration, fusion weights and run selection all need labelled Hindi/Bengali
examples, and none exist. Hand-labelling 150 comments is ~2 hours. This module
does most of that work with API calls instead — but the naive version of this
idea is actively dangerous, so read why before using it.

Why the naive version is worse than nothing
-------------------------------------------
If you label the dev set with the same models that form the teacher committee,
then measure the committee on that dev set, the committee scores ~1.0 by
construction. Every calibration constant and every fusion weight is then fitted
to the committee's own biases, and you will confidently ship a worse system than
if you had not calibrated at all. The dev set must be *independent* of anything
it is used to evaluate.

Four safeguards make it usable
------------------------------
1. **Disjoint judge tier.** Judges come from ``llm.JUDGE_MEMBERS``, which shares
   no model with ``llm.TEACHER_MEMBERS``. ``llm.assert_disjoint()`` enforces it.
   Judges are also stronger and reasoning-capable (deepseek-reasoner et al.).

2. **English pivot.** Test-set forensics showed these are machine-translated
   English comments. Back-translating recovers near-original English, and every
   LLM is markedly more reliable in English than in Hindi or Bengali. Judges see
   the Indic original *and* the back-translation together.

3. **Adversarial verification.** A second, different judge sees the proposed
   label plus its reasoning and is asked to refute it. Only labels that survive
   refutation become silver. Survivors of a split vote go to the human queue
   instead of being guessed.

4. **Human-anchored reliability.** You still hand-label a small seed (default 40,
   ~25 minutes). We never train or calibrate on the seed — we use it to *measure
   the judges*. That converts "we hope the silver labels are good" into a number
   you can put in the paper, and it tells you whether to trust the silver set at
   all.

Interpreting the anchor score (printed at the end):

    >= 0.85 macro-F1 vs human   silver set is safe for calibration and weights
    0.70 - 0.85                 usable, but hand-label another 60-80 comments
    <  0.70                     do NOT calibrate on it; hand-label the full 150

Workflow
--------
    # 0. one-off sanity check that the tiers do not overlap
    python3.12 src/llm.py judge

    # 1. hand-label a small seed (~25 min) — this is the anchor, not optional
    python3.12 src/annotate_dev.py sample --committee artifacts/committee_hi.csv \\
        artifacts/committee_bn.csv --n 40 --out artifacts/seed_to_annotate.csv
    #    ... fill the `gold` column ...
    python3.12 src/annotate_dev.py finalise --input artifacts/seed_to_annotate.csv \\
        --out artifacts/seed_gold.csv

    # 2. let the judges label a larger sample, and score them on the seed
    python3.12 src/silver_dev.py --committee artifacts/committee_hi.csv \\
        artifacts/committee_bn.csv --test-en artifacts/test_hi_en.csv \\
        artifacts/test_bn_en.csv --seed-gold artifacts/seed_gold.csv \\
        --n 300 --out artifacts/dev_gold.csv
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from common import (ARTIFACTS_DIR, LABELS, SEED, macro_f1, normalize_label,
                    read_csv, score_report, set_seed, write_json)
from taxonomy import ANNOTATION_GUIDELINES, NODE_IDS, UNKNOWN_NODE, stance_of_node

BATCH = 8

_SYSTEM = """\
You are the senior adjudicator for a stance-detection shared task at FIRE 2026.
Junior annotators disagree on these comments; your ruling is final and is used as
reference data, so you apply the codebook literally and you do not guess.

You never confuse sentiment with stance. You return only JSON.
"""

_CONTEXT = (
    "These comments were originally posted in ENGLISH under YouTube videos about "
    "climate change featuring the science communicator Bill Nye, then machine-"
    "translated into Hindi or Bengali. You are shown the translated text and an "
    "automatic back-translation into English. The back-translation is usually "
    "closer to what the commenter actually wrote — prefer it when the two differ, "
    "but check the original for meaning the back-translation may have dropped."
)


# ---------------------------------------------------------------------------
def build_judge_prompt(items: list[dict]) -> str:
    listing = []
    for it in items:
        listing.append(
            f'  "{it["id"]}": {{\n'
            f'      "original_{it["lang"]}": {json.dumps(it["text"], ensure_ascii=False)},\n'
            f'      "back_translation_en": {json.dumps(it.get("text_en") or "", ensure_ascii=False)}\n'
            f'  }}')
    return f"""\
{_CONTEXT}

{ANNOTATION_GUIDELINES}

For EACH comment below:
  1. Decide what the commenter is actually asserting about the CLAIM.
  2. Pick the single best argument node id (use "{UNKNOWN_NODE}" only if truly none fit).
  3. Give the stance, which must be consistent with that node.
  4. Give a one-sentence reason (max 25 words) naming the specific phrase that decided it.
  5. Give confidence in [0,1]. Use < 0.6 when the comment is genuinely ambiguous —
     under-confidence is useful to us, false certainty is not.

Comments:
{{
{",".join(chr(10) + x for x in listing)}
}}

Return exactly:
{{"labels": {{"<id>": {{"node": "<node id>", "stance": "Favour|Against|None",
"reason": "<...>", "confidence": 0.0}}, ...}}}}
"""


def build_refute_prompt(items: list[dict]) -> str:
    listing = []
    for it in items:
        listing.append(
            f'  "{it["id"]}": {{\n'
            f'      "comment": {json.dumps(it.get("text_en") or it["text"], ensure_ascii=False)},\n'
            f'      "proposed_stance": "{it["stance"]}",\n'
            f'      "proposed_reason": {json.dumps(it.get("reason", ""), ensure_ascii=False)}\n'
            f'  }}')
    return f"""\
{_CONTEXT}

{ANNOTATION_GUIDELINES}

Another annotator has proposed a stance for each comment below. Your job is to
CHALLENGE it. Assume they may have fallen into one of the standard traps:

  - reading negative sentiment as Against when the comment actually accepts the claim
  - reading praise of the video as Favour when it says nothing about the claim
  - reading an attack on a person as an attack on the claim, or vice versa
  - reading a question or a neutral fact as if it took a side

For each comment answer:
  "agree": true if the proposed stance is correct, false if it is wrong.
  "correct_stance": the stance you believe is right (repeat theirs if you agree).
  "why": one short sentence.

Be strict but not contrarian: if the proposal is right, say so.

{{
{",".join(chr(10) + x for x in listing)}
}}

Return exactly:
{{"reviews": {{"<id>": {{"agree": true, "correct_stance": "Favour|Against|None",
"why": "<...>"}}, ...}}}}
"""


def _uid(item) -> str:
    """Composite key for a comment.

    Both organizer test files are numbered 1..500, so `id` alone is NOT unique
    across the pooled sample. Keying vote dicts on it silently collapses Hindi and
    Bengali onto each other — the later batch overwrites the earlier and every
    Hindi comment then receives the Bengali comment's verdict. The symptom was two
    judges each reporting exactly 500/1000 adjudicated on a 1000-row pool.
    """
    return f"{item['lang']}:{item['id']}"


# ---------------------------------------------------------------------------
def _parse_labels(res, items) -> dict:
    out = {}
    table = (res or {}).get("labels") if isinstance(res, dict) else None
    if not isinstance(table, dict):
        return out
    valid = set(NODE_IDS) | {UNKNOWN_NODE}
    for it in items:
        v = table.get(str(it["id"]))
        if isinstance(v, str):
            v = {"stance": v}
        if not isinstance(v, dict):
            continue
        node = str(v.get("node") or UNKNOWN_NODE).strip()
        if node not in valid:
            node = UNKNOWN_NODE
        stance = normalize_label(v.get("stance", ""), default=None)
        if stance is None and node != UNKNOWN_NODE:
            stance = stance_of_node(node, default=None)
        if stance is None:
            continue
        try:
            conf = float(v.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        out[_uid(it)] = {"stance": stance, "node": node,
                         "reason": str(v.get("reason", ""))[:220],
                         "confidence": min(max(conf, 0.0), 1.0)}
    return out


def _parse_reviews(res, items) -> dict:
    out = {}
    table = (res or {}).get("reviews") if isinstance(res, dict) else None
    if not isinstance(table, dict):
        return out
    for it in items:
        v = table.get(str(it["id"]))
        if not isinstance(v, dict):
            continue
        corrected = normalize_label(v.get("correct_stance", ""), default=None)
        out[_uid(it)] = {
            "agree": bool(v.get("agree", True)),
            "correct_stance": corrected or it["stance"],
            "why": str(v.get("why", ""))[:220],
        }
    return out


def run_judges(items: list[dict], members, batch: int, workers: int,
               temperature: float) -> dict:
    """{member_id: {comment_id: verdict}} over the judge tier."""
    from llm import get_client
    votes = {}
    chunks = [items[i:i + batch] for i in range(0, len(items), batch)]
    for mid, prov, model in members:
        cli = get_client(prov, model)
        results = cli.chat_many([build_judge_prompt(c) for c in chunks],
                                system=_SYSTEM, temperature=temperature,
                                max_tokens=min(4096, 200 * batch + 400),
                                workers=workers, as_json=True, desc=f"judge/{mid}")
        v = {}
        for chunk, res in zip(chunks, results):
            v.update(_parse_labels(res, chunk))
        missing = [it for it in items if _uid(it) not in v]
        if missing:
            singles = cli.chat_many([build_judge_prompt([m]) for m in missing],
                                    system=_SYSTEM, temperature=temperature,
                                    max_tokens=500, workers=workers, as_json=True,
                                    desc=f"judge/{mid}/retry", cache_salt="single")
            for m, res in zip(missing, singles):
                v.update(_parse_labels(res, [m]))
        votes[mid] = v
        print(f"  {mid}: {len(v)}/{len(items)} adjudicated "
              f"{Counter(x['stance'] for x in v.values())} | {cli.stats()}")
    return votes


def run_refutation(proposals: list[dict], member, batch: int, workers: int) -> dict:
    from llm import get_client
    mid, prov, model = member
    cli = get_client(prov, model)
    chunks = [proposals[i:i + batch] for i in range(0, len(proposals), batch)]
    out = {}
    results = cli.chat_many([build_refute_prompt(c) for c in chunks],
                            system=_SYSTEM, temperature=0.0,
                            max_tokens=min(4096, 130 * batch + 300),
                            workers=workers, as_json=True, desc=f"refute/{mid}")
    for chunk, res in zip(chunks, results):
        out.update(_parse_reviews(res, chunk))
    n_challenged = sum(1 for v in out.values() if not v["agree"])
    print(f"  refuter {mid}: reviewed {len(out)}, challenged {n_challenged} "
          f"({n_challenged / max(len(out), 1):.0%})")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="LLM-judged silver dev set, anchored to a human seed")
    ap.add_argument("--committee", nargs="+", required=True,
                    help="artifacts/committee_hi.csv artifacts/committee_bn.csv")
    ap.add_argument("--test-en", nargs="+", default=[],
                    help="back-translated test files (artifacts/test_hi_en.csv ...); "
                         "strongly recommended — the English pivot is most of the gain")
    ap.add_argument("--seed-gold", default=str(ARTIFACTS_DIR / "seed_gold.csv"),
                    help="small hand-labelled anchor set; without it the silver "
                         "labels have no measured reliability")
    ap.add_argument("--n", type=int, default=300, help="comments to adjudicate")
    ap.add_argument("--hard-frac", type=float, default=0.65,
                    help="share drawn from where the teacher committee disagreed")
    ap.add_argument("--judges", nargs="+", default=None)
    ap.add_argument("--no-refute", action="store_true",
                    help="skip adversarial verification (faster, less reliable)")
    ap.add_argument("--min-agree", type=float, default=0.66,
                    help="judge agreement required to accept a silver label")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "dev_gold.csv"))
    ap.add_argument("--queue-out", default=str(ARTIFACTS_DIR / "human_queue.csv"),
                    help="unresolved comments for you to label by hand")
    ap.add_argument("--spot-check", type=int, default=15,
                    help="if no human seed exists, emit this many comments for a "
                         "~5 minute spot check (0 to skip)")
    ap.add_argument("--spot-out", default=str(ARTIFACTS_DIR / "spot_check.csv"))
    args = ap.parse_args()

    set_seed(SEED)
    from llm import assert_disjoint, live_members
    assert_disjoint()
    judges = live_members(args.judges, tier="judge")
    if len(judges) < 2:
        raise SystemExit(f"need at least 2 working judges, found {len(judges)}. "
                         f"Run `python3.12 src/llm.py judge` to see what is failing.")
    print(f"judges: {[j[0] for j in judges]}")

    # ---- assemble the candidate pool ---------------------------------------
    frames = []
    for p in args.committee:
        if not Path(p).exists():
            print(f"  SKIP (missing): {p}")
            continue
        df = read_csv(p)
        df["lang"] = "bn" if "bn" in Path(p).stem.lower() else "hi"
        frames.append(df)
    if not frames:
        raise SystemExit("no committee files — run llm_committee.py first")
    pool = pd.concat(frames, ignore_index=True)
    pool["id"] = pool["id"].astype(str)

    en_map = {}
    for p in args.test_en:
        if not Path(p).exists():
            print(f"  SKIP (missing back-translation): {p}")
            continue
        e = read_csv(p)
        lang = "bn" if "bn" in Path(p).stem.lower() else "hi"
        col = "text_en" if "text_en" in e.columns else e.columns[-1]
        for i, t in zip(e["id"].astype(str), e[col].astype(str)):
            en_map[(lang, i)] = t
    print(f"back-translations available for {len(en_map)} comments"
          + ("" if en_map else "  ← running WITHOUT the English pivot, quality will drop"))

    # sample: mostly where the teacher committee split
    picks = []
    per_lang = max(1, args.n // max(pool["lang"].nunique(), 1))
    for lang, sub in pool.groupby("lang"):
        sub = sub.copy()
        if "entropy" not in sub.columns:
            sub["entropy"] = 0.0
        hard = sub.sort_values("entropy", ascending=False).head(
            int(per_lang * args.hard_frac)).assign(stratum="disagreement")
        rest = sub[~sub["id"].isin(hard["id"])]
        rand = rest.sample(min(per_lang - len(hard), len(rest)),
                           random_state=SEED).assign(stratum="random")
        picks.append(pd.concat([hard, rand], ignore_index=True))
    sample = pd.concat(picks, ignore_index=True)

    # always include the human seed, so the judges can be scored on it
    seed_ids = set()
    seed_gold = None
    if Path(args.seed_gold).exists():
        seed_gold = read_csv(args.seed_gold)
        seed_gold["id"] = seed_gold["id"].astype(str)
        seed_gold = seed_gold[seed_gold["gold"].isin(LABELS)]
        seed_ids = {(r["lang"], r["id"]) for _, r in seed_gold.iterrows()
                    if "lang" in seed_gold.columns}
        extra = pool[pool.apply(lambda r: (r["lang"], r["id"]) in seed_ids
                                and r["id"] not in set(sample["id"]), axis=1)]
        if len(extra):
            sample = pd.concat([sample, extra.assign(stratum="seed")],
                               ignore_index=True)
        print(f"human seed: {len(seed_gold)} comments folded in for scoring")
    else:
        print(f"  WARNING: no {args.seed_gold} — the silver labels will have NO "
              f"measured reliability. Hand-label 40 comments first; it takes ~25 min "
              f"and it is what makes the rest trustworthy.")

    items = [{"id": r["id"], "lang": r["lang"], "text": str(r["text"]),
              "text_en": en_map.get((r["lang"], r["id"]), ""),
              "stratum": r["stratum"]}
             for _, r in sample.iterrows()]
    print(f"adjudicating {len(items)} comments")

    # ---- pass 1: judges vote ------------------------------------------------
    print("\n--- pass 1: judge panel ---")
    votes = run_judges(items, judges, args.batch, args.workers, args.temperature)

    # ---- consensus ----------------------------------------------------------
    rows = []
    for it in items:
        cid = _uid(it)
        cast = [(m, v[cid]) for m, v in votes.items() if cid in v]
        if not cast:
            rows.append({**it, "silver": None, "judge_agree": 0.0, "reason": "",
                         "node": UNKNOWN_NODE, "n_judges": 0})
            continue
        tally = Counter(v["stance"] for _, v in cast)
        top, n = tally.most_common(1)[0]
        best = max((v for _, v in cast if v["stance"] == top),
                   key=lambda v: v["confidence"])
        rows.append({**it, "silver": top, "judge_agree": n / len(cast),
                     "reason": best["reason"], "node": best["node"],
                     "n_judges": len(cast),
                     "mean_conf": sum(v["confidence"] for _, v in cast) / len(cast)})
    adj = pd.DataFrame(rows)
    print(f"\njudge consensus: {adj['silver'].value_counts().to_dict()}")
    print(f"  unanimous: {int((adj['judge_agree'] == 1.0).sum())}/{len(adj)}")

    # ---- pass 2: adversarial refutation ------------------------------------
    if not args.no_refute and len(judges) >= 2:
        print("\n--- pass 2: adversarial refutation ---")
        # the refuter is the judge that did NOT propose most of the labels
        refuter = judges[-1]
        props = [{"id": r["id"], "lang": r["lang"], "text": r["text"],
                  "text_en": r["text_en"], "stance": r["silver"],
                  "reason": r["reason"]}
                 for _, r in adj.iterrows() if r["silver"]]
        reviews = run_refutation(props, refuter, args.batch, args.workers)
        agree, flipped = [], 0
        for _, r in adj.iterrows():
            rv = reviews.get(f"{r['lang']}:{r['id']}")
            if rv is None:
                agree.append(True)
                continue
            agree.append(bool(rv["agree"]))
            if not rv["agree"]:
                flipped += 1
        adj["survived_refutation"] = agree
        print(f"  {flipped} labels challenged and sent to the human queue")
    else:
        adj["survived_refutation"] = True

    # ---- accept / defer -----------------------------------------------------
    accepted = adj[(adj["silver"].notna())
                   & (adj["judge_agree"] >= args.min_agree)
                   & (adj["survived_refutation"])].copy()
    deferred = adj[~adj.index.isin(accepted.index)].copy()
    print(f"\naccepted as silver: {len(accepted)}   deferred to human: {len(deferred)}")

    # ---- the anchor: how good are the judges, really? ----------------------
    report = {"judges": [j[0] for j in judges], "adjudicated": len(adj),
              "accepted": len(accepted), "deferred": len(deferred),
              "english_pivot": bool(en_map),
              "silver_distribution": accepted["silver"].value_counts().to_dict()}

    # Reliability evidence obtainable WITHOUT human labels. None of this proves
    # correctness — only a human can do that — but all three are real, reportable
    # numbers, and together they bound how much trust the silver set deserves.
    from llm_committee import fleiss_kappa
    strong_votes = [[v[cid]["stance"] for v in votes.values() if cid in v]
                    for cid in (adj["lang"].astype(str) + ":" + adj["id"].astype(str))]
    kappa = fleiss_kappa(strong_votes)
    report["judge_panel_fleiss_kappa"] = round(float(kappa), 4)
    print(f"\n{'='*70}\nRELIABILITY EVIDENCE (no human labels involved)")
    print(f"  1. agreement among the {len(judges)} strong judges: Fleiss kappa "
          f"{kappa:.3f}")
    print(f"     {'substantial/near-perfect — the task is well defined and strong'
                 ' models converge on it' if kappa >= 0.6 else
                 'only moderate — treat the silver labels with real caution'}")

    # 2. strong panel vs the cheap teacher panel: how much does 4x the spend buy?
    # Key on (id, lang): both organizer files are numbered 1..500, so a dict keyed on
    # id alone silently compares Hindi gold against Bengali predictions half the time.
    # And report per stratum — the disagreement stratum was SELECTED for teacher
    # disagreement, so agreement there is low by construction and says nothing about
    # the teacher's overall quality. The random stratum is the unbiased estimate.
    cheap = {(str(r["id"]), r["lang"]): r["committee_label"]
             for _, r in pool.iterrows()}
    acc = accepted.copy()
    acc["cheap"] = [cheap.get((str(r["id"]), r["lang"])) for _, r in acc.iterrows()]
    acc = acc[acc["cheap"].isin(LABELS) & acc["silver"].isin(LABELS)]
    if len(acc):
        report["strong_vs_cheap"] = {}
        for strat in ("random", "disagreement", "seed"):
            sub = acc[acc["stratum"] == strat]
            if len(sub) < 10:
                continue
            ag = float((sub["silver"] == sub["cheap"]).mean())
            f1 = macro_f1(sub["silver"].tolist(), sub["cheap"].tolist())
            report["strong_vs_cheap"][strat] = {
                "n": len(sub), "agreement": round(ag, 4), "teacher_macro_f1": round(f1, 4)}
            tag = "  <- UNBIASED ESTIMATE" if strat == "random" else                   "  (selected for teacher disagreement; low by construction)"
            print(f"  2. strong vs cheap, {strat:13} n={len(sub):3} "
                  f"agreement {ag:.1%}  teacher macro-F1 {f1:.3f}{tag}")
        rnd = report["strong_vs_cheap"].get("random")
        if rnd:
            print(f"     {'the teacher panel tracks the strong one closely, so the labels'
                         ' the encoder distils are sound' if rnd['agreement'] >= 0.80 else
                         'a real gap even on the unbiased sample — the teacher labels'
                         ' feeding the encoder are noisy'}")

    # 3. how often a strong challenger overturned a strong proposal
    if "survived_refutation" in adj.columns:
        overturned = int((~adj["survived_refutation"]).sum())
        report["refutation_overturn_rate"] = round(overturned / max(len(adj), 1), 4)
        print(f"  3. adversarial challenger overturned {overturned}/{len(adj)} "
              f"proposals ({report['refutation_overturn_rate']:.1%})")

    verdict = "UNMEASURED"
    if seed_gold is not None and len(seed_gold):
        m = accepted.merge(seed_gold[["id", "gold"]], on="id", how="inner")
        if len(m) >= 15:
            print(f"\n{'='*70}\nANCHOR — judges vs your {len(m)} hand-labelled comments")
            rep = score_report(m["gold"].tolist(), m["silver"].tolist(),
                               title="SILVER JUDGES vs HUMAN GOLD")
            f1 = rep["macro_f1"]
            report["anchor"] = rep
            verdict = ("SAFE" if f1 >= 0.85 else
                       "USABLE" if f1 >= 0.70 else "UNSAFE")
            report["verdict"] = verdict
            print({
                "SAFE": "  → Silver set is reliable enough for calibration and "
                        "fusion weights. Proceed.",
                "USABLE": "  → Borderline. Use it, but hand-label another 60-80 "
                          "comments and re-check before trusting the fusion weights.",
                "UNSAFE": "  → DO NOT calibrate on this. The judges disagree with you "
                          "too often. Hand-label the full 150 instead.",
            }[verdict])
            # per-language, since one language may be much worse than the other
            if "lang" in m.columns:
                for lang, sub in m.groupby("lang"):
                    if len(sub) >= 8:
                        print(f"    {lang}: macro-F1 "
                              f"{macro_f1(sub['gold'], sub['silver']):.3f} (n={len(sub)})")
        else:
            print(f"\n  only {len(m)} seed comments overlap — cannot measure the "
                  f"judges. Label a few more.")

    # ---- write --------------------------------------------------------------
    out = accepted[["id", "lang", "stratum", "text", "silver", "judge_agree",
                    "node", "reason"]].rename(columns={"silver": "gold"})
    out["source"] = "silver"
    # real human labels always override a silver one
    if seed_gold is not None and len(seed_gold):
        human = seed_gold.copy()
        human["source"] = "human"
        keep = [c for c in ("id", "lang", "text", "gold", "source") if c in human.columns]
        out = out[~out["id"].isin(set(human["id"]))]
        out = pd.concat([out, human[keep]], ignore_index=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {len(out)} dev labels -> {args.out}")
    print(f"  {out['source'].value_counts().to_dict()}")
    print(f"  {out['gold'].value_counts().to_dict()}")

    if len(deferred):
        q = deferred[["id", "lang", "stratum", "text", "silver", "judge_agree",
                      "reason"]].copy()
        q["gold"] = ""
        q["note"] = ""
        q.to_csv(args.queue_out, index=False)
        print(f"\nwrote {len(q)} unresolved -> {args.queue_out}")
        print("  These are the genuinely hard ones. Labelling even 30 of them by")
        print("  hand is the highest-value time you can spend from here.")

    # A cheap way to convert "unvalidated" into "spot-validated": a stratified
    # handful, weighted toward what the judges were least sure about. Fifteen
    # comments is about five minutes and is enough to catch a systematic error,
    # which is the failure mode that actually costs macro-F1.
    if seed_gold is None or not len(seed_gold):
        n_spot = min(args.spot_check, len(accepted))
        if n_spot:
            spot = pd.concat([
                accepted.nsmallest(max(n_spot // 2, 1), "judge_agree"),
                accepted.sample(n_spot - max(n_spot // 2, 1), random_state=SEED),
            ]).drop_duplicates(subset=["id"])
            spot = spot[["id", "lang", "text", "silver", "judge_agree", "reason"]]
            spot = spot.rename(columns={"silver": "llm_label"})
            spot["your_label"] = ""
            spot["agree"] = ""
            spot.to_csv(args.spot_out, index=False)
            print(f"\nwrote a {len(spot)}-comment SPOT CHECK -> {args.spot_out}")
            print("  ~5 minutes. Fill `your_label`; if you agree with 13+ of 15 the")
            print("  silver set is behaving. This is the cheapest way to stop flying")
            print("  blind, and it gives the working notes a human-agreement number.")

    write_json(Path(args.out).with_suffix(".silver_report.json"), report)
    print(f"\nverdict: {verdict}")
    if verdict == "UNMEASURED":
        print("  No human labels were involved, so judge ACCURACY is unmeasured — the")
        print("  three numbers above bound it but cannot establish it. Calibration")
        print("  fitted to these labels inherits whatever the judges get wrong.")


if __name__ == "__main__":
    main()
