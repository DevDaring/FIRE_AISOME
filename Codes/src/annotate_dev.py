"""Build and score a hand-annotated Hindi/Bengali dev set.

Why this is the single highest-value script in the repo
------------------------------------------------------
Training data is English; the test set is Hindi and Bengali; there is no labelled
target-language validation data anywhere. Without a dev set:

  * we cannot tell whether the synthetic corpus helped or hurt,
  * we cannot temperature-scale or tune class weights (calibrate.py needs labels),
  * we cannot choose between the three runs,
  * a bug that collapses a class reaches the organizers undetected.

Every one of those is a bigger threat to the score than any modelling choice. A reader
of Hindi/Bengali can annotate 150 comments in roughly two hours, and it converts the
whole exercise from guesswork into measurement.

Sampling is **committee-disagreement stratified**, not uniform. Comments the committee
agrees on unanimously are cheap and teach us little; comments it splits on are where the
decision boundary lives and where calibration parameters are actually determined. We
still include a uniform-random stratum so the dev set can also serve as a (roughly)
unbiased distribution estimate — the report prints both the stratified and the
random-stratum-only macro-F1 so the bias is visible rather than hidden.

Using the *unlabelled* test comments plus our own annotations is standard transductive
practice. No organizer-provided gold label is read at any point.

Workflow
--------
    # 1. draw the sample and write a fill-in sheet with the codebook inline
    python3.12 src/annotate_dev.py sample --committee artifacts/committee_hi.csv \
        artifacts/committee_bn.csv --n 150 --out artifacts/dev_to_annotate.csv

    # 2. open it, fill the `gold` column with Favour / Against / None
    #    (`?` to skip a comment you cannot decide)

    # 3. finalise: validates, reports human-vs-committee agreement, writes dev_gold.csv
    python3.12 src/annotate_dev.py finalise --input artifacts/dev_to_annotate.csv \
        --out artifacts/dev_gold.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ARTIFACTS_DIR, LABELS, SEED, read_csv, score_report, set_seed,
                    write_json)
from taxonomy import ANNOTATION_GUIDELINES


def cmd_sample(args):
    set_seed(args.seed)
    frames = []
    for path in args.committee:
        p = Path(path)
        if not p.exists():
            print(f"  SKIP (missing): {p}")
            continue
        df = read_csv(p)
        df["lang"] = "bn" if "bn" in p.stem.lower() else "hi"
        frames.append(df)
    if not frames:
        raise SystemExit("no committee files found — run llm_committee.py first")
    df = pd.concat(frames, ignore_index=True)
    df["id"] = df["id"].astype(str)

    per_lang = max(1, args.n // max(df["lang"].nunique(), 1))
    picks = []
    for lang, sub in df.groupby("lang"):
        sub = sub.copy()
        if "entropy" not in sub.columns:
            sub["entropy"] = 0.0

        # stratum 1: maximal committee disagreement — where the boundary is
        n_hard = int(per_lang * args.hard_frac)
        hard = sub.sort_values("entropy", ascending=False).head(n_hard)

        # stratum 2: uniform random from the rest — an unbiased view of the distribution
        rest = sub[~sub["id"].isin(hard["id"])]
        n_rand = per_lang - len(hard)
        rand = rest.sample(min(n_rand, len(rest)), random_state=args.seed) \
            if len(rest) else rest

        hard = hard.assign(stratum="disagreement")
        rand = rand.assign(stratum="random")
        picks.append(pd.concat([hard, rand], ignore_index=True))
        print(f"  {lang}: {len(hard)} disagreement + {len(rand)} random "
              f"= {len(hard) + len(rand)}")

    out = pd.concat(picks, ignore_index=True)
    keep = ["id", "lang", "stratum", "text_raw", "text", "committee_label",
            "agreement", "entropy", "node_id"]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    # The committee's own guess is deliberately NOT shown — seeing it would anchor the
    # annotator and inflate agreement. It is kept in a separate file for scoring.
    ref = out[["id", "lang", "committee_label", "agreement", "entropy"]].copy()
    sheet = out.drop(columns=[c for c in ("committee_label", "agreement", "entropy",
                                          "node_id") if c in out.columns])
    sheet["gold"] = ""
    sheet["note"] = ""
    sheet = sheet.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(dest, index=False)
    ref.to_csv(dest.with_suffix(".ref.csv"), index=False)
    guide = dest.with_name(dest.stem + "_CODEBOOK.txt")
    guide.write_text(ANNOTATION_GUIDELINES, encoding="utf-8")

    print(f"\nwrote {dest} ({len(sheet)} rows to annotate)")
    print(f"      {dest.with_suffix('.ref.csv')} (committee reference, do not peek)")
    print(f"      {guide} (the codebook — read it before starting)")
    print("\nFill the `gold` column with exactly: Favour | Against | None")
    print("Use `?` for comments you genuinely cannot decide; they are dropped, and the")
    print("count is reported as a difficulty statistic for the working notes.")
    print("\nThe four rules that resolve most hard cases:")
    print("  1. sentiment is not stance ('shame on the govt for the pollution' = Favour)")
    print("  2. praising the video is not agreeing with the claim (= None)")
    print("  3. hypocrisy cuts both ways — check whether the CLAIM is being rejected")
    print("  4. whataboutism used to reject the concern for India = Against")


def cmd_finalise(args):
    df = read_csv(args.input)
    if "gold" not in df.columns:
        raise SystemExit(f"{args.input}: no `gold` column")
    df["id"] = df["id"].astype(str)
    df["gold_raw"] = df["gold"].astype(str).str.strip()

    from common import normalize_label
    resolved, skipped, bad = [], 0, []
    for v in df["gold_raw"]:
        if v in ("", "?", "nan", "None?", "-"):
            resolved.append(None)
            skipped += 1
            continue
        try:
            resolved.append(normalize_label(v))
        except ValueError:
            resolved.append(None)
            bad.append(v)
    df["gold"] = resolved
    if bad:
        print(f"  {len(bad)} unrecognised label values, treated as skipped: "
              f"{sorted(set(bad))[:10]}")

    done = df[df["gold"].notna()].copy()
    print(f"annotated: {len(done)}/{len(df)}   skipped/unusable: {len(df) - len(done)}")
    if len(done) < 40:
        print("  WARNING: fewer than 40 usable rows. calibrate.py and fuse.py "
              "--search will overfit badly. Aim for >= 120.")
    print(f"  distribution: {done['gold'].value_counts().to_dict()}")
    if "lang" in done.columns:
        print(f"  by language: "
              f"{done.groupby('lang')['gold'].value_counts().unstack().fillna(0).to_dict()}")
    if "stratum" in done.columns:
        print(f"  by stratum: {done['stratum'].value_counts().to_dict()}")

    cols = [c for c in ("id", "lang", "stratum", "text", "text_raw", "gold", "note")
            if c in done.columns]
    out = done[cols]
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")

    stats = {"annotated": len(done), "skipped": len(df) - len(done),
             "distribution": done["gold"].value_counts().to_dict()}

    # human vs committee — a genuine agreement number for the working notes
    ref_path = Path(args.input).with_suffix(".ref.csv")
    if args.ref:
        ref_path = Path(args.ref)
    if ref_path.exists():
        ref = read_csv(ref_path)
        ref["id"] = ref["id"].astype(str)
        m = done.merge(ref[["id", "committee_label", "agreement", "entropy"]],
                       on="id", how="inner")
        m = m[m["committee_label"].isin(LABELS)]
        if len(m):
            rep = score_report(m["gold"].tolist(), m["committee_label"].tolist(),
                               title="LLM COMMITTEE vs HUMAN GOLD")
            stats["committee_vs_human"] = rep
            print("\nThis is the committee's honest macro-F1 — it is the number to beat,")
            print("and the number that tells you whether committee distillation is")
            print("worth doing at all.")
            for stratum, sub in m.groupby("stratum") if "stratum" in m.columns else []:
                if len(sub) >= 10:
                    from common import macro_f1
                    print(f"  stratum {stratum}: committee macro-F1 "
                          f"{macro_f1(sub['gold'], sub['committee_label']):.4f} "
                          f"(n={len(sub)})")
    else:
        print(f"(no {ref_path.name} — skipping the committee-vs-human comparison)")

    write_json(Path(args.out).with_suffix(".stats.json"), stats)


def main():
    ap = argparse.ArgumentParser(description="Hand-annotated dev set for SETU")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="draw a disagreement-stratified sample")
    s.add_argument("--committee", nargs="+", required=True,
                   help="artifacts/committee_hi.csv artifacts/committee_bn.csv")
    s.add_argument("--n", type=int, default=150, help="total comments to annotate")
    s.add_argument("--hard-frac", type=float, default=0.6,
                   help="fraction drawn from the highest-disagreement stratum")
    s.add_argument("--out", default=str(ARTIFACTS_DIR / "dev_to_annotate.csv"))
    s.add_argument("--seed", type=int, default=SEED)
    s.set_defaults(func=cmd_sample)

    f = sub.add_parser("finalise", help="validate a filled sheet -> dev_gold.csv")
    f.add_argument("--input", default=str(ARTIFACTS_DIR / "dev_to_annotate.csv"))
    f.add_argument("--ref", default=None, help="committee reference csv")
    f.add_argument("--out", default=str(ARTIFACTS_DIR / "dev_gold.csv"))
    f.set_defaults(func=cmd_finalise)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
