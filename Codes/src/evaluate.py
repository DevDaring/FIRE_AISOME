"""Score any prediction/probability frame against the hand-annotated dev set.

One place to answer "did that change help?", and the source of the working-notes
results table. Given several frames it prints a ranked comparison and dumps the
disagreements between the best system and the gold labels for error analysis — which is
where the working notes' qualitative section comes from.

Usage
-----
    python3.12 src/evaluate.py --dev artifacts/dev_gold.csv \
        --pred encoder=artifacts/probs_setu_hi.csv \
               committee=artifacts/committee_hi.csv \
               nli=artifacts/probs_nli_hi.csv \
               fused=artifacts/probs_fused_hi.csv \
        --errors artifacts/error_analysis.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import (LABELS, normalize_label, read_csv, read_table, score_report,
                    write_json)

PCOLS = [f"p_{l.lower()}" for l in LABELS]


def load_pred(path: str) -> pd.DataFrame:
    df = read_table(path)
    if "id" not in df.columns:
        raise SystemExit(f"{path}: no `id` column")
    out = pd.DataFrame({"id": df["id"].astype(str)})
    if all(c in df.columns for c in PCOLS):
        out["pred"] = [LABELS[i] for i in df[PCOLS].to_numpy().argmax(axis=1)]
        out["conf"] = df[PCOLS].to_numpy().max(axis=1)
    else:
        col = next((c for c in ("pred", "label", "Label", "committee_label", "stance")
                    if c in df.columns), None)
        if col is None:
            raise SystemExit(f"{path}: no probabilities and no label column "
                             f"(has {list(df.columns)})")
        out["pred"] = [normalize_label(v, default="None") for v in df[col]]
        out["conf"] = df["agreement"] if "agreement" in df.columns else 1.0
    return out


def main():
    ap = argparse.ArgumentParser(description="Score frames against the gold dev set")
    ap.add_argument("--dev", required=True, help="artifacts/dev_gold.csv")
    ap.add_argument("--pred", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--by-lang", action="store_true",
                    help="also break every score down by language")
    ap.add_argument("--errors", default=None,
                    help="write the best system's mistakes here for error analysis")
    ap.add_argument("--report", default=None, help="write a JSON summary here")
    args = ap.parse_args()

    dev = read_csv(args.dev)
    dev["id"] = dev["id"].astype(str)
    dev = dev[dev["gold"].isin(LABELS)].reset_index(drop=True)
    print(f"gold dev: {len(dev)} rows {dev['gold'].value_counts().to_dict()}")
    if "lang" in dev.columns:
        print(f"  by language: {dev['lang'].value_counts().to_dict()}")

    results, frames = {}, {}
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit(f"--pred needs NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if not Path(path).exists():
            print(f"  SKIP (missing): {name} -> {path}")
            continue
        pr = load_pred(path)
        m = dev.merge(pr, on="id", how="inner")
        if len(m) < 10:
            print(f"  SKIP ({name}): only {len(m)} rows overlap the dev set")
            continue
        rep = score_report(m["gold"].tolist(), m["pred"].tolist(),
                           title=f"{name}  (n={len(m)}, {path})")
        rep["source"] = path
        if args.by_lang and "lang" in m.columns:
            rep["by_lang"] = {}
            for lang, sub in m.groupby("lang"):
                if len(sub) >= 10:
                    from common import macro_f1
                    rep["by_lang"][lang] = {
                        "n": len(sub),
                        "macro_f1": round(macro_f1(sub["gold"], sub["pred"]), 4)}
            print(f"  by language: {rep['by_lang']}")
        results[name] = rep
        frames[name] = m

    if not results:
        raise SystemExit("nothing scored")

    ranked = sorted(results.items(), key=lambda kv: -kv[1]["macro_f1"])
    print(f"\n{'='*72}\nRANKING by macro-F1 on {len(dev)} hand-annotated comments\n")
    print(f"  {'system':22} {'macro-F1':>9}  {'Favour':>8} {'Against':>8} {'None':>8}"
          f"  {'acc':>6}")
    for name, rep in ranked:
        pc = rep["per_class_f1"]
        print(f"  {name:22} {rep['macro_f1']:>9.4f}  {pc['Favour']:>8.3f} "
              f"{pc['Against']:>8.3f} {pc['None']:>8.3f}  {rep['accuracy']:>6.3f}")

    best = ranked[0][0]
    print(f"\nbest: {best}")
    worst_class = min(ranked[0][1]["per_class_f1"],
                      key=lambda k: ranked[0][1]["per_class_f1"][k])
    print(f"weakest class for the best system: {worst_class} "
          f"(F1 {ranked[0][1]['per_class_f1'][worst_class]:.3f}) — under macro-F1 this "
          f"is where the remaining points are.")

    if args.errors:
        m = frames[best]
        err = m[m["gold"] != m["pred"]].copy()
        err["system"] = best
        cols = [c for c in ("id", "lang", "stratum", "text", "gold", "pred", "conf",
                            "note") if c in err.columns] + ["system"]
        Path(args.errors).parent.mkdir(parents=True, exist_ok=True)
        err[cols].to_csv(args.errors, index=False)
        print(f"\nwrote {len(err)} errors -> {args.errors}")
        print("  most common confusions (gold -> pred):")
        for (g, p), n in err.groupby(["gold", "pred"]).size().sort_values(
                ascending=False).head(6).items():
            print(f"    {g:8} -> {p:8}  {n}")

    if args.report:
        write_json(args.report, {"dev_rows": len(dev), "results": results,
                                 "ranking": [n for n, _ in ranked]})
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
