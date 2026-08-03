"""Split the judge labels into a distillation set and a held-out dev set.

Why the split is not optional
-----------------------------
The strong judge panel has now labelled 971 of the 1000 test comments, and those
labels are the best signal we have (inter-judge Fleiss kappa 0.859; the cheap
committee agrees with them 92.7 % of the time). The temptation is to distil all
971 into the encoder — but then every row available for calibration is a row the
encoder was trained on, and the calibration constants get fitted on training data.
Temperature, prior correction and per-class weights would all be tuned against
predictions the model has effectively memorised, and they would not transfer to
the rows we are actually scored on.

So: stratify by (language x label), hold out `--dev-frac`, distil the rest.

What this measures, honestly
----------------------------
The held-out set is still judge-labelled, so scoring against it measures
*agreement with the judge panel*, not truth. That is the strongest signal
available without a full human annotation pass, and the panel was spot-checked by
a human reader — but the working notes must state it plainly rather than calling
it gold.

Output
------
  artifacts/distil_{hi,bn}.csv  committee-format, for selftrain.py --committee
  artifacts/dev_holdout.csv     id,lang,text,gold  for calibrate/fuse/evaluate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ARTIFACTS_DIR, LABELS, SEED, read_csv, set_seed, write_json


def main():
    ap = argparse.ArgumentParser(description="Stratified distil/dev split")
    ap.add_argument("--judges", default=str(ARTIFACTS_DIR / "judge_all.csv"))
    ap.add_argument("--dev-frac", type=float, default=0.25,
                    help="share held out for calibration and fusion weights")
    ap.add_argument("--outdir", default=str(ARTIFACTS_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    df = read_csv(args.judges)
    df["id"] = df["id"].astype(str).str.strip()
    df = df[df["gold"].isin(LABELS)].reset_index(drop=True)
    print(f"{len(df)} judge labels: {df['gold'].value_counts().to_dict()}")

    dev_parts, distil_parts = [], []
    for (lang, lab), grp in df.groupby(["lang", "gold"]):
        n_dev = max(1, int(round(len(grp) * args.dev_frac)))
        g = grp.sample(frac=1.0, random_state=args.seed)
        dev_parts.append(g.iloc[:n_dev])
        distil_parts.append(g.iloc[n_dev:])
    dev = pd.concat(dev_parts, ignore_index=True)
    distil = pd.concat(distil_parts, ignore_index=True)

    # ---- dev set ----------------------------------------------------------
    cols = [c for c in ("id", "lang", "text", "gold", "judge_agree", "node")
            if c in dev.columns]
    dev_out = dev[cols].copy()
    dev_out["stratum"] = "judge_holdout"
    dev_path = Path(args.outdir) / "dev_holdout.csv"
    dev_out.to_csv(dev_path, index=False)

    # ---- distillation set, in the committee format selftrain.py reads -----
    for lang in ("hi", "bn"):
        sub = distil[distil["lang"] == lang]
        if not len(sub):
            continue
        # Written to satisfy BOTH consumers: selftrain.py reads committee-format
        # (`committee_label` + `agreement`), train_transformer.py reads `label`.
        # Emitting only one of them makes the file silently unusable by the other —
        # train_transformer just reports "no 'label' column" and exits.
        out = pd.DataFrame({
            "id": sub["id"], "lang": lang,
            "text": sub["text"] if "text" in sub.columns else "",
            "label": sub["gold"],
            "committee_label": sub["gold"],
            # judge_agree is the panel's internal agreement on that row; use it as
            # the per-row sample weight so split rows teach proportionally less
            "agreement": pd.to_numeric(sub.get("judge_agree", 1.0),
                                       errors="coerce").fillna(1.0),
            "node_id": sub["node"] if "node" in sub.columns else "UNK",
        })
        for lab in LABELS:
            out[f"p_{lab.lower()}"] = (out["label"] == lab).astype(float)
        out["entropy"] = 0.0
        # train_transformer.py uses `weight` as the per-row sample weight; keep it
        # in step with the committee-format `agreement` column
        out["weight"] = out["agreement"]
        dest = Path(args.outdir) / f"distil_{lang}.csv"
        out.to_csv(dest, index=False)
        print(f"  distil_{lang}.csv: {len(out)} rows "
              f"{out['committee_label'].value_counts().to_dict()}")

    stats = {
        "judge_labels": len(df), "dev_rows": len(dev), "distil_rows": len(distil),
        "dev_by_lang": dev["lang"].value_counts().to_dict(),
        "dev_by_label": dev["gold"].value_counts().to_dict(),
        "distil_by_label": distil["gold"].value_counts().to_dict(),
        "overlap_ids": 0,
    }
    # the whole point: zero overlap between what we train on and what we tune on
    dev_keys = set(dev["lang"] + ":" + dev["id"])
    distil_keys = set(distil["lang"] + ":" + distil["id"])
    stats["overlap_ids"] = len(dev_keys & distil_keys)
    write_json(Path(args.outdir) / "dev_holdout.stats.json", stats)

    print(f"\n  dev_holdout.csv: {len(dev)} rows {dev['gold'].value_counts().to_dict()}")
    print(f"  by lang: {stats['dev_by_lang']}")
    print(f"  overlap between distil and dev: {stats['overlap_ids']} "
          f"({'clean' if stats['overlap_ids'] == 0 else 'LEAK — fix before calibrating'})")
    if stats["overlap_ids"]:
        raise SystemExit("distil/dev overlap is non-zero; refusing to proceed")


if __name__ == "__main__":
    main()
