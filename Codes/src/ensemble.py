"""Hard majority vote — kept ONLY as the ablation baseline for the working notes.

`fuse.py` supersedes this for the actual submission: with three channels over three
classes, hard voting can only distinguish "3-0", "2-1" and "1-1-1", so it discards the
confidence information that macro-F1 calibration depends on. This script exists so the
paper can report *how much* soft geometric fusion buys over the majority vote every
other team will use.

Usage
-----
    python3.12 src/ensemble.py --preds artifacts/probs_setu_hi.csv \
        artifacts/committee_hi.csv artifacts/probs_nli_hi.csv \
        --out artifacts/probs_hardvote_hi.csv [--dev artifacts/dev_gold.csv]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from common import LABELS, normalize_label, read_csv, read_table, score_report

PCOLS = [f"p_{l.lower()}" for l in LABELS]


def load_labels(path: str) -> pd.Series:
    df = read_table(path)
    if "id" not in df.columns:
        raise SystemExit(f"{path}: no `id` column")
    ids = df["id"].astype(str)
    if all(c in df.columns for c in PCOLS):
        labels = [LABELS[i] for i in df[PCOLS].to_numpy().argmax(axis=1)]
    else:
        col = next((c for c in ("pred", "label", "Label", "committee_label", "stance")
                    if c in df.columns), None)
        if col is None:
            raise SystemExit(f"{path}: no probabilities and no label column")
        labels = [normalize_label(v, default="None") for v in df[col]]
    return pd.Series(labels, index=ids)


def main():
    ap = argparse.ArgumentParser(description="Hard majority vote (ablation baseline)")
    ap.add_argument("--preds", nargs="+", required=True,
                    help="prediction files; the FIRST breaks 3-way ties")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default=None, help="artifacts/dev_gold.csv")
    args = ap.parse_args()

    series = []
    for p in args.preds:
        s = load_labels(p)
        print(f"  {p}: {len(s)} rows {s.value_counts().to_dict()}")
        series.append(s)

    ids = series[0].index
    for s in series[1:]:
        common = ids.intersection(s.index)
        if len(common) != len(ids):
            print(f"  note: intersecting ids {len(ids)} -> {len(common)}")
        ids = common

    preds, n_unanimous, n_tie = [], 0, 0
    for i in ids:
        votes = [s.loc[i] for s in series]
        top, cnt = Counter(votes).most_common(1)[0]
        if cnt == len(votes):
            n_unanimous += 1
        elif cnt == 1:                 # complete disagreement -> first model wins
            n_tie += 1
            top = votes[0]
        preds.append(top)

    out = pd.DataFrame({"id": list(ids), "pred": preds})
    for lab in LABELS:                 # one-hot, so downstream tools can consume it
        out[f"p_{lab.lower()}"] = (out["pred"] == lab).astype(float)
    out = out[["id"] + PCOLS + ["pred"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nunanimous: {n_unanimous}/{len(ids)} | 3-way ties broken by "
          f"{args.preds[0]}: {n_tie}")
    print(f"distribution: {out['pred'].value_counts().to_dict()}")
    print(f"wrote -> {args.out}")

    if args.dev and Path(args.dev).exists():
        dev = read_csv(args.dev)
        dev["id"] = dev["id"].astype(str)
        m = out.merge(dev[dev["gold"].isin(LABELS)][["id", "gold"]], on="id")
        if len(m):
            score_report(m["gold"].tolist(), m["pred"].tolist(),
                         title="hard majority vote (ablation)")


if __name__ == "__main__":
    main()
