"""Build the unified English training pool from the two permitted sources.

Sources (both linked from the official track page):
  1. GWStance / GWSD.tsv — 2300 English news sentences, 8 crowd votes each
     (agree / disagree / neutral). We keep BOTH the majority-vote hard label and
     the **normalised vote distribution as a soft label**: 5-3 splits and 8-0
     splits are very different training signals, and the soft target is a free
     regulariser for the small `Against` class. Consumed by
     train_transformer.py's soft-label KL term.
  2. SemEval-2016 Task 6 — only Target == "Climate Change is a Real Concern"
     (395 tweets), as the track page instructs. Hard labels only; expressed as
     one-hot soft labels so the two sources share a schema.

Output: artifacts/train_en.csv with columns
    id, text, label, p_favour, p_against, p_none, n_votes, agreement, source

Note the class skew this leaves us with — SemEval-CC is FAVOR 212 / NONE 168 /
AGAINST 15. That 4 % `Against` rate is exactly why synth_generate.py exists.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from common import (ARTIFACTS_DIR, LABELS, TRAIN_DATA_DIR, normalize_label,
                    write_json)

GWSD_NAME = "GWSD.tsv"
SEMEVAL_NAME = "semeval2016-task6-trainingdata.txt"
SEMEVAL_TARGET = "Climate Change is a Real Concern"
N_WORKERS = 8


def _find(name: str) -> Path:
    """Tolerate the file living in Training_Data/ or straight in Dataset/."""
    from common import DATASET_DIR
    for cand in (TRAIN_DATA_DIR / name, DATASET_DIR / name):
        if cand.exists():
            return cand
    hits = list(DATASET_DIR.rglob(name))
    if hits:
        return hits[0]
    raise SystemExit(f"{name} not found under {DATASET_DIR}")


def load_gwsd(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            votes = []
            for w in range(N_WORKERS):
                v = (row.get(f"worker_{w}") or "").strip()
                if v:
                    try:
                        votes.append(normalize_label(v))
                    except ValueError:
                        continue
            text = (row.get("sentence") or "").strip()
            if not votes or not text:
                continue
            c = Counter(votes)
            n = len(votes)
            top, top_n = c.most_common(1)[0]
            rows.append({
                "id": f"gwsd_{row.get('guid') or i}",
                "text": text,
                "label": top,
                "p_favour": c.get("Favour", 0) / n,
                "p_against": c.get("Against", 0) / n,
                "p_none": c.get("None", 0) / n,
                "n_votes": n,
                "agreement": top_n / n,          # 1.0 = unanimous
                "source": "gwsd",
            })
    return rows


def load_semeval(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="cp1252", newline="") as f:   # the file is Windows-1252
        for row in csv.DictReader(f, delimiter="\t"):
            if (row.get("Target") or "").strip() != SEMEVAL_TARGET:
                continue
            stance = (row.get("Stance") or "").strip()
            text = (row.get("Tweet") or "").replace("#SemST", "").strip()
            if not text or not stance:
                continue
            label = normalize_label(stance)
            rows.append({
                "id": f"semeval_{row.get('ID')}",
                "text": text,
                "label": label,
                "p_favour": float(label == "Favour"),
                "p_against": float(label == "Against"),
                "p_none": float(label == "None"),
                "n_votes": 1,
                "agreement": 1.0,
                "source": "semeval",
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "train_en.csv"))
    ap.add_argument("--min-agreement", type=float, default=0.0,
                    help="drop GWSD rows whose majority share is below this "
                         "(e.g. 0.5 removes 3-3-2 style noise)")
    args = ap.parse_args()

    gwsd = load_gwsd(_find(GWSD_NAME))
    semeval = load_semeval(_find(SEMEVAL_NAME))
    df = pd.DataFrame(gwsd + semeval)

    df = df[df["text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if args.min_agreement > 0:
        df = df[df["agreement"] >= args.min_agreement].reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    stats = {
        "rows": len(df),
        "dropped_duplicates": before - len(df),
        "by_source": df["source"].value_counts().to_dict(),
        "by_label": df["label"].value_counts().to_dict(),
        "by_source_label": {s: sub["label"].value_counts().to_dict()
                            for s, sub in df.groupby("source")},
        "mean_gwsd_agreement": round(
            float(df.loc[df["source"] == "gwsd", "agreement"].mean()), 4),
        "against_share": round(float((df["label"] == "Against").mean()), 4),
    }
    write_json(ARTIFACTS_DIR / "train_en.stats.json", stats)

    print(f"Wrote {len(df)} rows -> {out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n  NOTE: only {stats['against_share']:.1%} of the pool is 'Against'. "
          f"Under macro-F1 that class is worth a third of the score —\n"
          f"        this is what synth_generate.py is for.")
    assert set(df["label"].unique()) <= set(LABELS)


if __name__ == "__main__":
    main()
