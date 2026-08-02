"""Multilingual sentence-embedding baseline: LaBSE + Logistic Regression.

Role in SETU: the **reference baseline** for the working notes' results table, and a
sanity check that finishes in minutes on CPU while the transformers train. LaBSE covers
109 languages including Hindi and Bengali, so it works zero-shot on the raw test
comments without any translation step — which makes it the honest "what do you get for
free" row that a shared-task paper needs.

It also exposes `predict_proba`, so it plugs into fuse.py as an extra channel if it
turns out to be competitive.

Usage
-----
    python3.12 src/train_baseline.py \
        --train artifacts/train_en.csv \
                artifacts/train_en.to-hi.csv::text_hi \
                artifacts/train_en.to-bn.csv::text_bn \
                artifacts/synth_train.csv \
        --out artifacts/model_baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import (ARTIFACTS_DIR, LABELS, SEED, device_str, read_csv, score_report,
                    set_seed, write_json)
from train_transformer import load_train_files          # same 'path::col' spec


def main():
    ap = argparse.ArgumentParser(description="LaBSE + LogisticRegression baseline")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--encoder", default="sentence-transformers/LaBSE",
                    help="alternatives worth an ablation row: "
                         "sentence-transformers/paraphrase-multilingual-mpnet-base-v2, "
                         "intfloat/multilingual-e5-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default=None,
                    help="artifacts/dev_gold.csv — the honest macro-F1 number")
    ap.add_argument("--dev-text-col", default="text")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    set_seed(args.seed)
    print(f"device: {device_str()}")

    df = load_train_files(args.train, args.text_col)
    if args.max_train:
        df = df.sample(min(args.max_train, len(df)), random_state=args.seed)
    print(f"pool: {len(df)} rows {df['label'].value_counts().to_dict()}")

    if args.dev_frac > 0 and len(df) > 50:
        tr, va = train_test_split(df, test_size=args.dev_frac,
                                  stratify=df["label"], random_state=args.seed)
    else:
        tr, va = df, None

    print(f"loading encoder {args.encoder} ...")
    enc = SentenceTransformer(args.encoder, device=device_str())
    Xtr = enc.encode(tr["text"].astype(str).tolist(), batch_size=args.batch,
                     show_progress_bar=True, convert_to_numpy=True,
                     normalize_embeddings=True)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=args.C,
                             n_jobs=-1, random_state=args.seed)
    clf.fit(Xtr, tr["label"])

    result = {"encoder": args.encoder, "train_rows": len(tr), "C": args.C,
              "train_files": args.train}
    if va is not None:
        Xva = enc.encode(va["text"].astype(str).tolist(), batch_size=args.batch,
                         show_progress_bar=True, convert_to_numpy=True,
                         normalize_embeddings=True)
        rep = score_report(va["label"].tolist(), clf.predict(Xva).tolist(),
                           title="internal dev (same distribution as training)")
        result["internal_dev"] = rep

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "encoder": args.encoder, "labels": LABELS},
                out / "baseline.joblib")

    if args.dev and Path(args.dev).exists():
        gold = read_csv(args.dev)
        gold = gold[gold["gold"].isin(LABELS)]
        if len(gold):
            Xg = enc.encode(gold[args.dev_text_col].astype(str).tolist(),
                            batch_size=args.batch, convert_to_numpy=True,
                            normalize_embeddings=True)
            rep = score_report(gold["gold"].tolist(), clf.predict(Xg).tolist(),
                               title=f"GOLD DEV — LaBSE baseline")
            result["gold_dev"] = rep

    write_json(out / "metrics.json", result)
    print("\n" + json.dumps(result, indent=2))
    print(f"saved -> {out / 'baseline.joblib'}")
    print("predict with:  python3.12 src/predict.py --model "
          f"{out / 'baseline.joblib'} --test artifacts/test_hi.csv --out ...")


if __name__ == "__main__":
    main()
