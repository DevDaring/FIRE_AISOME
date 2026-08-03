"""Probability-level fusion of the three SETU channels.

Replaces hard majority voting. With three channels and three classes, hard voting
throws away almost everything: a 2-1 split and a 1-1-1 split are the only signals it
sees, and it cannot express "the encoder is 0.51/0.49 between Favour and Against while
the committee is 0.95 Against". Soft fusion can.

Channels:
  A  encoder    — MuRIL/XLM-R multi-task, self-trained     (probs_setu_<lang>.csv)
  B  committee  — agreement-weighted LLM soft labels       (committee_<lang>.csv)
  C  nli        — training-free claim-conditioned NLI      (probs_nli_<lang>.csv)

Fusion is a weighted geometric mean of the (calibrated) per-channel distributions:

    log p_fused  =  sum_k  w_k * log p_k        (then renormalised)

Geometric rather than arithmetic because it behaves like a product of experts: a channel
that is *confidently* against a label can veto it, which is what we want when the NLI
channel is confident and the encoder is merely uncertain. `--mean arithmetic` is
available for the ablation table.

Weights: pass `--weights` explicitly, or `--search` to grid-search them on the gold dev
set for macro-F1 (with a simplex grid coarse enough not to overfit ~150 dev rows).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (LABELS, lang_from_path, load_dev, macro_f1, read_csv,
                    score_report, write_json)

PCOLS = [f"p_{l.lower()}" for l in LABELS]
EPS = 1e-12


def load_channel(path, name: str) -> pd.DataFrame:
    df = read_csv(path)
    missing = [c for c in PCOLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing {missing} — every channel must write "
                         f"p_favour/p_against/p_none")
    out = df[["id"] + PCOLS].copy()
    out["id"] = out["id"].astype(str)
    P = out[PCOLS].to_numpy(dtype=np.float64)
    P = np.clip(P, EPS, None)
    out[PCOLS] = P / P.sum(axis=1, keepdims=True)
    out.columns = ["id"] + [f"{name}__{c}" for c in PCOLS]
    return out


def align(channels: dict) -> tuple[pd.Series, dict]:
    """Inner-join all channels on id; returns ids and {name: (n,3) array}."""
    items = list(channels.items())
    merged = items[0][1]
    for _, df in items[1:]:
        merged = merged.merge(df, on="id", how="inner")
    ids = merged["id"].astype(str)
    mats = {name: merged[[f"{name}__{c}" for c in PCOLS]].to_numpy(dtype=np.float64)
            for name, _ in items}
    return ids, mats


def fuse(mats: dict, weights: dict, mean: str = "geometric") -> np.ndarray:
    names = list(mats)
    w = np.array([max(weights.get(n, 0.0), 0.0) for n in names], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(names))
    w = w / w.sum()
    if mean == "arithmetic":
        Q = sum(wi * mats[n] for wi, n in zip(w, names))
    else:
        L = sum(wi * np.log(np.clip(mats[n], EPS, None)) for wi, n in zip(w, names))
        L -= L.max(axis=1, keepdims=True)
        Q = np.exp(L)
    return Q / np.clip(Q.sum(axis=1, keepdims=True), EPS, None)


def search_weights(mats: dict, ids: pd.Series, dev: pd.DataFrame, mean: str,
                   step: int = 4) -> tuple[dict, float]:
    """Grid search over a coarse simplex. Coarse on purpose: ~150 dev rows cannot
    support fine-grained weight tuning without overfitting."""
    names = list(mats)
    pos = {i: k for k, i in enumerate(ids)}
    rows = [pos[i] for i in dev["id"] if i in pos]
    y = dev[dev["id"].isin(pos)]["gold"].tolist()
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} dev rows overlap the channels")
    sub = {n: m[rows] for n, m in mats.items()}

    best_w, best_s = {n: 1.0 for n in names}, -1.0
    grids = itertools.product(*[range(step + 1)] * len(names))
    for combo in grids:
        if sum(combo) == 0:
            continue
        w = {n: c / sum(combo) for n, c in zip(names, combo)}
        s = macro_f1(y, [LABELS[i] for i in fuse(sub, w, mean).argmax(1)])
        if s > best_s + 1e-9:
            best_w, best_s = w, s
    return best_w, float(best_s)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fuse the SETU channels")
    ap.add_argument("--channel", nargs="+", required=True, metavar="NAME=PATH",
                    help="e.g. encoder=artifacts/probs_setu_hi.cal.csv "
                         "committee=artifacts/committee_hi.csv "
                         "nli=artifacts/probs_nli_hi.csv")
    ap.add_argument("--weights", nargs="+", default=None, metavar="NAME=W",
                    help="e.g. encoder=0.5 committee=0.35 nli=0.15")
    ap.add_argument("--search", action="store_true",
                    help="grid-search weights on --dev for macro-F1")
    ap.add_argument("--dev", default=None, help="dev_gold.csv (id,gold)")
    ap.add_argument("--mean", default="geometric", choices=["geometric", "arithmetic"])
    ap.add_argument("--lang", default=None,
                    help="hi|bn — dev rows to use (default: inferred from --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None, help="write a JSON report here")
    args = ap.parse_args()

    channels = {}
    for spec in args.channel:
        if "=" not in spec:
            raise SystemExit(f"--channel needs NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        channels[name] = load_channel(path, name)
        print(f"  channel {name}: {path} ({len(channels[name])} rows)")

    ids, mats = align(channels)
    print(f"aligned on {len(ids)} ids across {len(mats)} channels")

    weights = {n: 1.0 for n in mats}
    if args.weights:
        weights = {}
        for spec in args.weights:
            n, v = spec.split("=", 1)
            weights[n] = float(v)
        for n in mats:
            weights.setdefault(n, 0.0)

    report = {"channels": {n: len(d) for n, d in channels.items()},
              "aligned": len(ids), "mean": args.mean}

    dev = None
    if args.dev and Path(args.dev).exists():
        lang = args.lang or lang_from_path(args.out)
        dev = load_dev(args.dev, lang=lang)
        print(f"gold dev rows for lang={lang!r}: {len(dev)}")

        # single-channel reference scores — the paper's ablation column
        pos = {i: k for k, i in enumerate(ids)}
        rows = [pos[i] for i in dev["id"] if i in pos]
        y = dev[dev["id"].isin(pos)]["gold"].tolist()
        report["single_channel_macro_f1"] = {}
        for n, m in mats.items():
            s = macro_f1(y, [LABELS[i] for i in m[rows].argmax(1)])
            report["single_channel_macro_f1"][n] = round(s, 4)
            print(f"  {n:10} alone: macro-F1 {s:.4f}")

    if args.search:
        if dev is None:
            raise SystemExit("--search needs --dev")
        weights, best = search_weights(mats, ids, dev, args.mean)
        print(f"\nsearched weights: "
              f"{ {k: round(v, 3) for k, v in weights.items()} } "
              f"-> dev macro-F1 {best:.4f}")
        report["searched_dev_macro_f1"] = round(best, 4)

    report["weights"] = weights
    Q = fuse(mats, weights, args.mean)
    out = pd.DataFrame({"id": ids})
    for i, l in enumerate(LABELS):
        out[f"p_{l.lower()}"] = Q[:, i]
    out["pred"] = [LABELS[i] for i in Q.argmax(1)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"  distribution: {out['pred'].value_counts().to_dict()}")
    report["distribution"] = out["pred"].value_counts().to_dict()

    if dev is not None:
        m = out.merge(dev[["id", "gold"]], on="id", how="inner")
        if len(m):
            rep = score_report(m["gold"].tolist(), m["pred"].tolist(),
                               title=f"FUSED ({args.mean})")
            report["fused_dev"] = rep

    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
