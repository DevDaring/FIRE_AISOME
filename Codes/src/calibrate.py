"""Macro-F1-optimal decision rules (SETU contribution C4, part 4).

The cheapest points on the table, and almost nobody in a shared task collects them.

`argmax p(y|x)` maximises *accuracy*. The track is scored by **macro-F1**, which weights
a ~10 % `Against` class as heavily as a ~45 % `Favour` class. Under that metric the
Bayes-optimal decision rule is not argmax — it is argmax over *reweighted* scores, and
the reweighting depends on the test-time class priors. Three stages, each independently
useful and each an ablation row in the working notes:

1. **Temperature scaling** (Guo et al. 2017). One scalar per channel, fitted on the gold
   dev set by NLL. Fine-tuned transformers are badly over-confident; ensembling
   over-confident probabilities lets one channel shout down the others.

2. **Prior-shift correction** (Saerens, Latinne & Decaestecker 2002). Our training pool
   is synthetic-balanced and translated; the test set is not. The EM procedure
   re-estimates the *test* priors from the unlabelled predictions themselves and
   divides them out of the training priors. No labels required — it is the principled
   version of "the model predicts too few Against".

3. **Per-class weight search.** Grid/coordinate search over per-class multiplicative
   weights `w` maximising macro-F1 of `argmax_c w_c * p_c` on the gold dev set. Directly
   optimises the competition metric instead of a surrogate.

Fitted parameters are saved as JSON and applied to unseen probability frames, so the
same transform is reproducible across runs — important, because the organizers may ask
us to reproduce a submitted run.

Usage
-----
    # fit on dev, save the transform
    python3.12 src/calibrate.py fit --probs artifacts/probs_setu_hi.csv \
        --dev artifacts/dev_gold.csv --out artifacts/calib_setu_hi.json

    # apply to the full test frame
    python3.12 src/calibrate.py apply --probs artifacts/probs_setu_hi.csv \
        --calib artifacts/calib_setu_hi.json --out artifacts/probs_setu_hi.cal.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import LABELS, macro_f1, read_csv, score_report, write_json

PCOLS = [f"p_{l.lower()}" for l in LABELS]
EPS = 1e-12


# ---------------------------------------------------------------------------
def load_probs(path) -> tuple[pd.Series, np.ndarray]:
    df = read_csv(path)
    missing = [c for c in PCOLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing probability columns {missing}")
    P = df[PCOLS].to_numpy(dtype=np.float64)
    P = np.clip(P, EPS, 1.0)
    P /= P.sum(axis=1, keepdims=True)
    return df["id"].astype(str), P


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---- 1. temperature scaling -------------------------------------------------
def fit_temperature(P: np.ndarray, y: np.ndarray,
                    grid=None) -> float:
    """Scalar T minimising NLL of softmax(log P / T). T>1 softens, T<1 sharpens."""
    logits = np.log(np.clip(P, EPS, 1.0))
    grid = grid if grid is not None else np.concatenate(
        [np.arange(0.25, 1.0, 0.05), np.arange(1.0, 6.05, 0.1)])
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        p = _softmax(logits / T)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean()
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
    return best_T


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    return _softmax(np.log(np.clip(P, EPS, 1.0)) / max(T, 1e-3))


# ---- 2. prior shift (Saerens-Latinne-Decaestecker EM) -----------------------
def em_prior(P: np.ndarray, train_prior: np.ndarray, iters: int = 100,
             tol: float = 1e-7, floor: float = 0.03,
             damping: float = 0.5) -> np.ndarray:
    """Estimate the test-set class prior from unlabelled predicted probabilities.

    Two guards on top of the textbook procedure, both learned the hard way on this
    task: an under-confident model that rarely predicts `Against` sends vanilla EM to
    prior(Against) = 0, which then *removes the class entirely* and destroys macro-F1.
    So we (a) floor every prior at `floor` and (b) damp each update. Neither changes
    the fixed point when the estimate is well behaved; both prevent the collapse.
    """
    k = P.shape[1]
    floor = min(floor, 1.0 / (2 * k))
    train_prior = np.clip(np.asarray(train_prior, dtype=np.float64), EPS, None)
    train_prior /= train_prior.sum()
    prior = train_prior.copy()
    for _ in range(iters):
        w = prior / train_prior
        num = P * w
        post = num / np.clip(num.sum(axis=1, keepdims=True), EPS, None)
        new = post.mean(axis=0)
        new = (1 - damping) * prior + damping * new       # damped update
        new = np.clip(new, floor, None)                   # no class may vanish
        new /= new.sum()
        if np.abs(new - prior).max() < tol:
            prior = new
            break
        prior = new
    return prior


def apply_prior_shift(P: np.ndarray, train_prior, test_prior) -> np.ndarray:
    w = np.clip(np.asarray(test_prior, float), EPS, None) / \
        np.clip(np.asarray(train_prior, float), EPS, None)
    Q = P * w
    return Q / np.clip(Q.sum(axis=1, keepdims=True), EPS, None)


# ---- 3. per-class weight search --------------------------------------------
def fit_class_weights(P: np.ndarray, y_true: list[str], coarse: int = 13,
                      rounds: int = 3) -> tuple[list[float], float]:
    """Coordinate-ascent search for w maximising macro-F1 of argmax(w * p).

    Only ratios matter, so w_Favour is pinned to 1.0 and the other two classes are
    searched on a log grid, refined around the incumbent for `rounds` passes.
    """
    def score(w):
        pred = [LABELS[i] for i in (P * np.asarray(w)).argmax(axis=1)]
        return macro_f1(y_true, pred)

    w = [1.0, 1.0, 1.0]
    best = score(w)
    lo, hi = 0.2, 6.0
    for r in range(rounds):
        grid = np.exp(np.linspace(np.log(lo), np.log(hi), coarse))
        for ci in (1, 2):                      # Against, None (Favour pinned)
            local_best, local_w = best, w[ci]
            for g in grid:
                cand = list(w)
                cand[ci] = float(g)
                s = score(cand)
                if s > local_best + 1e-9:
                    local_best, local_w = s, float(g)
            w[ci] = local_w
            best = local_best
        # refine the bracket around the incumbent
        centre = max(w[1], w[2])
        lo, hi = max(0.05, centre / 2.5), centre * 2.5
    return w, float(best)


def apply_class_weights(P: np.ndarray, w) -> np.ndarray:
    Q = P * np.asarray(w, dtype=np.float64)
    return Q / np.clip(Q.sum(axis=1, keepdims=True), EPS, None)


# ---------------------------------------------------------------------------
def transform(P: np.ndarray, calib: dict) -> np.ndarray:
    Q = P
    if calib.get("temperature"):
        Q = apply_temperature(Q, calib["temperature"])
    if calib.get("train_prior") and calib.get("test_prior"):
        Q = apply_prior_shift(Q, calib["train_prior"], calib["test_prior"])
    if calib.get("class_weights"):
        Q = apply_class_weights(Q, calib["class_weights"])
    return Q


def decide(P: np.ndarray) -> list[str]:
    return [LABELS[i] for i in P.argmax(axis=1)]


# ---------------------------------------------------------------------------
def cmd_fit(args):
    ids, P = load_probs(args.probs)
    dev = read_csv(args.dev)
    dev["id"] = dev["id"].astype(str)
    dev = dev[dev["gold"].isin(LABELS)]
    idx = {i: k for k, i in enumerate(ids)}
    mask = [idx[i] for i in dev["id"] if i in idx]
    dev = dev[dev["id"].isin(idx)].reset_index(drop=True)
    if len(mask) < 20:
        raise SystemExit(f"only {len(mask)} dev rows overlap {args.probs} — "
                         f"cannot calibrate reliably")
    Pd = P[mask]
    y_true = dev["gold"].tolist()
    y_idx = np.array([LABELS.index(l) for l in y_true])
    print(f"calibrating on {len(mask)} gold dev rows "
          f"{dev['gold'].value_counts().to_dict()}")

    base = score_report(y_true, decide(Pd), title="BEFORE calibration (argmax)")
    calib: dict = {"source": str(args.probs), "n_dev": len(mask),
                   "before_macro_f1": base["macro_f1"]}
    Q = Pd
    cur = base["macro_f1"]

    # Stages are accepted GREEDILY: each is kept only if it does not lower dev
    # macro-F1. Applying them unconditionally is how calibration ends up hurting —
    # EM prior correction in particular can misfire badly on an under-confident model,
    # and there is no reason to pay for a stage that does not earn its place.
    calib["stages"] = []

    def try_stage(name, params, Q_new, min_gain=0.0):
        nonlocal Q, cur
        s = macro_f1(y_true, decide(Q_new))
        keep = s >= cur + min_gain
        print(f"  {'KEEP  ' if keep else 'REJECT'} {name:16} macro-F1 "
              f"{cur:.4f} -> {s:.4f}")
        calib["stages"].append({"stage": name, "macro_f1": round(s, 4),
                                "kept": bool(keep)})
        if keep:
            calib.update(params)
            Q, cur = Q_new, s
        return keep

    print(f"\ngreedy stage selection (baseline macro-F1 {cur:.4f}):")

    if not args.no_temperature:
        T = fit_temperature(Q, y_idx)
        try_stage(f"temperature", {"temperature": T}, apply_temperature(Q, T))

    if not args.no_prior_shift:
        train_prior = (np.asarray([float(x) for x in args.train_prior])
                       if args.train_prior
                       else np.full(len(LABELS), 1.0 / len(LABELS)))
        train_prior = train_prior / train_prior.sum()
        # estimate the prior from the FULL unlabelled test frame, not just the dev slice
        P_all = (apply_temperature(P, calib["temperature"])
                 if calib.get("temperature") else P)
        test_prior = em_prior(P_all, train_prior)
        print(f"  EM test prior = "
              f"{ {l: round(float(p), 3) for l, p in zip(LABELS, test_prior)} }")
        try_stage("prior_shift",
                  {"train_prior": train_prior.tolist(),
                   "test_prior": test_prior.tolist()},
                  apply_prior_shift(Q, train_prior, test_prior))

    if not args.no_class_weights:
        w, _ = fit_class_weights(Q, y_true)
        print(f"  searched weights = "
              f"{ {l: round(x, 3) for l, x in zip(LABELS, w)} }")
        # require a real gain: this stage has 2 free parameters fitted on ~150 rows,
        # so it is the one most likely to be fitting noise
        try_stage("class_weights", {"class_weights": w},
                  apply_class_weights(Q, w), min_gain=args.min_weight_gain)

    after = score_report(y_true, decide(Q), title="AFTER calibration")
    calib["after_macro_f1"] = after["macro_f1"]
    calib["gain"] = round(after["macro_f1"] - base["macro_f1"], 4)
    write_json(args.out, calib)
    print(f"\nmacro-F1 {base['macro_f1']:.4f} -> {after['macro_f1']:.4f} "
          f"({calib['gain']:+.4f})")
    print(f"wrote {args.out}")
    if calib["gain"] < 0:
        print("  WARNING: calibration hurt on dev — the dev set is probably too "
              "small. Consider --no-class-weights, which overfits the fastest.")


def cmd_apply(args):
    ids, P = load_probs(args.probs)
    calib = json.loads(Path(args.calib).read_text())
    Q = transform(P, calib)
    out = pd.DataFrame({"id": ids})
    for i, l in enumerate(LABELS):
        out[f"p_{l.lower()}"] = Q[:, i]
    out["pred"] = decide(Q)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    before = pd.Series(decide(P)).value_counts().to_dict()
    print(f"wrote {args.out}")
    print(f"  before: {before}")
    print(f"  after : {out['pred'].value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser(description="Macro-F1-optimal calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit a calibration transform on the gold dev set")
    f.add_argument("--probs", required=True, help="probability frame for one language")
    f.add_argument("--dev", required=True, help="dev_gold.csv (id,gold)")
    f.add_argument("--out", required=True)
    f.add_argument("--train-prior", nargs="+", default=None,
                   help="training class prior in LABELS order (default: uniform, "
                        "which is right after balanced synthetic augmentation)")
    f.add_argument("--no-temperature", action="store_true")
    f.add_argument("--no-prior-shift", action="store_true")
    f.add_argument("--no-class-weights", action="store_true")
    f.add_argument("--min-weight-gain", type=float, default=0.01,
                   help="dev macro-F1 gain required to keep the class-weight stage; "
                        "it has the most free parameters and overfits the fastest")
    f.set_defaults(func=cmd_fit)

    a = sub.add_parser("apply", help="apply a saved transform")
    a.add_argument("--probs", required=True)
    a.add_argument("--calib", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
