"""Pick the fusion strategy by HELD-OUT performance, not in-sample score.

Why this file exists
--------------------
The 5-channel weight search in fuse.py scored hi 0.8252 / bn 0.8158 on the dev
set. Under a repeated half-split — search the weights on one half, score them on
the other — Hindi fell to 0.744. Roughly 0.11 of that headline was the search
fitting 3,125 weight combinations to 123 rows.

With ~120 dev rows per language, *how* you combine channels has to be chosen the
same way any other hyperparameter would be: on data not used to choose it. This
module enumerates a small set of combination rules, scores each under repeated
half-splits, and reports the winner per language.

It deliberately does NOT modify the submission. Replacing a validated submission
is a judgement call; this only produces the evidence for one.

    python3.12 src/select_strategy.py [--repeats 8]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from common import (ARTIFACTS_DIR, LABELS, SEED, load_dev, macro_f1, read_csv,
                    write_json)

PCOLS = [f"p_{l.lower()}" for l in LABELS]

# every channel we might have; missing files are skipped silently
CANDIDATES = {
    "distil": "probs_distil_{L}.cal.csv",
    "stage2": "probs_stage2_{L}.csv",
    "refit":  "probs_refit_{L}.csv",
    "seed2":  "probs_seed2_{L}.cal.csv",
    "indic":  "probs_indic_{L}.cal.csv",
    "setu":   "probs_setu_{L}.cal.csv",
    "xlmr":   "probs_xlmr_{L}.cal.csv",
    "proj":   "probs_proj_{L}.cal.csv",
    "nli":    "probs_nli_{L}.cal.csv",
}


def _load(path: Path) -> dict:
    df = read_csv(path)
    df["id"] = df["id"].astype(str)
    P = np.clip(df[PCOLS].to_numpy(dtype=float), 1e-9, None)
    return dict(zip(df["id"], P / P.sum(axis=1, keepdims=True)))


def fuse(mats: dict, w: dict) -> np.ndarray:
    ws = np.array([max(w.get(k, 0.0), 0.0) for k in mats])
    ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(mats)) / len(mats)
    L = sum(wi * np.log(np.clip(mats[k], 1e-9, None)) for wi, k in zip(ws, mats))
    L -= L.max(axis=1, keepdims=True)
    Q = np.exp(L)
    return Q / Q.sum(axis=1, keepdims=True)


def score(mats: dict, w: dict, y) -> float:
    return macro_f1(y, [LABELS[i] for i in fuse(mats, w).argmax(1)])


def search(mats: dict, y, step: int) -> dict:
    best, bw = -1.0, None
    for combo in itertools.product(range(step + 1), repeat=len(mats)):
        if sum(combo) == 0:
            continue
        w = {k: c / sum(combo) for k, c in zip(mats, combo)}
        s = score(mats, w, y)
        if s > best:
            best, bw = s, w
    return bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=str(ARTIFACTS_DIR / "dev_holdout.csv"))
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--artifacts", default=str(ARTIFACTS_DIR))
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "strategy_choice.json"))
    args = ap.parse_args()

    A = Path(args.artifacts)
    report = {"repeats": args.repeats, "languages": {}}

    for lang in ("hi", "bn"):
        ch = {}
        for name, tmpl in CANDIDATES.items():
            p = A / tmpl.format(L=lang)
            if p.exists():
                ch[name] = _load(p)
        if len(ch) < 2:
            print(f"{lang}: only {len(ch)} channel(s); nothing to select")
            continue

        dev = load_dev(args.dev, lang=lang)
        ids = [i for i in dev["id"] if all(i in c for c in ch.values())]
        if len(ids) < 40:
            print(f"{lang}: only {len(ids)} usable dev rows; skipping")
            continue
        y = np.array(dev.set_index("id").loc[ids, "gold"].tolist())
        mats = {k: np.vstack([ch[k][i] for i in ids]) for k in ch}
        solo = {k: score({k: mats[k]}, {k: 1}, y) for k in mats}
        ranked = [k for k, _ in sorted(solo.items(), key=lambda kv: -kv[1])]

        print(f"\n=== {lang}: {len(ch)} channels, {len(ids)} dev rows ===")
        print("  solo: " + ", ".join(f"{k}={solo[k]:.3f}" for k in ranked))

        strategies = {
            "single_best":   (ranked[:1], "equal"),
            "equal_top2":    (ranked[:2], "equal"),
            "equal_top3":    (ranked[:3], "equal"),
            "equal_top4":    (ranked[:4], "equal"),
            "equal_all":     (ranked,      "equal"),
            "search_top3":   (ranked[:3], "search2"),
            "search_top4":   (ranked[:4], "search2"),
        }

        rng = np.random.default_rng(SEED)
        results = {}
        for name, (keys, mode) in strategies.items():
            if len(keys) > len(mats):
                continue
            outs = []
            for _ in range(args.repeats):
                perm = rng.permutation(len(ids))
                a, b = perm[:len(ids) // 2], perm[len(ids) // 2:]
                sub_a = {k: mats[k][a] for k in keys}
                sub_b = {k: mats[k][b] for k in keys}
                w = ({k: 1 for k in keys} if mode == "equal"
                     else search(sub_a, y[a], 2))
                outs.append(score(sub_b, w, y[b]))
            results[name] = {"held_out_mean": round(float(np.mean(outs)), 4),
                             "held_out_std": round(float(np.std(outs)), 4),
                             "channels": keys}
        order = sorted(results.items(), key=lambda kv: -kv[1]["held_out_mean"])
        for n, r in order:
            print(f"  {n:14} held-out {r['held_out_mean']:.4f} "
                  f"+/-{r['held_out_std']:.3f}  {r['channels']}")
        win, wr = order[0]
        print(f"  -> RECOMMEND {win}: {wr['channels']} "
              f"(held-out {wr['held_out_mean']:.4f})")
        report["languages"][lang] = {
            "solo": {k: round(v, 4) for k, v in solo.items()},
            "strategies": results, "recommended": win,
            "recommended_channels": wr["channels"],
            "recommended_held_out": wr["held_out_mean"],
        }

    write_json(args.out, report)
    print(f"\nwrote {args.out}")
    print("This is a RECOMMENDATION only. The submission was not modified.")


if __name__ == "__main__":
    main()
