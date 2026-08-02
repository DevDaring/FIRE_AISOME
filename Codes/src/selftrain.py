"""Agreement-weighted transductive self-training (SETU contribution C4, part 2).

Idea
----
The encoder trained on translated-English + synthetic data has never seen a real
Hindi/Bengali YouTube comment. The committee has seen all 1000 of them but is expensive,
opaque and not submittable as a reproducible artefact. So we **distil the committee onto
the test distribution**, then let the student bootstrap itself:

    round 0:  train on {translated EN, synthetic}
              + committee pseudo-labels on the test set, weighted by agreement
    round r:  predict the test set; absorb items the student is now confident about
              (and that do not contradict a unanimous committee); retrain

Two guards against the classic self-training failure — confirmation bias collapsing the
minority class:

  1. **Per-class quotas.** New pseudo-labels are admitted per class in proportion to the
     committee's estimated prior, so `Against` cannot be starved out even though it is
     the class the student is least confident about. This is the difference between
     self-training that helps macro-F1 and self-training that destroys it.
  2. **Committee veto.** A student prediction is never absorbed if the committee was
     unanimous for a different label.

Sample weights combine committee agreement and student confidence, so uncertain rows
contribute proportionally less rather than being all-or-nothing.

Everything here is transduction on *unlabelled* data — no gold label is read at any
point. The hand-annotated dev set (annotate_dev.py) is used only to report macro-F1
per round, never to train.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ARTIFACTS_DIR, LABELS, SEED, read_csv, score_report, set_seed,
                    write_json)

SOFT_COLS = ["p_favour", "p_against", "p_none"]


# ---------------------------------------------------------------------------
def committee_pseudo_labels(committee_csvs: list[str], min_agreement: float,
                            text_col: str = "text") -> pd.DataFrame:
    """Rows from committee_<lang>.csv confident enough to train on."""
    frames = []
    for path in committee_csvs:
        p = Path(path)
        if not p.exists():
            print(f"  SKIP (missing): {p}")
            continue
        df = read_csv(p)
        lang = "bn" if "bn" in p.stem.lower() else "hi"
        keep = df[(df["agreement"] >= min_agreement)
                  & df["committee_label"].isin(LABELS)].copy()
        out = pd.DataFrame({
            "text": keep[text_col].astype(str),
            "label": keep["committee_label"],
            "node_id": keep.get("node_id", "UNK"),
            "weight": keep["agreement"].astype(float),
            "lang": lang,
            "src_id": keep["id"].astype(str),
        })
        for c in SOFT_COLS:
            out[c] = keep[c].astype(float) if c in keep.columns else np.nan
        need = out[SOFT_COLS].isna().any(axis=1)
        for c, lab in zip(SOFT_COLS, LABELS):
            out.loc[need, c] = (out.loc[need, "label"] == lab).astype(float)
        frames.append(out)
        print(f"  {p.name}: {len(out)}/{len(df)} rows at agreement >= {min_agreement} "
              f"{out['label'].value_counts().to_dict()}")
    if not frames:
        return pd.DataFrame(columns=["text", "label", "node_id", "weight", "lang",
                                     "src_id"] + SOFT_COLS)
    return pd.concat(frames, ignore_index=True)


def student_pseudo_labels(prob_csvs: list[str], test_csvs: list[str],
                          committee_csvs: list[str], threshold: float,
                          prior: dict, budget: int, text_col: str = "text",
                          seed: int = SEED) -> pd.DataFrame:
    """Confident student predictions, admitted under per-class quotas + committee veto."""
    rows = []
    for prob_path, test_path in zip(prob_csvs, test_csvs):
        if not (Path(prob_path).exists() and Path(test_path).exists()):
            continue
        pr = read_csv(prob_path)
        te = read_csv(test_path)
        pr["id"] = pr["id"].astype(str)
        te["id"] = te["id"].astype(str)
        m = te[["id", text_col]].merge(pr, on="id", how="inner")
        lang = "bn" if "bn" in Path(test_path).stem.lower() else "hi"

        veto = {}
        cpath = next((c for c in committee_csvs
                      if lang in Path(c).stem.lower() and Path(c).exists()), None)
        if cpath:
            cdf = read_csv(cpath)
            cdf["id"] = cdf["id"].astype(str)
            unan = cdf[cdf["agreement"] >= 0.999]
            veto = dict(zip(unan["id"], unan["committee_label"]))

        conf = m[SOFT_COLS].to_numpy().max(axis=1)
        pred = [LABELS[i] for i in m[SOFT_COLS].to_numpy().argmax(axis=1)]
        for j in range(len(m)):
            if conf[j] < threshold:
                continue
            cid = m["id"].iloc[j]
            if cid in veto and veto[cid] != pred[j]:
                continue                      # committee veto
            rows.append({
                "text": str(m[text_col].iloc[j]),
                "label": pred[j],
                "node_id": "UNK",
                "weight": float(conf[j]),
                "lang": lang,
                "src_id": cid,
                **{c: float(m[c].iloc[j]) for c in SOFT_COLS},
            })
    if not rows:
        return pd.DataFrame(columns=["text", "label", "node_id", "weight", "lang",
                                     "src_id"] + SOFT_COLS)

    df = pd.DataFrame(rows)
    # per-class quota proportional to the estimated prior — protects `Against`
    quotas = {lab: max(1, int(round(budget * prior.get(lab, 1 / 3)))) for lab in LABELS}
    kept = []
    for lab in LABELS:
        sub = df[df["label"] == lab].sort_values("weight", ascending=False)
        kept.append(sub.head(quotas[lab]))
        print(f"    {lab:8}: {len(sub):4} available, quota {quotas[lab]:4}, "
              f"taking {min(len(sub), quotas[lab])}")
    return pd.concat(kept, ignore_index=True).sample(frac=1.0, random_state=seed)


def estimate_prior(committee_csvs: list[str]) -> dict:
    """Class prior of the test set, from the committee's soft labels."""
    frames = [read_csv(p) for p in committee_csvs if Path(p).exists()]
    if not frames:
        return {l: 1 / len(LABELS) for l in LABELS}
    df = pd.concat(frames, ignore_index=True)
    if all(c in df.columns for c in SOFT_COLS):
        m = df[SOFT_COLS].to_numpy().mean(axis=0)
    else:
        vc = df["committee_label"].value_counts(normalize=True)
        m = np.array([vc.get(l, 0.0) for l in LABELS])
    m = m / max(m.sum(), 1e-9)
    return {lab: float(m[i]) for i, lab in enumerate(LABELS)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Transductive self-training / committee distillation")
    ap.add_argument("--base-train", nargs="+", required=True,
                    help="the static pool, e.g. artifacts/train_en.csv "
                         "artifacts/train_en.to-hi.csv::text_hi "
                         "artifacts/synth_train.csv")
    ap.add_argument("--committee", nargs="+", required=True,
                    help="artifacts/committee_hi.csv artifacts/committee_bn.csv")
    ap.add_argument("--test", nargs="+", required=True,
                    help="artifacts/test_hi.csv artifacts/test_bn.csv")
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "model_setu"))
    ap.add_argument("--rounds", type=int, default=2,
                    help="self-training rounds after round 0")
    ap.add_argument("--min-agreement", type=float, default=0.6,
                    help="committee agreement needed to pseudo-label a row")
    ap.add_argument("--student-threshold", type=float, default=0.9)
    ap.add_argument("--budget", type=int, default=400,
                    help="max new student pseudo-labels absorbed per round")
    ap.add_argument("--dev", default=str(ARTIFACTS_DIR / "dev_gold.csv"),
                    help="hand-annotated gold dev set; reported, never trained on")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--soft-alpha", type=float, default=0.4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    src = Path(__file__).resolve().parent
    outroot = Path(args.out)
    outroot.mkdir(parents=True, exist_ok=True)

    prior = estimate_prior(args.committee)
    print(f"estimated test prior from committee: "
          f"{ {k: round(v, 3) for k, v in prior.items()} }")

    print("\n--- committee pseudo-labels (round 0 seed) ---")
    pseudo = committee_pseudo_labels(args.committee, args.min_agreement,
                                     text_col=args.text_col)
    history = []
    absorbed = pd.DataFrame()

    for rnd in range(args.rounds + 1):
        rdir = outroot / f"round{rnd}"
        pool = pd.concat([pseudo, absorbed], ignore_index=True) \
            if len(absorbed) else pseudo
        pool = pool.drop_duplicates(subset=["text"], keep="first")
        pseudo_path = outroot / f"pseudo_round{rnd}.csv"
        pool.to_csv(pseudo_path, index=False)
        print(f"\n{'='*70}\nROUND {rnd}: {len(pool)} pseudo-labelled rows "
              f"{pool['label'].value_counts().to_dict() if len(pool) else {}}")

        cmd = [args.python, str(src / "train_transformer.py"),
               "--train", *args.base_train, str(pseudo_path),
               "--model", args.model, "--out", str(rdir),
               "--epochs", str(args.epochs), "--batch", str(args.batch),
               "--aux-weight", str(args.aux_weight),
               "--soft-alpha", str(args.soft_alpha),
               "--max-len", str(args.max_len), "--seed", str(args.seed)]
        if Path(args.dev).exists():
            cmd += ["--dev", args.dev]
        print("  $ " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(src))

        # predict the test sets with this round's student
        prob_paths = []
        for tpath in args.test:
            lang = "bn" if "bn" in Path(tpath).stem.lower() else "hi"
            dest = outroot / f"probs_round{rnd}_{lang}.csv"
            subprocess.run([args.python, str(src / "predict.py"),
                            "--model", str(rdir), "--test", tpath,
                            "--out", str(dest), "--text-col", args.text_col,
                            "--max-len", str(args.max_len)],
                           check=True, cwd=str(src))
            prob_paths.append(str(dest))

        rec = {"round": rnd, "pool_rows": len(pool),
               "pool_labels": pool["label"].value_counts().to_dict() if len(pool) else {}}
        mp = rdir / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            rec["internal_dev_macro_f1"] = m.get("eval_macro_f1")
            if "gold_dev" in m:
                rec["gold_dev_macro_f1"] = m["gold_dev"]["macro_f1"]
                rec["gold_dev_per_class_f1"] = m["gold_dev"]["per_class_f1"]
        history.append(rec)
        print(f"  round {rnd}: {json.dumps(rec)}")

        if rnd == args.rounds:
            for pth in prob_paths:
                shutil.copy(pth, outroot / Path(pth).name.replace(
                    f"round{rnd}_", "final_"))
            break

        print(f"\n--- absorbing new student pseudo-labels (round {rnd} -> {rnd+1}) ---")
        new = student_pseudo_labels(prob_paths, args.test, args.committee,
                                     args.student_threshold, prior, args.budget,
                                     text_col=args.text_col, seed=args.seed)
        seen = set(pool["src_id"].astype(str)) if "src_id" in pool.columns else set()
        new = new[~new["src_id"].astype(str).isin(seen)]
        print(f"  absorbed {len(new)} new rows "
              f"{new['label'].value_counts().to_dict() if len(new) else {}}")
        if new.empty:
            print("  nothing new to absorb — stopping early")
            for pth in prob_paths:
                shutil.copy(pth, outroot / Path(pth).name.replace(
                    f"round{rnd}_", "final_"))
            break
        absorbed = pd.concat([absorbed, new], ignore_index=True) \
            if len(absorbed) else new

    write_json(outroot / "selftrain_history.json",
               {"prior": prior, "history": history, "args": vars(args)})
    print(f"\n{'='*70}\nself-training history:")
    for rec in history:
        print(f"  round {rec['round']}: "
              f"internal={rec.get('internal_dev_macro_f1')} "
              f"gold_dev={rec.get('gold_dev_macro_f1')}")
    print(f"\nfinal probability frames: {list(outroot.glob('probs_final_*.csv'))}")
    print(f"pick the best round's checkpoint by GOLD dev macro-F1, not internal dev — "
          f"internal dev is synthetic + pseudo-labelled and will be optimistic.")


if __name__ == "__main__":
    main()
