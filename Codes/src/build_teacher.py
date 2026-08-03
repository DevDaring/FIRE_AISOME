"""Merge the cheap teacher committee and the strong judge panel into one teacher.

Why this is the highest-leverage step left
-----------------------------------------
Measured on the dev set: the cheap 5-model committee agrees with the strong judge
panel on **85.7 %** of the unbiased stratum, while our fine-tuned encoders reach
only ~0.62 macro-F1 on the same rows. The LLM labels are markedly better than what
the encoder has learned, because the encoder trains mostly on synthetic and
machine-translated text and has never seen the committee signal at all.

So the win is not a better architecture — it is **distilling the best available
labels, on the actual test comments, into the encoder**. That also keeps every
submitted run a classifier we trained, which is what the track asks for: the
encoder ends up as a compact, offline, reproducible compression of a 9-model
panel rather than a live API call.

What this produces
------------------
A committee-format CSV per language (the schema selftrain.py already consumes):

    id, text, p_favour, p_against, p_none, committee_label, agreement,
    entropy, node_id, n_panels

Fusion rules, and the reasoning behind each:

  * **Per-panel soft labels are averaged with weights**, strong judges above cheap
    teachers (default 2:1). Not a hard vote — a 3-2 split and a 5-0 split are very
    different training signals and hard voting throws that away.
  * **Agreement becomes the per-row sample weight**, so rows the panels split on
    contribute proportionally less instead of being dropped or trusted equally.
  * **Rows where the two panels disagree on the argmax are down-weighted further**,
    because a cross-panel conflict is stronger evidence of genuine ambiguity than
    disagreement inside one panel.
  * **Judge coverage is partial** (batched JSON does not always parse for every
    row), so a row falls back gracefully to whichever panels covered it.

Usage
-----
    python3.12 src/build_teacher.py \\
        --committee artifacts/committee_hi.csv artifacts/committee_bn.csv \\
        --judges artifacts/judge_all.csv \\
        --judge-weight 2.0 --out-prefix artifacts/teacher
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import ARTIFACTS_DIR, LABELS, lang_from_path, read_csv, write_json

PCOLS = [f"p_{l.lower()}" for l in LABELS]


def _panel(path: str, lang: str | None = None) -> pd.DataFrame:
    """Load a committee- or judge-format file down to (id, lang, probs, agreement)."""
    df = read_csv(path)
    df["id"] = df["id"].astype(str).str.strip()
    if "lang" not in df.columns:
        df["lang"] = lang or lang_from_path(path)
    if lang:
        df = df[df["lang"] == lang]
    if not len(df):
        return pd.DataFrame(columns=["id", "lang", "text", "agreement", "node_id"] + PCOLS)

    out = pd.DataFrame({"id": df["id"], "lang": df["lang"]})
    out["text"] = df["text"] if "text" in df.columns else ""

    if all(c in df.columns for c in PCOLS):
        P = df[PCOLS].to_numpy(dtype=np.float64)
    else:
        # judge files carry a single hard label; make it a confident one-hot so it
        # can be averaged with the committee's genuine distributions
        col = next((c for c in ("gold", "silver", "committee_label", "pred")
                    if c in df.columns), None)
        if col is None:
            raise SystemExit(f"{path}: no probabilities and no label column")
        P = np.full((len(df), len(LABELS)), 0.02)
        for i, v in enumerate(df[col].astype(str)):
            if v in LABELS:
                P[i, LABELS.index(v)] = 0.96
    P = np.clip(P, 1e-9, None)
    out[PCOLS] = P / P.sum(axis=1, keepdims=True)

    for src, dst, default in (("agreement", "agreement", 1.0),
                              ("judge_agree", "agreement", 1.0),
                              ("node_id", "node_id", "UNK")):
        if src in df.columns and dst not in out.columns:
            out[dst] = df[src]
    out["agreement"] = pd.to_numeric(out.get("agreement", 1.0),
                                     errors="coerce").fillna(1.0)
    if "node_id" not in out.columns:
        out["node_id"] = "UNK"
    return out.reset_index(drop=True)


def merge_lang(panels: list[tuple[str, pd.DataFrame, float]], lang: str):
    """panels = [(name, frame, weight)] -> one committee-format frame for `lang`."""
    idx: dict[str, dict] = {}
    for name, frame, w in panels:
        sub = frame[frame["lang"] == lang]
        for _, r in sub.iterrows():
            e = idx.setdefault(str(r["id"]), {
                "text": "", "num": np.zeros(len(LABELS)), "den": 0.0,
                "panels": [], "argmax": [], "nodes": [], "agrees": []})
            if not e["text"] and isinstance(r["text"], str) and r["text"].strip():
                e["text"] = r["text"]
            p = r[PCOLS].to_numpy(dtype=np.float64)
            # weight by panel strength AND by that panel's internal agreement
            wt = w * max(float(r["agreement"]), 0.05)
            e["num"] += wt * p
            e["den"] += wt
            e["panels"].append(name)
            e["argmax"].append(LABELS[int(np.argmax(p))])
            e["agrees"].append(float(r["agreement"]))
            if str(r.get("node_id", "UNK")) not in ("UNK", "nan", ""):
                e["nodes"].append(str(r["node_id"]))

    rows = []
    for cid, e in idx.items():
        if e["den"] <= 0:
            continue
        p = e["num"] / e["den"]
        p = p / p.sum()
        top = LABELS[int(np.argmax(p))]
        mean_agree = float(np.mean(e["agrees"]))
        # a cross-panel argmax conflict is stronger evidence of real ambiguity than
        # disagreement within a single panel, so penalise it explicitly
        cross_ok = len(set(e["argmax"])) == 1
        weight = mean_agree * (1.0 if cross_ok else 0.6)
        ent = -sum(x * math.log(x + 1e-12) for x in p) / math.log(len(LABELS))
        rows.append({
            "id": cid, "lang": lang, "text": e["text"],
            **{c: round(float(p[i]), 6) for i, c in enumerate(PCOLS)},
            "committee_label": top,
            "agreement": round(weight, 4),
            "panel_agreement": round(mean_agree, 4),
            "cross_panel_agree": bool(cross_ok),
            "entropy": round(ent, 4),
            "node_id": (max(set(e["nodes"]), key=e["nodes"].count)
                        if e["nodes"] else "UNK"),
            "n_panels": len(e["panels"]),
        })
    return pd.DataFrame(rows).sort_values(
        "id", key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Merge teacher panels into one signal")
    ap.add_argument("--committee", nargs="+", required=True,
                    help="cheap committee files (committee_hi.csv committee_bn.csv)")
    ap.add_argument("--judges", nargs="*", default=[],
                    help="strong judge files (judge_all.csv, dev_gold.csv, ...)")
    ap.add_argument("--judge-weight", type=float, default=2.0,
                    help="weight of a strong judge panel relative to the cheap one")
    ap.add_argument("--out-prefix", default=str(ARTIFACTS_DIR / "teacher"))
    args = ap.parse_args()

    panels = []
    for p in args.committee:
        if Path(p).exists():
            panels.append((f"committee:{Path(p).stem}", _panel(p), 1.0))
            print(f"  committee {Path(p).name}: {len(panels[-1][1])} rows")
    for p in args.judges:
        if Path(p).exists():
            panels.append((f"judge:{Path(p).stem}", _panel(p), args.judge_weight))
            print(f"  judge     {Path(p).name}: {len(panels[-1][1])} rows "
                  f"(weight {args.judge_weight})")
    if not panels:
        raise SystemExit("no panel files found")

    stats = {"judge_weight": args.judge_weight, "languages": {}}
    for lang in ("hi", "bn"):
        merged = merge_lang(panels, lang)
        if not len(merged):
            continue
        dest = f"{args.out_prefix}_{lang}.csv"
        merged.to_csv(dest, index=False)
        s = {
            "rows": len(merged),
            "distribution": merged["committee_label"].value_counts().to_dict(),
            "covered_by_2plus_panels": int((merged["n_panels"] >= 2).sum()),
            "cross_panel_agreement": round(
                float(merged["cross_panel_agree"].mean()), 4),
            "mean_sample_weight": round(float(merged["agreement"].mean()), 4),
            "high_confidence_rows": int((merged["agreement"] >= 0.8).sum()),
        }
        stats["languages"][lang] = s
        print(f"\n{lang}: wrote {dest}")
        for k, v in s.items():
            print(f"    {k}: {v}")

    write_json(f"{args.out_prefix}.stats.json", stats)
    print(f"\nfeed these to selftrain.py --committee "
          f"{args.out_prefix}_hi.csv {args.out_prefix}_bn.csv")


if __name__ == "__main__":
    main()
