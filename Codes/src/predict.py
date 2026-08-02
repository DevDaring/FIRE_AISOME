"""Inference for every SETU model kind -> a probability frame.

Everything downstream (calibrate.py, fuse.py, make_submission.py) consumes the same
schema, so channels are interchangeable:

    id, p_favour, p_against, p_none, pred[, node]

Probabilities, not hard labels, on purpose: under macro-F1 with a ~10 % minority class,
`argmax` is *not* the decision rule that maximises the metric. Throwing the
distribution away at this stage forfeits the gains calibrate.py exists to collect.

Supported models:
  * a SETU multi-task checkpoint directory (train_transformer.py) — has setu_model.pt
  * a plain HuggingFace sequence-classification directory
  * a LaBSE + LogisticRegression bundle (train_baseline.py) — *.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import LABELS, device_str, read_table, set_seed
from taxonomy import NODE_IDS, UNKNOWN_NODE


# ---------------------------------------------------------------------------
def _load_setu(model_dir: str):
    import torch
    from train_transformer import build_model
    ckpt = torch.load(Path(model_dir) / "setu_model.pt", map_location="cpu",
                      weights_only=False)
    model = build_model(ckpt["model_name"], ckpt.get("aux_weight", 0.0), None,
                        ckpt.get("soft_alpha", 0.0), 0.0)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device_str())
    return model, ckpt


def predict_frame(model_dir: str, texts: list[str], batch: int = 64,
                  max_len: int = 128) -> pd.DataFrame:
    """Probabilities from a SETU checkpoint or a plain HF classifier directory."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    d = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    dev = device_str()
    is_setu = (d / "setu_model.pt").exists()
    node_ids = NODE_IDS + [UNKNOWN_NODE]

    if is_setu:
        model, ckpt = _load_setu(model_dir)
        max_len = ckpt.get("max_len", max_len)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval().to(dev)

    probs = np.zeros((len(texts), len(LABELS)), dtype=np.float64)
    nodes: list[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(dev)
            out = model(**enc)
            logits = out["logits"] if isinstance(out, dict) else out.logits
            probs[i:i + len(chunk)] = torch.softmax(
                logits.float(), dim=-1).cpu().numpy()
            if is_setu and isinstance(out, dict) and out.get("node_logits") is not None:
                nl = out["node_logits"].float().argmax(-1).cpu().numpy()
                nodes.extend(node_ids[j] if j < len(node_ids) else UNKNOWN_NODE
                             for j in nl)
            if (i // max(batch, 1)) % 10 == 0:
                print(f"  {min(i + batch, len(texts))}/{len(texts)}", end="\r")

    # a plain HF checkpoint may not order its labels as LABELS does
    if not is_setu:
        id2label = getattr(model.config, "id2label", None) or {}
        order = [str(id2label.get(i, LABELS[i] if i < len(LABELS) else ""))
                 for i in range(probs.shape[1])]
        if set(order) == set(LABELS) and order != LABELS:
            probs = probs[:, [order.index(l) for l in LABELS]]

    df = pd.DataFrame({f"p_{l.lower()}": probs[:, i] for i, l in enumerate(LABELS)})
    df["pred"] = [LABELS[i] for i in probs.argmax(1)]
    if nodes:
        df["node"] = nodes
    return df


def predict_baseline(joblib_path: str, texts: list[str],
                     batch: int = 64) -> pd.DataFrame:
    import joblib
    from sentence_transformers import SentenceTransformer
    bundle = joblib.load(joblib_path)
    enc = SentenceTransformer(bundle["encoder"], device=device_str())
    X = enc.encode(texts, batch_size=batch, show_progress_bar=True,
                   convert_to_numpy=True)
    clf = bundle["clf"]
    P = clf.predict_proba(X)
    order = list(clf.classes_)
    probs = np.zeros((len(texts), len(LABELS)))
    for i, lab in enumerate(LABELS):
        if lab in order:
            probs[:, i] = P[:, order.index(lab)]
    df = pd.DataFrame({f"p_{l.lower()}": probs[:, i] for i, l in enumerate(LABELS)})
    df["pred"] = [LABELS[i] for i in probs.argmax(1)]
    return df


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SETU inference -> probability CSV")
    ap.add_argument("--model", required=True,
                    help="SETU/HF model directory OR path to baseline.joblib")
    ap.add_argument("--test", required=True, help="normalised test CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    set_seed(args.seed) if args.seed else set_seed()
    df = read_table(args.test)
    if args.text_col not in df.columns:
        raise SystemExit(f"{args.test}: no column {args.text_col!r} "
                         f"(has {list(df.columns)})")
    texts = df[args.text_col].astype(str).tolist()
    ids = (df[args.id_col].astype(str) if args.id_col in df.columns
           else pd.Series([str(i + 1) for i in range(len(df))]))
    print(f"{len(texts)} rows | model={args.model}")

    out = (predict_baseline(args.model, texts, args.batch)
           if args.model.endswith(".joblib")
           else predict_frame(args.model, texts, args.batch, args.max_len))
    out.insert(0, "id", ids.values)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\nwrote {len(out)} rows -> {dest}")
    print(f"  argmax distribution: {out['pred'].value_counts().to_dict()}")
    print(f"  mean top-1 probability: "
          f"{out[[f'p_{l.lower()}' for l in LABELS]].max(axis=1).mean():.3f}")
    if "node" in out.columns:
        print(f"  top argument nodes: {out['node'].value_counts().head(8).to_dict()}")


if __name__ == "__main__":
    main()
