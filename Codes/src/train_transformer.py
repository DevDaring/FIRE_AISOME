"""Argument-aware multi-task stance encoder (SETU contribution C3).

Architecture
------------
A shared Indic encoder with two heads:

  1. **stance** (3-way, primary) — trained on a mixture of hard labels, GWSD crowd
     soft labels and committee soft labels, via
     ``L = (1-a)*CE(hard) + a*KL(soft || p)`` with per-example weights.
  2. **argument node** (|taxonomy|+1-way, auxiliary, weight λ) — predicts which SETU
     taxonomy node the comment instantiates.

Why the auxiliary head. The failure mode we are fighting is a model that latches onto
English lexical cues ("hoax", "Al Gore") which do not exist in the test set. Forcing the
shared representation to also predict *which argument is being made* pushes it toward
argument structure, which is the axis that actually transfers across languages. The
supervision is free: synth_generate.py labels every synthetic row by construction, and
llm_committee.py predicts a node for every real comment. Rows without a node get `UNK`
and are masked out of the auxiliary loss.

Recommended backbones (both worth submitting as separate runs):
  * ``google/muril-base-cased`` — pretrained on Indian-language corpora *including
    transliterated pairs*, which is why it beats XLM-R on Roman-script code-mix.
  * ``xlm-roberta-base`` — the standard multilingual baseline; different errors.

Class imbalance is handled by class-weighted loss (default) rather than oversampling —
with soft labels, duplicating rows distorts the target distribution, whereas reweighting
does not. `--balance oversample` reproduces the older behaviour for the ablation table.

Input files use a ``path[::text_column]`` spec so the same CSV can contribute its
English, Hindi and Bengali columns as separate views of the pool.

Usage
-----
    python3.12 src/train_transformer.py \
        --train artifacts/train_en.csv \
                artifacts/train_en.to-hi.csv::text_hi \
                artifacts/train_en.to-bn.csv::text_bn \
                artifacts/synth_train.csv \
        --model google/muril-base-cased --out artifacts/model_muril \
        --epochs 3 --batch 16 --aux-weight 0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ARTIFACTS_DIR, ID2LABEL, LABEL2ID, LABELS, SEED, device_str,
                    read_csv, score_report, set_seed, write_json)
from taxonomy import N_NODES, NODE2IDX, UNKNOWN_NODE

SOFT_COLS = ["p_favour", "p_against", "p_none"]


# ---------------------------------------------------------------------------
def load_train_files(specs: list[str], default_text_col: str = "text") -> pd.DataFrame:
    """Load 'path' or 'path::text_col' entries into one frame.

    Output columns: text, label, p_favour, p_against, p_none, node_idx, weight, origin
    Missing soft labels are synthesised as one-hot from `label`; missing nodes -> UNK.
    """
    frames = []
    for spec in specs:
        path, col = (spec.rsplit("::", 1) if "::" in spec
                     else (spec, default_text_col))
        p = Path(path)
        if not p.exists():
            print(f"  SKIP (missing): {p}")
            continue
        df = read_csv(p)
        if col not in df.columns:
            raise SystemExit(f"{p}: no column {col!r} (has {list(df.columns)})")
        if "label" not in df.columns:
            raise SystemExit(f"{p}: no 'label' column")

        sub = pd.DataFrame({
            "text": df[col].astype(str),
            "label": df["label"].astype(str),
        })
        if all(c in df.columns for c in SOFT_COLS):
            for c in SOFT_COLS:
                sub[c] = pd.to_numeric(df[c], errors="coerce")
            bad = sub[SOFT_COLS].isna().any(axis=1)
            for c, lab in zip(SOFT_COLS, LABELS):
                sub.loc[bad, c] = (sub.loc[bad, "label"] == lab).astype(float)
        else:
            for c, lab in zip(SOFT_COLS, LABELS):
                sub[c] = (sub["label"] == lab).astype(float)

        nodes = (df["node_id"].astype(str) if "node_id" in df.columns
                 else pd.Series([UNKNOWN_NODE] * len(df)))
        sub["node_idx"] = [NODE2IDX.get(n, NODE2IDX[UNKNOWN_NODE]) for n in nodes]
        sub["weight"] = (pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
                         if "weight" in df.columns else 1.0)
        sub["origin"] = f"{p.stem}[{col}]"
        sub = sub[sub["text"].str.strip().str.len() > 0]
        sub = sub[sub["label"].isin(LABELS)]
        frames.append(sub)
        print(f"  {p.name} [{col}]: {len(sub)} rows "
              f"{sub['label'].value_counts().to_dict()}")

    if not frames:
        raise SystemExit("no training data loaded")
    out = pd.concat(frames, ignore_index=True)
    # renormalise soft labels defensively (crowd fractions can be off by rounding)
    s = out[SOFT_COLS].sum(axis=1).replace(0, np.nan)
    out[SOFT_COLS] = out[SOFT_COLS].div(s, axis=0).fillna(1.0 / len(LABELS))
    return out.reset_index(drop=True)


def oversample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    target = df["label"].value_counts().max()
    parts = []
    for _, sub in df.groupby("label"):
        if len(sub) < target:
            sub = pd.concat(
                [sub, sub.sample(target - len(sub), replace=True, random_state=seed)],
                ignore_index=True)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)


# ---------------------------------------------------------------------------
def build_model(model_name: str, aux_weight: float, class_weights, soft_alpha: float,
                label_smoothing: float, attn: str = "auto"):
    """A HF sequence-classification model with an extra argument-node head."""
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    def _no_pool(name: str) -> bool:
        n = name.lower()
        return "roberta" in n and "muril" not in n

    class SetuStanceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = AutoConfig.from_pretrained(
                model_name, num_labels=len(LABELS),
                id2label={int(i): l for i, l in ID2LABEL.items()},
                label2id=dict(LABEL2ID))
            # Attention backend. flash_attention_2 is only implemented for some
            # architectures; sdpa is PyTorch's fused kernel (flash under the hood),
            # works everywhere, and at seq_len 128 the two are within noise of each
            # other anyway — attention is not the bottleneck at this length.
            kw = {}
            if attn != "eager":
                order = (["flash_attention_2", "sdpa"] if attn in ("auto", "flash")
                         else [attn])
                for impl in order:
                    try:
                        AutoModel.from_pretrained(model_name, config=self.config,
                                                  attn_implementation=impl,
                                                  **({"add_pooling_layer": False}
                                                     if _no_pool(model_name) else {}))
                        kw["attn_implementation"] = impl
                        print(f"  attention backend: {impl}")
                        break
                    except (ValueError, ImportError, RuntimeError, OSError) as e:
                        print(f"  {impl} unavailable ({type(e).__name__}), trying next")
            if _no_pool(model_name):
                kw["add_pooling_layer"] = False
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config,
                                                     **kw)
            h = self.config.hidden_size
            self.dropout = nn.Dropout(getattr(self.config, "hidden_dropout_prob", 0.1))
            self.stance_head = nn.Linear(h, len(LABELS))
            self.node_head = nn.Linear(h, N_NODES)
            self.aux_weight = aux_weight
            self.soft_alpha = soft_alpha
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float)
                if class_weights is not None else torch.ones(len(LABELS)))
            self.label_smoothing = label_smoothing

        def _pool(self, out, attention_mask):
            # mean pooling: more robust than [CLS] for very short comments,
            # which is most of this test set
            hs = out.last_hidden_state
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            return (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    labels=None, soft_labels=None, node_labels=None,
                    example_weight=None, **kw):
            enc_kw = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None and \
                    "token_type_ids" in self.encoder.forward.__code__.co_varnames:
                enc_kw["token_type_ids"] = token_type_ids
            out = self.encoder(**enc_kw)
            pooled = self.dropout(self._pool(out, attention_mask))
            stance_logits = self.stance_head(pooled)
            node_logits = self.node_head(pooled)

            loss = None
            if labels is not None:
                w = (example_weight if example_weight is not None
                     else torch.ones_like(labels, dtype=stance_logits.dtype))
                ce = nn.functional.cross_entropy(
                    stance_logits, labels, weight=self.class_weights,
                    label_smoothing=self.label_smoothing, reduction="none")
                loss = (ce * w).sum() / w.sum().clamp(min=1e-9)

                if soft_labels is not None and self.soft_alpha > 0:
                    logp = nn.functional.log_softmax(stance_logits, dim=-1)
                    kl = nn.functional.kl_div(logp, soft_labels, reduction="none").sum(-1)
                    kl = (kl * w).sum() / w.sum().clamp(min=1e-9)
                    loss = (1 - self.soft_alpha) * loss + self.soft_alpha * kl

                if node_labels is not None and self.aux_weight > 0:
                    mask = node_labels != NODE2IDX[UNKNOWN_NODE]
                    if mask.any():
                        aux = nn.functional.cross_entropy(
                            node_logits[mask], node_labels[mask])
                        loss = loss + self.aux_weight * aux

            return {"loss": loss, "logits": stance_logits, "node_logits": node_logits}

    return SetuStanceModel()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train the SETU multi-task stance encoder")
    ap.add_argument("--train", nargs="+", required=True,
                    help="CSV paths, optionally 'path::text_col'")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default=None,
                    help="gold dev CSV from annotate_dev.py (id,text,gold) — the only "
                         "honest macro-F1 signal we have")
    ap.add_argument("--dev-text-col", default="text")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--dev-frac", type=float, default=0.1,
                    help="internal stratified split for early stopping; 0 = use all data")
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="weight of the argument-node auxiliary loss (0 = ablation)")
    ap.add_argument("--soft-alpha", type=float, default=0.4,
                    help="weight of the soft-label KL term (0 = hard labels only)")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--balance", default="class_weight",
                    choices=["class_weight", "oversample", "none"])
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-train", type=int, default=0, help="smoke test with N rows")
    ap.add_argument("--init-from", default=None,
                    help="resume from an existing SETU checkpoint directory instead "
                         "of the raw pretrained backbone. Enables two-stage "
                         "domain-adaptive fine-tuning: stage 1 on the large "
                         "off-distribution pool, stage 2 on in-distribution labels "
                         "only at a low LR. Without it the 727 judge-labelled test "
                         "rows are 8% of a 9k pool and get drowned out by translated "
                         "and synthetic text.")
    ap.add_argument("--attn", default="auto",
                    choices=["auto", "flash", "flash_attention_2", "sdpa", "eager"],
                    help="attention kernel; 'auto' tries flash_attention_2 then sdpa")
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 instead of fp16 (safer on Ampere+ with flash attn)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from transformers import (AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)

    set_seed(args.seed)
    dev_name = device_str()
    print(f"device: {dev_name} | backbone: {args.model}")

    df = load_train_files(args.train, args.text_col)
    if args.max_train:
        df = df.sample(min(args.max_train, len(df)), random_state=args.seed)
    print(f"\npool: {len(df)} rows | labels {df['label'].value_counts().to_dict()}")
    print(f"      node-labelled rows: "
          f"{int((df['node_idx'] != NODE2IDX[UNKNOWN_NODE]).sum())}")

    if args.dev_frac > 0 and len(df) > 50:
        tr, va = train_test_split(df, test_size=args.dev_frac,
                                  stratify=df["label"], random_state=args.seed)
    else:
        tr, va = df, None
    if args.balance == "oversample":
        tr = oversample(tr, args.seed)

    class_weights = None
    if args.balance == "class_weight":
        counts = np.array([max((tr["label"] == l).sum(), 1) for l in LABELS],
                          dtype=np.float64)
        class_weights = (counts.sum() / (len(LABELS) * counts)).tolist()
        print(f"class weights: {dict(zip(LABELS, [round(w, 3) for w in class_weights]))}")

    tok = AutoTokenizer.from_pretrained(args.model)

    class StanceDataset(torch.utils.data.Dataset):
        def __init__(self, frame):
            self.enc = tok(frame["text"].astype(str).tolist(), truncation=True,
                           max_length=args.max_len)
            self.labels = frame["label"].map(LABEL2ID).tolist()
            self.soft = frame[SOFT_COLS].to_numpy(dtype="float32")
            self.nodes = frame["node_idx"].astype(int).tolist()
            self.w = np.asarray(frame["weight"], dtype="float32")

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            item["soft_labels"] = self.soft[i].tolist()
            item["node_labels"] = self.nodes[i]
            item["example_weight"] = float(self.w[i])
            return item

    base_collator = DataCollatorWithPadding(tok)

    def collate(features):
        soft = torch.tensor([f.pop("soft_labels") for f in features],
                            dtype=torch.float)
        nodes = torch.tensor([f.pop("node_labels") for f in features],
                             dtype=torch.long)
        w = torch.tensor([f.pop("example_weight") for f in features],
                         dtype=torch.float)
        batch = base_collator(features)
        batch["soft_labels"] = soft
        batch["node_labels"] = nodes
        batch["example_weight"] = w
        return batch

    model = build_model(args.model, args.aux_weight, class_weights, args.soft_alpha,
                        args.label_smoothing, attn=args.attn)
    if args.init_from:
        ckpt_path = Path(args.init_from) / "setu_model.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"--init-from {args.init_from}: no setu_model.pt")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck.get("model_name") != args.model:
            raise SystemExit(
                f"--init-from backbone mismatch: checkpoint is "
                f"{ck.get('model_name')!r} but --model is {args.model!r}")
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        print(f"resumed from {args.init_from}"
              + (f" (missing {len(missing)}, unexpected {len(unexpected)})"
                 if missing or unexpected else ""))

    def metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.asarray(logits).argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, preds, average="macro",
                                     zero_division=0)}

    out_dir = Path(args.out)
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch" if va is not None else "no",
        save_strategy="epoch" if va is not None else "no",
        save_total_limit=1,
        load_best_model_at_end=va is not None,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=(dev_name == "cuda" and not args.bf16),
        bf16=(dev_name == "cuda" and args.bf16),
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
        logging_steps=50,
        label_names=["labels"],
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=StanceDataset(tr),
                      eval_dataset=StanceDataset(va) if va is not None else None,
                      data_collator=collate,
                      compute_metrics=metrics if va is not None else None)
    print(f"\ntraining on {len(tr)} rows"
          + (f", internal dev {len(va)}" if va is not None else " (no internal dev)"))
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "model_name": args.model,
                "aux_weight": args.aux_weight,
                "soft_alpha": args.soft_alpha,
                "max_len": args.max_len,
                "labels": LABELS}, out_dir / "setu_model.pt")
    tok.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)

    result = {"backbone": args.model, "train_rows": len(tr), "epochs": args.epochs,
              "aux_weight": args.aux_weight, "soft_alpha": args.soft_alpha,
              "balance": args.balance, "lr": args.lr,
              "train_files": args.train}
    if va is not None:
        result.update({k: round(float(v), 4)
                       for k, v in trainer.evaluate().items() if k.startswith("eval_")})

    # The honest number: macro-F1 on the hand-annotated Hindi/Bengali dev set.
    if args.dev and Path(args.dev).exists():
        gold = read_csv(args.dev)
        gold = gold[gold["gold"].isin(LABELS)]
        if len(gold):
            from predict import predict_frame
            preds = predict_frame(str(out_dir), gold[args.dev_text_col].astype(str).tolist(),
                                  batch=args.eval_batch, max_len=args.max_len)
            rep = score_report(gold["gold"].tolist(), preds["pred"],
                               title=f"GOLD DEV — {args.model}")
            result["gold_dev"] = rep

    write_json(out_dir / "metrics.json", result)
    print("\n" + json.dumps(result, indent=2))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
