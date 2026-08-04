"""Argument-KG projection channel — a hybrid of Deb et al. and the SETU taxonomy.

Provenance
----------
Adapts the hybrid projection framework of

    Deb, Mukherjee & Sanyal, "Hybrid Projection Methods for Multilingual
    Zero-shot Text Classification" (LREC, submitted)

which aligns multilingual sentence embeddings with *label* embeddings drawn from a
knowledge graph (ConceptNet / GloVe), learns the alignment with Ridge regression,
SVR or a Kolmogorov-Arnold Network, and classifies by cosine similarity in the
label space. Published zero-shot work on this family reports F1 gains up to ~21 %
over raw sentence-embedding baselines.

Two adaptations for this task
-----------------------------
1. **The taxonomy is the knowledge graph.** The paper needs ConceptNet because its
   labels are bare category names ("sports", "politics") with no internal
   structure. We already have something much richer and far more task-specific:
   27 argument nodes, each with an English gloss and cue phrases in English,
   Hindi and Bengali (228 in total). Embedding those gives a semantic space of
   *climate arguments* rather than of three label words — so the projection target
   carries roughly nine times more supervision signal than a 3-way label space
   would, and it is domain-matched instead of generic commonsense.

2. **Few-shot, not zero-shot.** The paper assumes no target-language labels. We
   have 727 judge-labelled test comments, each carrying an argument node id as
   well as a stance. So the projection is *supervised*: learn text-embedding ->
   node-embedding, then read the stance off the nearest nodes.

Why bother when a fine-tuned encoder already exists
---------------------------------------------------
Error decorrelation, which is the only thing that makes a fusion member worth its
weight. This channel differs from the encoders on every axis that matters:

  * the encoder is **frozen** — no gradient ever touches LaBSE;
  * classification is by **similarity in an argument space**, not a softmax head;
  * decisions route through **27 arguments**, not 3 labels;
  * the learner is **Ridge / SVR / KAN** on 768-d vectors, not a transformer.

The claim-conditioned NLI channel it is meant to replace was assigned weight
exactly 0.0 by the fusion search in both languages, so there is a free slot.

Usage
-----
    python3.12 src/projection_channel.py \\
        --train artifacts/distil_hi.csv artifacts/distil_bn.csv \\
        --test artifacts/test_hi.csv artifacts/test_bn.csv \\
        --dev artifacts/dev_holdout.csv --method hybrid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ARTIFACTS_DIR, LABELS, device_str, lang_from_path, load_dev,
                    macro_f1, read_csv, score_report, set_seed, SEED, write_json)
from taxonomy import NODES, NODE2STANCE

EMBED_MODEL = "sentence-transformers/LaBSE"   # 109 languages incl. hi + bn


# ---------------------------------------------------------------------------
# Embedding via plain transformers (mean pooling).
# sentence-transformers pulls in a torchvision build that is incompatible with
# this torch, and it is not needed: LaBSE is a BERT encoder and mean pooling over
# the attention mask reproduces its sentence embedding.
# ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, batch: int = 32,
                 max_len: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = device_str()
        print(f"loading {model_name} on {self.dev} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.dev)
        self.batch, self.max_len = batch, max_len

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch):
                chunk = [t if isinstance(t, str) and t.strip() else "empty"
                         for t in texts[i:i + self.batch]]
                enc = self.tok(chunk, return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_len).to(self.dev)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                out.append(v.cpu().numpy())
                if (i // self.batch) % 10 == 0:
                    print(f"  embedded {min(i + self.batch, len(texts))}/{len(texts)}",
                          end="\r")
        return np.vstack(out)


def node_embeddings(emb: Embedder) -> tuple[np.ndarray, list[str]]:
    """One vector per taxonomy node: mean of its gloss and all its cue phrases.

    This is the knowledge-graph side of the projection. Averaging the multilingual
    cue phrases in with the English gloss puts the node vector between the
    languages, which is what lets a Hindi or Bengali comment land near it.
    """
    texts, owner = [], []
    for n in NODES:
        variants = [n.gloss] + [c for lang_cues in n.cues.values() for c in lang_cues]
        texts.extend(variants)
        owner.extend([n.id] * len(variants))
    V = emb(texts)
    ids = [n.id for n in NODES]
    M = np.zeros((len(ids), V.shape[1]))
    for i, nid in enumerate(ids):
        rows = [j for j, o in enumerate(owner) if o == nid]
        v = V[rows].mean(axis=0)
        M[i] = v / max(np.linalg.norm(v), 1e-9)
    print(f"\n  {len(ids)} node embeddings from {len(texts)} gloss+cue strings")
    return M, ids


# ---------------------------------------------------------------------------
# Projection learners
# ---------------------------------------------------------------------------
def fit_ridge(X, Y, alpha: float = 1.0):
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha)
    m.fit(X, Y)
    return lambda Z: m.predict(Z)


def fit_svr(X, Y, C: float = 2.0, eps: float = 0.05):
    """Per-dimension SVR. Paper's stable, well-regularised component."""
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import LinearSVR
    m = MultiOutputRegressor(
        LinearSVR(C=C, epsilon=eps, max_iter=4000, random_state=SEED), n_jobs=-1)
    m.fit(X, Y)
    return lambda Z: m.predict(Z)


def fit_kan(X, Y, hidden: int = 96, epochs: int = 260, lr: float = 3e-3):
    """Kolmogorov-Arnold Network projection.

    The paper found KAN best for morphologically rich Hindi and Bengali, where the
    text->label mapping is less linear than in English. Learnable univariate
    spline activations give a non-linear map with few parameters, which matters at
    727 training rows.
    """
    import torch
    from efficient_kan import KAN
    torch.manual_seed(SEED)
    dev = device_str()
    net = KAN([X.shape[1], hidden, Y.shape[1]]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    net.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 60 == 0:
            print(f"    KAN epoch {ep:3d}  mse {loss.item():.5f}")
    net.eval()

    def predict(Z):
        with torch.no_grad():
            return net(torch.tensor(Z, dtype=torch.float32, device=dev)).cpu().numpy()
    return predict


# ---------------------------------------------------------------------------
def stance_probs(proj: np.ndarray, node_M: np.ndarray, node_ids: list[str],
                 temp: float = 0.05) -> np.ndarray:
    """Project -> cosine to every node -> softmax over nodes -> sum per stance.

    Aggregating over nodes rather than taking the single nearest one means several
    weakly-matching arguments of the same stance can outvote one strong match of
    another — which is the behaviour we want when a comment makes a compound point.
    """
    P = proj / np.linalg.norm(proj, axis=1, keepdims=True).clip(1e-9)
    sim = P @ node_M.T
    sim = sim / max(temp, 1e-6)
    sim -= sim.max(axis=1, keepdims=True)
    W = np.exp(sim)
    W /= W.sum(axis=1, keepdims=True)
    out = np.zeros((len(P), len(LABELS)))
    for j, nid in enumerate(node_ids):
        out[:, LABELS.index(NODE2STANCE[nid])] += W[:, j]
    return out / out.sum(axis=1, keepdims=True).clip(1e-9)


def main():
    ap = argparse.ArgumentParser(description="Argument-KG projection channel")
    ap.add_argument("--train", nargs="+", required=True,
                    help="labelled rows with a node_id column (distil_*.csv)")
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--dev", default=str(ARTIFACTS_DIR / "dev_holdout.csv"))
    ap.add_argument("--method", default="hybrid",
                    choices=["ridge", "svr", "kan", "hybrid"],
                    help="'hybrid' = KAN + SVR weighted blend, as in the paper")
    ap.add_argument("--kan-weight", type=float, default=0.9,
                    help="paper's best Hindi setting was 90% KAN + 10% SVR")
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--outdir", default=str(ARTIFACTS_DIR))
    args = ap.parse_args()

    set_seed(SEED)
    emb = Embedder(args.embed_model)
    node_M, node_ids = node_embeddings(emb)

    # ---- training rows: text -> its argument node's embedding ---------------
    frames = [read_csv(p) for p in args.train if Path(p).exists()]
    tr = pd.concat(frames, ignore_index=True)
    tr = tr[tr["label"].isin(LABELS)].reset_index(drop=True)
    nid_index = {n: i for i, n in enumerate(node_ids)}
    # rows whose judge-assigned node is UNK cannot supervise the projection, so
    # fall back to the centroid of that stance's nodes
    stance_centroid = {}
    for lab in LABELS:
        rows = [i for i, n in enumerate(node_ids) if NODE2STANCE[n] == lab]
        c = node_M[rows].mean(axis=0)
        stance_centroid[lab] = c / max(np.linalg.norm(c), 1e-9)

    Y = np.vstack([
        node_M[nid_index[n]] if n in nid_index else stance_centroid[lab]
        for n, lab in zip(tr.get("node_id", pd.Series(["UNK"] * len(tr))).astype(str),
                          tr["label"])])
    n_known = sum(1 for n in tr.get("node_id", pd.Series(["UNK"] * len(tr))).astype(str)
                  if n in nid_index)
    print(f"train rows {len(tr)} ({n_known} with a known argument node, "
          f"{len(tr) - n_known} via stance centroid)")
    X = emb(tr["text"].astype(str).tolist())

    # ---- fit ---------------------------------------------------------------
    print(f"\nfitting projection ({args.method}) {X.shape} -> {Y.shape}")
    preds = {}
    if args.method in ("ridge",):
        preds["ridge"] = fit_ridge(X, Y)
    if args.method in ("svr", "hybrid"):
        preds["svr"] = fit_svr(X, Y)
    if args.method in ("kan", "hybrid"):
        preds["kan"] = fit_kan(X, Y)

    def project(Z):
        if args.method == "hybrid":
            w = args.kan_weight
            return w * preds["kan"](Z) + (1 - w) * preds["svr"](Z)
        return preds[args.method](Z)

    # ---- predict each language --------------------------------------------
    report = {"method": args.method, "kan_weight": args.kan_weight,
              "embed_model": args.embed_model, "train_rows": len(tr),
              "nodes": len(node_ids), "languages": {}}
    for tpath in args.test:
        lang = lang_from_path(tpath)
        df = read_csv(tpath)
        Z = emb(df["text"].astype(str).tolist())
        P = stance_probs(project(Z), node_M, node_ids, temp=args.temp)
        out = pd.DataFrame({"id": df["id"].astype(str)})
        for i, lab in enumerate(LABELS):
            out[f"p_{lab.lower()}"] = P[:, i]
        out["pred"] = [LABELS[i] for i in P.argmax(1)]
        dest = Path(args.outdir) / f"probs_proj_{lang}.csv"
        out.to_csv(dest, index=False)
        print(f"\nwrote {dest}  {out['pred'].value_counts().to_dict()}")

        if Path(args.dev).exists():
            dev = load_dev(args.dev, lang=lang)
            m = out.merge(dev[["id", "gold"]], on="id", how="inner")
            if len(m) >= 10:
                rep = score_report(m["gold"].tolist(), m["pred"].tolist(),
                                   title=f"projection[{args.method}] {lang}")
                report["languages"][lang] = rep

    write_json(Path(args.outdir) / "projection_report.json", report)
    if report["languages"]:
        print("\nsummary: " + "  ".join(
            f"{l}={r['macro_f1']:.4f}" for l, r in report["languages"].items()))


if __name__ == "__main__":
    main()
