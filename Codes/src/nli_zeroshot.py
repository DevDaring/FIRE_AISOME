"""Claim-conditioned NLI: a training-free third channel (SETU contribution C4, part 3).

Stance is a *relation between a text and a claim*, which makes it natively an
entailment problem. So we score each comment against three hypotheses with an
off-the-shelf multilingual NLI model and read the stance off the entailment scores:

    premise    = the comment (in its own language)
    hypothesis = "This comment agrees that climate change ... is a serious concern."
                 "This comment denies that climate change ... is a serious concern."
                 "This comment does not express any opinion about climate change."

Why bother when we already have a fine-tuned encoder and an LLM committee?
**Error decorrelation.** The encoder's errors come from its training pool; the
committee's come from shared LLM priors (all of them over-predict `Favour` on polite
comments). This channel has seen neither our training data nor our prompts — its
errors are independent, which is exactly what makes an ensemble member worth adding.
It costs one CPU pass over 1000 comments and needs zero labelled data.

Default model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
(100 languages, strong zero-shot; a well-established zero-shot stance baseline).
Alternative: `joeddav/xlm-roberta-large-xnli` (better, ~3x slower on CPU).

Output: artifacts/probs_nli_<lang>.csv  ->  id, p_favour, p_against, p_none, pred
which is the same probability-frame schema every other channel writes, so fuse.py
can consume it directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ARTIFACTS_DIR, LABELS, device_str, read_table, set_seed

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Hypotheses per label, per language. Written so that the *same* premise can be
# tested against all three and the entailment scores compared.
HYPOTHESES = {
    "en": {
        "Favour": "This comment agrees that climate change and global warming are a "
                  "serious concern.",
        "Against": "This comment denies or dismisses that climate change and global "
                   "warming are a serious concern.",
        "None": "This comment expresses no opinion about whether climate change is a "
                "serious concern.",
    },
    "hi": {
        "Favour": "यह टिप्पणी इस बात से सहमत है कि जलवायु परिवर्तन और ग्लोबल वार्मिंग "
                  "एक गंभीर चिंता का विषय है।",
        "Against": "यह टिप्पणी इस बात को नकारती या खारिज करती है कि जलवायु परिवर्तन और "
                   "ग्लोबल वार्मिंग एक गंभीर चिंता का विषय है।",
        "None": "यह टिप्पणी जलवायु परिवर्तन की गंभीरता पर कोई राय नहीं देती।",
    },
    "bn": {
        "Favour": "এই মন্তব্যটি একমত যে জলবায়ু পরিবর্তন এবং বিশ্ব উষ্ণায়ন একটি গুরুতর "
                  "উদ্বেগের বিষয়।",
        "Against": "এই মন্তব্যটি অস্বীকার বা উড়িয়ে দেয় যে জলবায়ু পরিবর্তন এবং বিশ্ব "
                   "উষ্ণায়ন একটি গুরুতর উদ্বেগের বিষয়।",
        "None": "এই মন্তব্যটি জলবায়ু পরিবর্তনের গুরুত্ব নিয়ে কোনো মত প্রকাশ করে না।",
    },
}


def entail_index(config) -> int:
    """Locate the 'entailment' logit — label order differs between NLI checkpoints."""
    id2label = getattr(config, "id2label", None) or {}
    for i, name in id2label.items():
        if str(name).lower().startswith("entail"):
            return int(i)
    return 2 if len(id2label) == 3 else 0


def score(texts: list[str], lang: str, model_name: str = DEFAULT_MODEL,
          batch: int = 16, max_len: int = 256,
          hypothesis_lang: str | None = None) -> np.ndarray:
    """-> (n, 3) probabilities over LABELS from softmaxed entailment scores."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hyp_lang = hypothesis_lang or lang
    hyps = HYPOTHESES.get(hyp_lang, HYPOTHESES["en"])
    dev = device_str()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval().to(dev)
    e_idx = entail_index(model.config)
    print(f"NLI: {model_name} | entailment logit index {e_idx} | device {dev} | "
          f"hypotheses in {hyp_lang!r}")

    raw = np.zeros((len(texts), len(LABELS)), dtype=np.float64)
    with torch.no_grad():
        for li, lab in enumerate(LABELS):
            hyp = hyps[lab]
            for i in range(0, len(texts), batch):
                chunk = texts[i:i + batch]
                enc = tok(chunk, [hyp] * len(chunk), return_tensors="pt",
                          padding=True, truncation=True, max_length=max_len).to(dev)
                logits = model(**enc).logits.float().cpu().numpy()
                raw[i:i + len(chunk), li] = logits[:, e_idx]
            print(f"  scored hypothesis {lab!r}")

    # softmax across the three entailment scores -> a proper distribution
    raw -= raw.max(axis=1, keepdims=True)
    ex = np.exp(raw)
    return ex / ex.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description="Zero-shot claim-conditioned NLI channel")
    ap.add_argument("--test", nargs="+", required=True,
                    help="normalised test csv(s), e.g. artifacts/test_hi.csv")
    ap.add_argument("--lang", nargs="+", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--hypothesis-lang", default=None,
                    help="force hypothesis language ('en' is often more reliable than "
                         "hi/bn for XNLI checkpoints — worth an ablation)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--outdir", default=str(ARTIFACTS_DIR))
    args = ap.parse_args()

    set_seed()
    outdir = Path(args.outdir)
    for k, path in enumerate(args.test):
        p = Path(path)
        lang = (args.lang[k] if args.lang and k < len(args.lang)
                else ("bn" if "bn" in p.stem.lower() else "hi"))
        df = read_table(p)
        texts = df[args.text_col].astype(str).tolist()
        print(f"\n=== {p.name} lang={lang} rows={len(texts)} ===")
        probs = score(texts, lang, args.model, args.batch, args.max_len,
                      hypothesis_lang=args.hypothesis_lang)
        out = pd.DataFrame({"id": df["id"].astype(str)})
        for i, lab in enumerate(LABELS):
            out[f"p_{lab.lower()}"] = probs[:, i]
        out["pred"] = [LABELS[i] for i in probs.argmax(1)]
        dest = outdir / f"probs_nli_{lang}.csv"
        out.to_csv(dest, index=False)
        print(f"wrote {dest}")
        print(f"  distribution: {out['pred'].value_counts().to_dict()}")
        print(f"  mean confidence: {probs.max(1).mean():.3f}")


if __name__ == "__main__":
    main()
