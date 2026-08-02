"""Batched, cached machine translation for the cross-lingual channels.

Two directions are needed:
  * translate-train : English pool -> Hindi + Bengali  (augmentation for the encoder)
  * translate-test  : Hindi/Bengali test -> English    (for English-only channels)

What changed vs. a naive per-row translator
-------------------------------------------
  * **Batched**: ~20 texts per LLM call in a numbered JSON envelope, so translating
    2.7k rows to two languages is ~270 calls instead of ~5 400. That is the
    difference between "runs inside a free tier in 15 minutes" and "rate-limited
    for hours".
  * **Key rotation** through llm.py, which reads GEMINI_API_KEY_1..4 etc. The old
    code looked for a bare `GEMINI_API_KEY`, found nothing, and silently fell back
    to a 1 GB CPU model — the single most expensive bug in the previous pipeline.
  * **Stance-preserving prompt**: negation, sarcasm and hedging are the whole signal
    in stance detection; a translator that smooths "ye sab bakwas hai" into "this is
    a matter of concern" destroys the label.
  * Per-row cache, so interrupted runs resume for free.

Local IndicTrans2 remains available as an offline fallback (`--provider local`).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import ARTIFACTS_DIR, LANG_NAMES, read_table

BATCH = 20


# ---------------------------------------------------------------------------
# LLM-backed batched translator
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a professional translator specialising in Indian social-media text.\n"
    "Rules:\n"
    "1. Translate meaning AND stance exactly. Never soften, never neutralise, never "
    "add or remove negation. If the source mocks, sneers or sarcastically agrees, the "
    "translation must mock, sneer or sarcastically agree.\n"
    "2. Keep the informal register of a YouTube comment. Do not make it literary.\n"
    "3. Preserve emojis, ALL-CAPS, elongations and punctuation intensity (!!!).\n"
    "4. Keep proper nouns, brand names and established English terms "
    "(e.g. 'climate change', 'AC', 'IPCC') as they are.\n"
    "5. Return ONLY the JSON object described by the user. No commentary."
)


def _batch_prompt(items: list[tuple[int, str]], src: str, tgt: str) -> str:
    listing = "\n".join(f'  "{i}": {json.dumps(t, ensure_ascii=False)}'
                        for i, t in items)
    return (f"Translate each numbered {LANG_NAMES[src]} text into "
            f"{LANG_NAMES[tgt]}.\n\n"
            f"Input:\n{{\n{listing}\n}}\n\n"
            f'Return exactly: {{"translations": {{"<same numbers>": '
            f'"<translation>", ...}}}}\n'
            f"Every input number must appear exactly once in the output.")


class LLMTranslator:
    def __init__(self, provider: str = "gemini", model: str | None = None,
                 batch: int = BATCH, workers: int = 6):
        from llm import get_client
        self.cli = get_client(provider, model)
        self.name = f"{self.cli.name}:{self.cli.model}"
        self.batch = batch
        self.workers = workers

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        chunks = [list(enumerate(texts))[i:i + self.batch]
                  for i in range(0, len(texts), self.batch)]
        prompts = [_batch_prompt(c, src, tgt) for c in chunks]
        results = self.cli.chat_many(prompts, system=_SYSTEM, temperature=0.0,
                                     max_tokens=min(4096, 220 * self.batch),
                                     workers=self.workers, as_json=True,
                                     desc=f"{src}->{tgt}")
        out = list(texts)                      # fall back to source on failure
        missing = []
        for chunk, res in zip(chunks, results):
            table = (res or {}).get("translations") if isinstance(res, dict) else None
            if not isinstance(table, dict):
                missing.extend(i for i, _ in chunk)
                continue
            for i, src_text in chunk:
                v = table.get(str(i), table.get(i))
                if isinstance(v, str) and v.strip():
                    out[i] = v.strip()
                else:
                    missing.append(i)
        if missing:
            print(f"  retrying {len(missing)} rows individually")
            for i in tqdm(missing, desc="single-row retry"):
                r = self.cli.chat_json(
                    _batch_prompt([(i, texts[i])], src, tgt), system=_SYSTEM,
                    temperature=0.0, max_tokens=512, cache_salt="single")
                table = (r or {}).get("translations") if isinstance(r, dict) else None
                if isinstance(table, dict):
                    v = table.get(str(i), table.get(i))
                    if isinstance(v, str) and v.strip():
                        out[i] = v.strip()
        return out


# ---------------------------------------------------------------------------
# Offline fallback: ai4bharat IndicTrans2 distilled (CPU, ~1 GB download)
# ---------------------------------------------------------------------------
class LocalIndicTrans:
    name = "indictrans2-dist-200M"
    _flores = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng"}

    def __init__(self):
        self._models = {}

    def _load(self, src):
        direction = "en-indic" if src == "en" else "indic-en"
        if direction not in self._models:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            name = f"ai4bharat/indictrans2-{direction}-dist-200M"
            print(f"loading {name} (first use downloads ~1 GB)...")
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            mod = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True)
            mod.eval()
            self._models[direction] = (tok, mod)
        return self._models[direction]

    def translate(self, texts, src, tgt, batch_size=16):
        import torch
        tok, mod = self._load(src)
        sc, tc = self._flores[src], self._flores[tgt]
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"indictrans {src}->{tgt}"):
            chunk = texts[i:i + batch_size]
            try:
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=256, src_lang=sc, tgt_lang=tc)
            except TypeError:
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=256)
            with torch.no_grad():
                gen = mod.generate(**enc, max_length=256, num_beams=4)
            out.extend(t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True))
        return out


def get_translator(provider: str = "gemini", model: str | None = None,
                   batch: int = BATCH, workers: int = 6):
    if provider == "local":
        return LocalIndicTrans()
    if provider == "auto":
        from llm import available_providers
        avail = available_providers()
        for pref in ("gemini", "deepseek", "openrouter", "mistral"):
            if pref in avail:
                provider = pref
                break
        else:
            print("no LLM provider keys usable — falling back to local IndicTrans2")
            return LocalIndicTrans()
    return LLMTranslator(provider, model, batch=batch, workers=workers)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Batched MT for the SETU pipeline")
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="text",
                    help="'auto' = longest average object column")
    ap.add_argument("--to", nargs="+", required=True, choices=["en", "hi", "bn"])
    ap.add_argument("--from-lang", default="en", choices=["en", "hi", "bn", "auto"],
                    help="'auto' = per-row script/word detection")
    ap.add_argument("--provider", default="gemini",
                    choices=["auto", "gemini", "deepseek", "mistral", "openrouter",
                             "llama", "gemma", "nanogpt", "local"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", default=None,
                    help="explicit output path (only valid with a single --to)")
    ap.add_argument("--max-rows", type=int, default=0, help="smoke test with N rows")
    args = ap.parse_args()

    inp = Path(args.input)
    df = read_table(inp)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    if args.text_col == "auto":
        from common import is_text_col
        obj = [c for c in df.columns if is_text_col(df[c])]
        args.text_col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
        print(f"auto text column: {args.text_col!r}")
    texts = df[args.text_col].astype(str).tolist()

    tr = get_translator(args.provider, args.model, batch=args.batch,
                        workers=args.workers)
    print(f"translator: {tr.name} | rows: {len(texts)}")

    if args.from_lang == "auto":
        from normalize import detect_lang
        srcs = [detect_lang(t) for t in texts]
    else:
        srcs = [args.from_lang] * len(texts)

    for tgt in args.to:
        out = list(texts)
        for src in sorted(set(srcs)):
            if src == tgt:
                continue
            idx = [i for i, s in enumerate(srcs) if s == src]
            if not idx:
                continue
            done = tr.translate([texts[i] for i in idx], src, tgt)
            for i, v in zip(idx, done):
                out[i] = v
        col = f"text_{tgt}"
        df[col] = out
        path = (Path(args.output) if (args.output and len(args.to) == 1)
                else inp.with_name(f"{inp.stem}.to-{tgt}.csv"))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        n_changed = sum(a != b for a, b in zip(texts, out))
        print(f"wrote {path}  (column {col!r}, {n_changed}/{len(texts)} rows changed)")

    if hasattr(tr, "cli"):
        print("provider stats:", tr.cli.stats())


if __name__ == "__main__":
    main()
