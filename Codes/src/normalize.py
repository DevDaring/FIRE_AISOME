"""Script-aware preprocessing for Indic YouTube comments.

Why this exists
---------------
Indian YouTube comments are not clean Devanagari/Bengali prose. A large fraction is
**Roman-script code-mix** — "bhai ye sab bakwas hai", "eta puro bhondami", often mixed
with English clauses. Two consequences:

  * MuRIL was pretrained on transliterated pairs and handles Roman-script Indic text
    relatively well; XLM-R degrades noticeably on it, and any pipeline that
    machine-translates assuming native script mistranslates it outright.
  * IndicTrans2 and most MT APIs expect native script for hi/bn source.

So we detect script per comment and route accordingly, optionally transliterating
Roman -> native so that the native-script models see in-distribution input. We keep
**both** the raw and the normalised text: the working notes reports the effect of
normalisation as an ablation, and the fusion can use whichever variant each channel
prefers.

Everything degrades gracefully: `indic_transliteration` is optional, and if it is
absent we fall back to LLM-based transliteration (cached) or simply pass text through.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from common import ARTIFACTS_DIR, read_table

# Unicode blocks
_DEVA = re.compile(r"[ऀ-ॿ]")
_BENG = re.compile(r"[ঀ-৿]")
_LATIN = re.compile(r"[A-Za-z]")

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@[\w.\-]+")
_WS = re.compile(r"\s+")
_REPEAT_CHAR = re.compile(r"(.)\1{3,}")          # sooooo -> sooo
_REPEAT_PUNCT = re.compile(r"([!?.,])\1{2,}")    # !!!!! -> !!!

# Function words that discriminate romanised Hindi from romanised Bengali.
# Deliberately small and high-precision — this is a router, not a classifier.
_ROMAN_HI = {
    "hai", "hain", "nahi", "nahin", "kya", "kyu", "kyun", "aur", "yeh", "ye", "wo",
    "woh", "bhai", "kar", "karo", "karna", "raha", "rahi", "rahe", "tha", "thi",
    "hoga", "hona", "sab", "kuch", "bahut", "bohot", "acha", "accha", "bakwas",
    "sahi", "galat", "abhi", "mein", "main", "hum", "aap", "tum", "log", "logo",
    "matlab", "isliye", "lekin", "magar", "jyada", "zyada", "paisa", "sarkar",
    "desh", "duniya", "garmi", "baarish", "barish", "paani", "hamesha", "kabhi",
}
_ROMAN_BN = {
    "ache", "achhe", "nei", "naa", "kore", "korche", "korbe", "hocche", "hoche",
    "bhalo", "kharap", "keno", "kintu", "ebong", "amar", "amader", "tumi", "tomar",
    "apni", "eta", "ota", "sheta", "sob", "onek", "khub", "aar", "aaro", "bondhu",
    "dada", "didi", "bhondami", "bakwas", "jano", "janina", "hobe", "hoye", "chai",
    "britti", "brishti", "gorom", "sarkar", "desh", "prithibi", "manush", "jol",
}

_EMOJI_HINT = {
    "🔥": " fire hot ", "🌍": " earth ", "🌏": " earth ", "🌎": " earth ",
    "😂": " laughing ", "🤣": " laughing ", "😭": " crying ", "😡": " angry ",
    "🙏": " thanks respect ", "👍": " agree good ", "👎": " disagree bad ",
    "❤": " love ", "💯": " strongly agree ", "🤡": " mocking clown ",
    "🌡": " temperature ", "☀": " sun heat ", "🌧": " rain ", "🥵": " very hot ",
}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_script(text: str) -> str:
    """'deva' | 'beng' | 'latin' | 'mixed' | 'other' by character counts."""
    t = str(text)
    d, b, l = len(_DEVA.findall(t)), len(_BENG.findall(t)), len(_LATIN.findall(t))
    total = d + b + l
    if total == 0:
        return "other"
    indic = d + b
    if indic and l and min(indic, l) / total > 0.2:
        return "mixed"
    if d >= b and d > l:
        return "deva"
    if b > d and b > l:
        return "beng"
    return "latin"


def detect_lang(text: str, hint: str | None = None) -> str:
    """'hi' | 'bn' | 'en' — script first, then romanised function words.

    `hint` is the language of the file the comment came from (we know it: the
    organizers ship separate Hindi and Bengali test files). It settles romanised
    cases that the word lists cannot, which is the majority of them.
    """
    script = detect_script(text)
    if script == "deva":
        return "hi"
    if script == "beng":
        return "bn"
    if script in ("latin", "mixed", "other"):
        toks = set(re.findall(r"[a-z]+", str(text).lower()))
        hi_hits, bn_hits = len(toks & _ROMAN_HI), len(toks & _ROMAN_BN)
        if hi_hits or bn_hits:
            if hi_hits > bn_hits:
                return "hi"
            if bn_hits > hi_hits:
                return "bn"
        if hint in ("hi", "bn"):
            return hint
        return "en"
    return hint or "en"


def is_romanised_indic(text: str, hint: str | None = None) -> bool:
    """Latin-script text that is (probably) Hindi/Bengali rather than English."""
    if detect_script(text) not in ("latin", "mixed"):
        return False
    toks = set(re.findall(r"[a-z]+", str(text).lower()))
    return bool(toks & (_ROMAN_HI | _ROMAN_BN))


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str, expand_emoji: bool = True) -> str:
    """Light, stance-preserving normalisation.

    Deliberately conservative: we do NOT strip emoji, casing or punctuation
    wholesale, because "!!!!" and 🤡 carry stance. We only tame the extremes that
    blow up subword tokenisation.
    """
    t = unicodedata.normalize("NFC", str(text))
    t = _URL.sub(" <url> ", t)
    t = _MENTION.sub(" <user> ", t)
    t = t.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    t = t.replace("<br>", " ").replace("<br />", " ")
    t = _REPEAT_CHAR.sub(r"\1\1\1", t)
    t = _REPEAT_PUNCT.sub(r"\1\1", t)
    if expand_emoji:
        for e, hint in _EMOJI_HINT.items():
            if e in t:
                t = t.replace(e, hint)
    t = _WS.sub(" ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Roman -> native transliteration
# ---------------------------------------------------------------------------
def _translit_local(text: str, lang: str) -> str | None:
    """indic_transliteration ITRANS/HK round-trip. Optional dependency."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        return None
    target = sanscript.DEVANAGARI if lang == "hi" else sanscript.BENGALI
    try:
        return transliterate(text, sanscript.ITRANS, target)
    except Exception:
        return None


def _translit_llm(texts: list[str], lang: str, provider: str = "gemini",
                  workers: int = 6) -> list[str]:
    """LLM transliteration — more robust than rule-based on real code-mix, and
    cached, so it is paid for once."""
    from common import LANG_NAMES
    from llm import get_client
    cli = get_client(provider)
    script = "Devanagari" if lang == "hi" else "Bengali"
    system = (f"You transliterate Roman-script {LANG_NAMES[lang]} social-media text into "
              f"{script} script. Keep English words that are genuinely English (brand "
              f"names, 'climate change', 'AC') in Latin script. Preserve emojis, "
              f"punctuation and meaning exactly. Do NOT translate. Do NOT explain. "
              f"Output only the transliterated text.")
    prompts = [f"Transliterate:\n{t}" for t in texts]
    outs = cli.chat_many(prompts, system=system, temperature=0.0, max_tokens=512,
                         workers=workers, desc=f"translit->{lang}")
    return [(o or src) for o, src in zip(outs, texts)]


def transliterate_frame(df, text_col: str, lang_col: str = "lang",
                        method: str = "auto", provider: str = "gemini",
                        workers: int = 6, out_col: str = "text_native"):
    """Add `out_col`: romanised Indic rows converted to native script, others copied."""
    texts = df[text_col].astype(str).tolist()
    langs = df[lang_col].astype(str).tolist() if lang_col in df.columns \
        else ["hi"] * len(texts)
    out = list(texts)
    for lang in ("hi", "bn"):
        idx = [i for i, (t, l) in enumerate(zip(texts, langs))
               if l == lang and is_romanised_indic(t, hint=lang)]
        if not idx:
            continue
        print(f"  transliterating {len(idx)} romanised {lang} rows (method={method})")
        sub = [texts[i] for i in idx]
        done = None
        if method in ("auto", "local"):
            loc = [_translit_local(t, lang) for t in sub]
            if all(x is not None for x in loc):
                done = loc
            elif method == "local":
                print("    indic_transliteration unavailable; leaving as-is")
                done = sub
        if done is None:
            done = _translit_llm(sub, lang, provider=provider, workers=workers)
        for i, v in zip(idx, done):
            out[i] = v
    df[out_col] = out
    return df


# ---------------------------------------------------------------------------
# CLI: normalise a test file into a canonical frame
# ---------------------------------------------------------------------------
# matched case-insensitively — the official test files use ID / COMMENT
ID_CANDIDATES = ["id", "sl", "sl_no", "sr_no", "sno", "s_no", "index", "uid",
                 "comment_id", "post_id"]
TEXT_CANDIDATES = ["text", "comment", "comments", "sentence", "tweet", "post",
                   "content", "body"]


def detect_columns(df, id_col: str = "auto", text_col: str = "auto"):
    from common import is_text_col
    lower = {str(c).strip().lower(): c for c in df.columns}
    if id_col == "auto":
        id_col = next((lower[c] for c in ID_CANDIDATES if c in lower), None)
        if id_col is None:
            uniq = [c for c in df.columns
                    if df[c].is_unique and df[c].astype(str).str.len().mean() < 25]
            id_col = uniq[0] if uniq else None
    if text_col == "auto":
        text_col = next((lower[c] for c in TEXT_CANDIDATES if c in lower), None)
        if text_col is None:
            obj = [c for c in df.columns if is_text_col(df[c]) and c != id_col]
            if not obj:
                raise SystemExit(f"no text column found in {list(df.columns)}")
            text_col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    return id_col, text_col


def normalize_test_file(path, lang_hint: str, translit: str = "none",
                        provider: str = "gemini", out: Path | None = None):
    """Read a raw test file -> canonical frame [id, text_raw, text, lang, script].

    `id` is preserved EXACTLY as it appears in the source file (as a string), because
    the submission must join back to the organizers' ids.
    """
    import pandas as pd
    df = read_table(path)
    id_col, text_col = detect_columns(df)
    print(f"{Path(path).name}: rows={len(df)} id={id_col!r} text={text_col!r} "
          f"cols={list(df.columns)}")

    ids = (df[id_col].astype(str).str.strip() if id_col
           else pd.Series([str(i + 1) for i in range(len(df))]))
    raw = df[text_col].astype(str)
    out_df = pd.DataFrame({
        "id": ids,
        "text_raw": raw,
        "text": raw.map(clean_text),
    })
    out_df["script"] = out_df["text"].map(detect_script)
    out_df["lang"] = [detect_lang(t, hint=lang_hint) for t in out_df["text"]]
    out_df["file_lang"] = lang_hint
    out_df["romanised"] = [is_romanised_indic(t, hint=lang_hint)
                           for t in out_df["text"]]

    if translit != "none":
        out_df = transliterate_frame(out_df, "text", "file_lang", method=translit,
                                     provider=provider, out_col="text_native")
    else:
        out_df["text_native"] = out_df["text"]

    if out_df["id"].duplicated().any():
        n = int(out_df["id"].duplicated().sum())
        print(f"  WARNING: {n} duplicate ids in the source file — kept as-is")

    print(f"  script mix: {out_df['script'].value_counts().to_dict()}")
    print(f"  detected lang: {out_df['lang'].value_counts().to_dict()}")
    print(f"  romanised Indic: {int(out_df['romanised'].sum())}/{len(out_df)}")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out, index=False)
        print(f"  wrote -> {out}")
    return out_df


def main():
    ap = argparse.ArgumentParser(description="Normalise the Hindi/Bengali test files")
    ap.add_argument("--hi", help="path to the Hindi test file")
    ap.add_argument("--bn", help="path to the Bengali test file")
    ap.add_argument("--translit", default="none",
                    choices=["none", "auto", "local", "llm"],
                    help="romanised Indic -> native script")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--outdir", default=str(ARTIFACTS_DIR))
    args = ap.parse_args()

    if not args.hi and not args.bn:
        from common import find_test_files
        found = find_test_files()
        args.hi = args.hi or (str(found["hi"]) if found["hi"] else None)
        args.bn = args.bn or (str(found["bn"]) if found["bn"] else None)
        if found["unknown"]:
            print(f"unclassified files in Testing_Data: "
                  f"{[p.name for p in found['unknown']]}")
    if not args.hi and not args.bn:
        raise SystemExit("no test files found — put them in Dataset/Testing_Data/ "
                         "or pass --hi/--bn")

    outdir = Path(args.outdir)
    for lang, path in (("hi", args.hi), ("bn", args.bn)):
        if not path:
            print(f"(no {lang} file)")
            continue
        normalize_test_file(path, lang, translit=args.translit,
                            provider=args.provider,
                            out=outdir / f"test_{lang}.csv")


if __name__ == "__main__":
    main()
