"""Shared constants, paths, .env handling and metric helpers for SETU.

SETU = Stance via Evidence-Taxonomy Unification (सेतु / সেতু = "bridge").
See ../STRATEGY.md for the design rationale.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — the dataset is organised as Dataset/{Training_Data,Testing_Data}/
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent                          # Codes/
DATASET_DIR = BASE_DIR / "Dataset"
TRAIN_DATA_DIR = DATASET_DIR / "Training_Data"
TEST_DATA_DIR = DATASET_DIR / "Testing_Data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
SUBMISSION_DIR = ARTIFACTS_DIR / "submission"
ENV_FILE = BASE_DIR / ".env"

for _d in (ARTIFACTS_DIR, CACHE_DIR, SUBMISSION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Labels — the submission must use exactly these strings
# ---------------------------------------------------------------------------
LABELS = ["Favour", "Against", "None"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

TARGET_CLAIM_EN = "Climate change and global warming is a serious concern"
TARGET_CLAIM_HI = "जलवायु परिवर्तन और ग्लोबल वार्मिंग एक गंभीर चिंता का विषय है"
TARGET_CLAIM_BN = "জলবায়ু পরিবর্তন এবং বিশ্ব উষ্ণায়ন একটি গুরুতর উদ্বেগের বিষয়"
TARGET_CLAIM = {"en": TARGET_CLAIM_EN, "hi": TARGET_CLAIM_HI, "bn": TARGET_CLAIM_BN}

LANG_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali"}

# Every label spelling seen across GWSD / SemEval / LLM output.
_LABEL_ALIASES = {
    "favour": "Favour", "favor": "Favour", "favours": "Favour", "favors": "Favour",
    "agrees": "Favour", "agree": "Favour", "pro": "Favour", "support": "Favour",
    "supports": "Favour", "for": "Favour",
    "against": "Against", "disagrees": "Against", "disagree": "Against",
    "anti": "Against", "oppose": "Against", "opposes": "Against", "con": "Against",
    "none": "None", "neutral": "None", "neither": "None", "unrelated": "None",
    "no stance": "None", "nostance": "None", "other": "None", "na": "None",
}


def normalize_label(raw, default: str | None = None) -> str:
    """Map any observed label spelling to a canonical label.

    `default` (if given) is returned instead of raising on unknown input — used
    when parsing LLM output, where a hard failure would be worse than a fallback.
    """
    key = str(raw).strip().strip(".\"'").lower()
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    # tolerate "Stance: Against", "label = favour", "**None**"
    for alias, canon in _LABEL_ALIASES.items():
        if key.endswith(alias) or key.startswith(alias):
            return canon
    if default is not None:
        return default
    raise ValueError(f"Unknown label: {raw!r}")


# ---------------------------------------------------------------------------
# .env loading (manual parse; no python-dotenv dependency)
# ---------------------------------------------------------------------------
_ENV_CACHE: dict | None = None


def load_env(path: Path = ENV_FILE, refresh: bool = False) -> dict:
    """Parse KEY=VALUE lines into a dict and into os.environ (non-destructive)."""
    global _ENV_CACHE
    if _ENV_CACHE is not None and not refresh:
        return _ENV_CACHE
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if not k or not v:
                continue
            env[k] = v
            os.environ.setdefault(k, v)
    _ENV_CACHE = env
    return env


def env_keys(*prefixes: str) -> list[str]:
    """All values whose key starts with any of `prefixes`, for API-key rotation.

    The .env in this repo stores several keys per provider as
    ``GEMINI_API_KEY_1 … GEMINI_API_KEY_4``, so a plain ``env["GEMINI_API_KEY"]``
    lookup finds nothing. Returns de-duplicated values in sorted key order.
    """
    env = load_env()
    out, seen = [], set()
    for k in sorted(env):
        if any(k.lower().startswith(p.lower()) for p in prefixes):
            v = env[k]
            if v and v not in seen and not v.startswith("http"):
                seen.add(v)
                out.append(v)
    return out


SEED = int(load_env().get("RANDOM_SEED", "42") or 42)


def set_seed(seed: int = SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device_str() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                          zero_division=0))


def score_report(y_true, y_pred, title: str = "") -> dict:
    """Print and return the per-class breakdown that the working notes needs."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    rep = {
        "n": len(y_true),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         labels=LABELS, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class_f1": {
            l: round(float(f), 4) for l, f in zip(
                LABELS, f1_score(y_true, y_pred, average=None, labels=LABELS,
                                 zero_division=0))
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
    if title:
        print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=4,
                                zero_division=0))
    print(f"macro-F1 = {rep['macro_f1']:.4f}   (rows={rep['n']})")
    print(f"confusion (rows=gold {LABELS}): {rep['confusion']}")
    return rep


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def is_text_col(series) -> bool:
    """True for string-ish columns under BOTH pandas <3 (object) and >=3 (str dtype).

    pandas 3.0 infers the dedicated ``str`` dtype instead of ``object``, so the
    classic ``df[c].dtype == object`` test silently matches nothing and every
    auto-detected text column disappears. Use this everywhere instead.
    """
    from pandas.api.types import is_object_dtype, is_string_dtype
    return bool(is_string_dtype(series) or is_object_dtype(series))


def read_csv(path, **kw):
    """pandas.read_csv that does NOT destroy the label "None".

    One of our three class labels is the literal string ``None``, which pandas
    treats as a missing-value marker by default. Reading a labelled CSV with a
    plain ``pd.read_csv`` therefore turns every ``None`` row into NaN and silently
    drops a third of the data — it looks like a two-class problem and macro-F1
    caps out around 0.45 with no error anywhere. Every CSV read in this repo must
    go through here (or pass the same two arguments).
    """
    import pandas as pd
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", [""])
    return pd.read_csv(path, **kw)


def read_table(path, sep=None):
    """Read csv/tsv/txt/xlsx into a DataFrame, sniffing the separator.

    Same "None"-preserving semantics as :func:`read_csv`.
    """
    import pandas as pd
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, keep_default_na=False, na_values=[""])
    if sep is not None:
        return read_csv(p, sep=sep)
    if p.suffix.lower() in (".tsv",):
        return read_csv(p, sep="\t")
    return read_csv(p, sep=None, engine="python")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def find_test_files() -> dict:
    """Locate the Hindi and Bengali test files under Dataset/Testing_Data/.

    Returns {'hi': Path|None, 'bn': Path|None, 'unknown': [Path, ...]}.
    Matching is by filename keyword first, then by script content of the file.
    """
    out = {"hi": None, "bn": None, "unknown": []}
    if not TEST_DATA_DIR.exists():
        return out
    cands = [p for p in sorted(TEST_DATA_DIR.iterdir())
             if p.suffix.lower() in (".csv", ".tsv", ".txt", ".xlsx", ".xls")]
    for p in cands:
        name = p.name.lower()
        if any(k in name for k in ("hindi", "_hi", "-hi", "hi_", "devanagari")):
            out["hi"] = out["hi"] or p
        elif any(k in name for k in ("bengali", "bangla", "_bn", "-bn", "bn_")):
            out["bn"] = out["bn"] or p
        else:
            out["unknown"].append(p)
    # fall back to sniffing the dominant script
    if out["unknown"] and (out["hi"] is None or out["bn"] is None):
        from normalize import detect_script
        still_unknown = []
        for p in out["unknown"]:
            try:
                df = read_table(p)
                texts = df.astype(str).agg(" ".join, axis=1).tolist()[:200]
                scripts = [detect_script(t) for t in texts]
                dev = sum(s == "deva" for s in scripts)
                ben = sum(s == "beng" for s in scripts)
                lang = "hi" if dev > ben else ("bn" if ben > dev else None)
                if lang and out[lang] is None:
                    out[lang] = p
                    continue
            except Exception:
                pass
            still_unknown.append(p)
        out["unknown"] = still_unknown
    return out
