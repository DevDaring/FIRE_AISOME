"""Build and validate the official AISoMe 2026 submission ZIP.

THE FORMAT IS NOT WHAT A KAGGLE-SHAPED PIPELINE WOULD PRODUCE. From the organizers'
email of 22 July 2026:

    id | model1_label | model2_label | model3_label
    1  |     ...      |     ...      |     ...

  * ONE FILE PER LANGUAGE — Hindi and Bengali separately (500 comments each).
  * `.csv` or `.xlsx`, both files compressed into a **single ZIP**.
  * A maximum of **3 classifiers**, side by side as three columns of the same file —
    *not* three separate run files.
  * "Evaluation will be based strictly on these submitted outputs."

Getting this wrong is a zero, so this script refuses to write a ZIP unless every check
passes:

  1. every id in the organizers' original test file is present, exactly once,
     in the original order and with the original string form (no int coercion,
     no reindexing, no sorting);
  2. row count matches the source file (500 expected per language);
  3. every label is exactly one of Favour / Against / None — no NaN, no casing drift,
     no "favour ";
  4. at least `model1_label` is populated; absent models are reported loudly rather
     than silently filled;
  5. the per-model label distributions are printed, because an all-`None` column is a
     bug that otherwise reaches the organizers.

Usage
-----
    python3.12 src/make_submission.py \
        --hi-test Dataset/Testing_Data/<hindi file> \
        --bn-test Dataset/Testing_Data/<bengali file> \
        --hi model1=artifacts/probs_fused_hi.csv \
             model2=artifacts/probs_setu_hi.cal.csv \
             model3=artifacts/committee_hi.csv \
        --bn model1=artifacts/probs_fused_bn.csv \
             model2=artifacts/probs_setu_bn.cal.csv \
             model3=artifacts/committee_bn.csv \
        --team-name <your team name>
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

from common import LABELS, SUBMISSION_DIR, normalize_label, read_table, write_json
from normalize import detect_columns

MODEL_COLS = ["model1_label", "model2_label", "model3_label"]
PCOLS = [f"p_{l.lower()}" for l in LABELS]


def load_predictions(path: str) -> pd.Series:
    """Read any SETU output frame -> Series(id -> label).

    Accepts a probability frame (p_favour/...), a `pred` column, a
    `committee_label` column, or a bare two-column id/label file.
    """
    df = read_table(path)
    if "id" not in df.columns:
        idc, _ = detect_columns(df)
        if idc is None:
            raise SystemExit(f"{path}: no id column")
        df = df.rename(columns={idc: "id"})
    ids = df["id"].astype(str).str.strip()

    if all(c in df.columns for c in PCOLS):
        labels = [LABELS[i] for i in df[PCOLS].to_numpy().argmax(axis=1)]
    else:
        col = next((c for c in ("pred", "label", "Label", "committee_label",
                                "stance", "prediction") if c in df.columns), None)
        if col is None:
            raise SystemExit(f"{path}: found neither {PCOLS} nor a label column "
                             f"(has {list(df.columns)})")
        labels = [normalize_label(v, default="None") for v in df[col]]
    return pd.Series(labels, index=ids)


def source_ids(test_path: str) -> pd.Series:
    """The organizers' ids, exactly as they appear, in file order."""
    df = read_table(test_path)
    id_col, _ = detect_columns(df)
    if id_col is None:
        print(f"  {Path(test_path).name}: no id column detected — "
              f"synthesising 1..N (verify this against the file!)")
        return pd.Series([str(i + 1) for i in range(len(df))])
    return df[id_col].astype(str).str.strip()


def build_language(lang: str, test_path: str, model_specs: list[str],
                   strict: bool) -> tuple[pd.DataFrame, dict]:
    ids = source_ids(test_path)
    print(f"\n=== {lang.upper()} === {Path(test_path).name}: {len(ids)} ids")
    if len(ids) != 500:
        msg = f"{lang}: expected 500 comments, found {len(ids)}"
        print(f"  WARNING: {msg}")
        if strict and len(ids) == 0:
            raise SystemExit(msg)
    if ids.duplicated().any():
        raise SystemExit(f"{lang}: the source test file has duplicate ids — "
                         f"cannot build an unambiguous submission")

    out = pd.DataFrame({"id": ids})
    report: dict = {"test_file": str(test_path), "rows": len(ids), "models": {}}

    for spec in model_specs:
        if "=" not in spec:
            raise SystemExit(f"expected model<N>=PATH, got {spec!r}")
        slot, path = spec.split("=", 1)
        slot = slot.strip().lower()
        if not slot.endswith("_label"):
            slot = f"{slot}_label"
        if slot not in MODEL_COLS:
            raise SystemExit(f"slot must be one of {MODEL_COLS}, got {slot!r}")

        preds = load_predictions(path)
        missing = [i for i in ids if i not in preds.index]
        extra = len(preds) - (len(preds) - len(missing))
        if missing:
            msg = (f"{lang}/{slot}: {len(missing)} of {len(ids)} ids have no "
                   f"prediction in {path} (first few: {missing[:5]})")
            if strict:
                raise SystemExit("ABORT — " + msg)
            print(f"  WARNING: {msg} — filling with 'None'")
        col = [preds.get(i, "None") for i in ids]
        col = [normalize_label(v, default="None") for v in col]
        out[slot] = col
        dist = pd.Series(col).value_counts().to_dict()
        report["models"][slot] = {"source": str(path), "distribution": dist,
                                  "missing_ids": len(missing)}
        print(f"  {slot:14} <- {path}")
        print(f"  {'':14}    {dist}")
        if len(dist) == 1:
            print(f"  WARNING: {slot} predicts a single class for every comment — "
                  f"that is almost certainly a bug, and macro-F1 will be <= 0.25")

    present = [c for c in MODEL_COLS if c in out.columns]
    if not present:
        raise SystemExit(f"{lang}: no models supplied")
    if "model1_label" not in present:
        raise SystemExit(f"{lang}: model1_label is required (got {present})")
    # keep the official column order, only for slots actually supplied
    out = out[["id"] + [c for c in MODEL_COLS if c in out.columns]]

    bad = {c: sorted(set(out[c]) - set(LABELS)) for c in present}
    bad = {c: v for c, v in bad.items() if v}
    if bad:
        raise SystemExit(f"{lang}: labels outside {LABELS}: {bad}")
    if out.isna().any().any():
        raise SystemExit(f"{lang}: NaN present in the submission frame")

    if len(present) > 1:
        agree = (out[present].nunique(axis=1) == 1).mean()
        report["all_models_agree_frac"] = round(float(agree), 4)
        print(f"  all supplied models agree on {agree:.1%} of comments")
    return out, report


def main():
    ap = argparse.ArgumentParser(description="Build the official submission ZIP")
    ap.add_argument("--hi-test", default=None,
                    help="the organizers' Hindi test file (source of truth for ids)")
    ap.add_argument("--bn-test", default=None)
    ap.add_argument("--hi", nargs="*", default=[], metavar="modelN=PATH")
    ap.add_argument("--bn", nargs="*", default=[], metavar="modelN=PATH")
    ap.add_argument("--format", default="csv", choices=["csv", "xlsx"])
    ap.add_argument("--team-name", default="Nirnay")
    ap.add_argument("--outdir", default=str(SUBMISSION_DIR))
    ap.add_argument("--zip-name", default=None)
    ap.add_argument("--no-strict", action="store_true",
                    help="warn instead of aborting on missing predictions "
                         "(NOT recommended for the real submission)")
    args = ap.parse_args()

    if not args.hi_test and not args.bn_test:
        from common import find_test_files
        found = find_test_files()
        args.hi_test = args.hi_test or (str(found["hi"]) if found["hi"] else None)
        args.bn_test = args.bn_test or (str(found["bn"]) if found["bn"] else None)
    if not (args.hi_test and args.bn_test):
        print("WARNING: the submission is supposed to contain BOTH a Hindi and a "
              "Bengali file.")
        if not args.hi_test and not args.bn_test:
            raise SystemExit("no test files given or found in Dataset/Testing_Data/")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    strict = not args.no_strict
    files, reports = [], {}

    for lang, test_path, specs in (("hindi", args.hi_test, args.hi),
                                   ("bengali", args.bn_test, args.bn)):
        if not test_path:
            continue
        if not specs:
            raise SystemExit(f"--{lang[:2]}-test given but no --{lang[:2]} model specs")
        df, rep = build_language(lang, test_path, specs, strict)
        dest = outdir / f"{args.team_name}_{lang}.{args.format}"
        if args.format == "xlsx":
            try:
                df.to_excel(dest, index=False)
            except ImportError:
                raise SystemExit("xlsx output needs openpyxl "
                                 "(`python3.12 -m pip install openpyxl`) — "
                                 "or use --format csv, which is equally acceptable")
        else:
            df.to_csv(dest, index=False)
        files.append(dest)
        reports[lang] = rep
        print(f"  -> {dest}")

    zip_path = outdir / (args.zip_name or f"{args.team_name}_AISoMe2026_submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)

    write_json(outdir / "submission_report.json",
               {"zip": str(zip_path), "files": [f.name for f in files],
                "team": args.team_name, "languages": reports})

    print(f"\n{'='*70}")
    print(f"ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            print(f"  {info.filename}  ({info.file_size} bytes)")
    print("\nFinal manual check before uploading:")
    print("  1. open each file — is the header exactly "
          "`id,model1_label,model2_label,model3_label`?")
    print("  2. do the ids match the organizers' file, same order, 500 rows each?")
    print("  3. is `model1_label` your strongest system? (winners are decided on the "
          "BEST run, so slot order does not matter for ranking — but keep the mapping "
          "recorded for the working notes)")
    print("  4. note which run is which in submission_report.json — you will need it "
          "for the CEUR working notes on 20 Sep.")


if __name__ == "__main__":
    main()
