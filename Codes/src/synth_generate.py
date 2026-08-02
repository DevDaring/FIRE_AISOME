"""Taxonomy-conditioned synthetic augmentation (SETU contribution C2).

The problem this solves
-----------------------
The permitted English pool is ~2.7k rows of which ~10-15 % is `Against`, and those
`Against` examples argue in a *US* idiom (hoax, Al Gore, liberal agenda). The test
set is Indian YouTube comments whose `Against` examples argue in a completely
different idiom (the West polluted first, development before environment, monsoon was
always erratic, yuga cycles, TRP drama). Translating GWSD into Hindi yields fluent
Hindi sentences about Al Gore — grammatically perfect and pedagogically useless.

Under macro-F1 the `Against` class carries a full third of the score, so this gap is
not a detail; it is the competition.

What this does
--------------
Walks the cartesian product

    taxonomy node x language {hi, bn} x script {native, roman} x register
        {sincere, angry, sarcastic, short_meme, question}

and asks an LLM for authentic YouTube comments for each cell. Because the stance is
fixed by the cell (`node.stance`), the corpus is **class-balanced by construction** and
every row carries a gold argument-node label for the auxiliary head in
train_transformer.py.

Diversity guards, because naive bulk generation collapses into ten paraphrases of one
sentence:
  * one call per (node, lang, script, register) cell with n comments requested, and
    temperature > 0 — cells are the diversity axis, not sampling;
  * generation seeded with that node's Indian-context cue phrases;
  * multiple generator models round-robin across cells (provenance recorded);
  * exact-dup and near-dup filtering (token Jaccard) after the fact;
  * explicit instruction to vary length, spelling errors, emoji use and code-mixing.

Output: artifacts/synth_train.csv
    id, text, label, node_id, branch, lang, script, register, generator
    p_favour, p_against, p_none        (one-hot, so it shares the train_en schema)

This script does NOT need the test data — run it first, it is the long pole.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from common import ARTIFACTS_DIR, LABELS, LANG_NAMES, SEED, set_seed, write_json
from taxonomy import NODES, Node

SCRIPTS = ["native", "roman"]
REGISTERS = {
    "sincere": "an ordinary sincere comment, plain and direct",
    "angry": "an angry or frustrated rant, possibly with CAPS and !!!",
    "sarcastic": "heavy sarcasm or mockery — the surface words may look like the "
                 "opposite of the intended stance",
    "short_meme": "very short (3-8 words), meme-like or emoji-laden, the way a "
                  "throwaway YouTube reply looks",
    "question": "phrased as a rhetorical question that still clearly carries the "
                "stance",
}

_SYSTEM = (
    "You generate realistic YouTube comments for a research dataset on climate-change "
    "stance detection in Indian languages. You write the way real Indian YouTube "
    "commenters write — informal, ungrammatical where natural, code-mixed, emoji-using, "
    "sometimes rude. You are not writing polished prose and you never write like a "
    "textbook. Return only JSON."
)


def _script_instruction(lang: str, script: str, style: str = "authentic") -> str:
    if style == "translationese":
        # Forensics on the released test files (STRATEGY.md §"Test-set forensics")
        # showed they are MACHINE-TRANSLATED English YouTube comments: pure native
        # script, zero romanisation, zero emoji, formal register, transliterated
        # names ("बिल नाई"), Indic digits ("९७%"). Matching that register matters
        # more for encoder transfer than sounding like an authentic Indian commenter.
        name = "Devanagari" if lang == "hi" else "Bengali"
        return (f"Write the comment the way a MACHINE TRANSLATION of an English "
                f"YouTube comment into {LANG_NAMES[lang]} reads: pure {name} script, "
                f"formal/bookish word choices, no Roman-script code-mixing, no emojis, "
                f"English names transliterated into {name} (e.g. 'Bill' -> "
                f"{'बिल' if lang == 'hi' else 'বিল'}), numbers sometimes in "
                f"{name} digits (e.g. ९७% / ৯৭%). Slightly stiff, literal phrasing is "
                f"GOOD here — it should read like translated text, not native slang.")
    if script == "native":
        name = "Devanagari" if lang == "hi" else "Bengali"
        return (f"Write in {LANG_NAMES[lang]} using {name} script. English loanwords "
                f"that Indians normally type in Latin script (e.g. 'climate change', "
                f"'AC', 'global warming', 'TRP') may stay in Latin script.")
    return (f"Write in ROMANISED {LANG_NAMES[lang]} — {LANG_NAMES[lang]} words typed "
            f"in the Latin alphabet, the way people actually type on a phone "
            f"(e.g. Hindi 'ye sab bakwas hai', Bengali 'eta puro bhondami'). "
            f"Mix in English words and clauses naturally, with inconsistent spelling.")


def build_prompt(node: Node, lang: str, script: str, register: str, n: int,
                 style: str = "authentic") -> str:
    cues = node.cues.get(lang) or node.cues.get("en") or []
    cue_block = ""
    if cues:
        cue_block = ("\nTypical ways this argument surfaces (paraphrase and go far "
                     "beyond these — do NOT copy them):\n"
                     + "\n".join(f"  - {c}" for c in cues[:5]))
    scene = ("a YouTube video about climate change / global warming, presented by a "
             "well-known science communicator (think Bill Nye-style explainer videos "
             "also watched for school assignments). Real viewers are commenting."
             if style == "translationese" else
             "a YouTube video about climate change / global warming in India. Real "
             "viewers are commenting.")
    return f"""\
Context: {scene}

The target claim under discussion is:
  "Climate change and global warming is a serious concern."

Write {n} DIFFERENT comments that all take the stance **{node.stance}** towards that
claim, and all use this specific argument:

  Argument id: {node.id}
  Argument:    {node.gloss}
{cue_block}

Style requirements:
  - {_script_instruction(lang, script, style)}
  - Register: {REGISTERS[register]}
  - Vary length a lot across the {n} comments (some 4 words, some 40).
  - {("Vary sub-topic and specifics: the presenter's credentials, the 97% figure, "
     "school assignments, US politics, weather anecdotes, documentaries, CO2 facts — "
     "the things commenters on a global science channel actually bring up."
     if style == "translationese" else
     "Vary sub-topic, region, spelling and emoji use. Some should mention specific "
     "Indian places, seasons, prices, politicians, festivals or news events.")}
  - They must be plausible as REAL comments, not as dataset examples. No hashtags
    unless a real commenter would use one. No numbering inside the text.
  - Do NOT restate the argument id or the word "{node.stance}" in the comment.

Critical: the stance must be **{node.stance}** towards the claim itself — not merely
positive or negative sentiment, and not an opinion about the video.

Return exactly:
{{"comments": ["...", "...", ...]}}   ({n} strings)
"""


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def dedupe(rows: list[dict], threshold: float = 0.8) -> list[dict]:
    """Exact + near-duplicate removal, scoped per (node, lang) for O(n*k) cost."""
    seen_exact: set[str] = set()
    buckets: dict[tuple, list[str]] = {}
    out = []
    for r in rows:
        t = r["text"].strip()
        key = t.lower()
        if not t or key in seen_exact:
            continue
        bk = (r["node_id"], r["lang"])
        if any(_jaccard(t, prev) >= threshold for prev in buckets.get(bk, [])):
            continue
        seen_exact.add(key)
        buckets.setdefault(bk, []).append(t)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description="Taxonomy-conditioned synthetic corpus")
    ap.add_argument("--per-cell", type=int, default=6,
                    help="comments per (node, lang, script, register) cell")
    ap.add_argument("--langs", nargs="+", default=["hi", "bn"], choices=["hi", "bn", "en"])
    ap.add_argument("--style", default="translationese",
                    choices=["translationese", "authentic"],
                    help="'translationese' mimics the released test files (MT'd English "
                         "comments: native script, formal, no emoji/code-mix); "
                         "'authentic' is organic Indian YouTube style — keep some as "
                         "an ablation/robustness slice")
    ap.add_argument("--scripts", nargs="+", default=None, choices=SCRIPTS,
                    help="default: native only for translationese, both otherwise")
    ap.add_argument("--registers", nargs="+", default=list(REGISTERS),
                    choices=list(REGISTERS))
    ap.add_argument("--generators", nargs="+", default=["gemini"],
                    help="providers to round-robin across cells, e.g. gemini deepseek")
    ap.add_argument("--nodes", nargs="+", default=None,
                    help="restrict to specific node ids (default: all)")
    ap.add_argument("--stances", nargs="+", default=None, choices=LABELS,
                    help="restrict to specific stances, e.g. --stances Against")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dedupe-threshold", type=float, default=0.8)
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "synth_train.csv"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cell plan and one example prompt, call nothing")
    args = ap.parse_args()

    set_seed(SEED)
    if args.scripts is None:
        args.scripts = ["native"] if args.style == "translationese" else SCRIPTS
    nodes = NODES
    if args.nodes:
        nodes = [n for n in nodes if n.id in set(args.nodes)]
    if args.stances:
        nodes = [n for n in nodes if n.stance in set(args.stances)]
    if not nodes:
        raise SystemExit("no taxonomy nodes selected")

    cells = [(n, lang, script, reg)
             for n in nodes
             for lang in args.langs
             for script in args.scripts
             for reg in args.registers]
    random.Random(SEED).shuffle(cells)

    per_stance = {}
    for n, lang, script, reg in cells:
        per_stance[n.stance] = per_stance.get(n.stance, 0) + args.per_cell
    print(f"{len(cells)} cells x {args.per_cell} comments "
          f"= {len(cells) * args.per_cell} target rows")
    print(f"target per stance (pre-dedupe): {per_stance}")
    print(f"generators: {args.generators}")

    if args.dry_run:
        n, lang, script, reg = cells[0]
        print(f"\n--- example prompt: {n.id} / {lang} / {script} / {reg} ---")
        print(build_prompt(n, lang, script, reg, args.per_cell, args.style))
        return

    from llm import get_client

    # group cells per generator so each provider's calls are issued concurrently
    by_gen: dict[str, list] = {g: [] for g in args.generators}
    for i, cell in enumerate(cells):
        by_gen[args.generators[i % len(args.generators)]].append(cell)

    rows: list[dict] = []
    for gen, gen_cells in by_gen.items():
        if not gen_cells:
            continue
        try:
            cli = get_client(gen)
        except SystemExit as e:
            print(f"  skipping generator {gen!r}: {e}")
            continue
        prompts = [build_prompt(n, lang, sc, reg, args.per_cell, args.style)
                   for n, lang, sc, reg in gen_cells]
        results = cli.chat_many(
            prompts, system=_SYSTEM, temperature=args.temperature,
            max_tokens=min(4096, 120 * args.per_cell + 400),
            workers=args.workers, as_json=True, desc=f"synth/{gen}")
        n_bad = 0
        for (node, lang, script, reg), res in zip(gen_cells, results):
            comments = (res or {}).get("comments") if isinstance(res, dict) else None
            if not isinstance(comments, list):
                n_bad += 1
                continue
            for j, c in enumerate(comments):
                if not isinstance(c, str) or not c.strip():
                    continue
                rows.append({
                    "id": f"synth_{node.id}_{lang}_{script}_{reg}_{j}",
                    "text": c.strip(),
                    "label": node.stance,
                    "node_id": node.id,
                    "branch": node.branch,
                    "lang": lang,
                    "script": script,
                    "register": reg,
                    "style": args.style,
                    "generator": f"{cli.name}:{cli.model}",
                })
        print(f"  {gen}: {n_bad}/{len(gen_cells)} cells returned unusable JSON")
        print(f"  {gen} stats: {cli.stats()}")

    if not rows:
        raise SystemExit("no synthetic rows produced — check API keys with "
                         "`python3.12 src/llm.py`")

    before = len(rows)
    rows = dedupe(rows, args.dedupe_threshold)
    df = pd.DataFrame(rows)
    for lab in LABELS:
        df[f"p_{lab.lower()}"] = (df["label"] == lab).astype(float)
    df["source"] = "synth"
    df["n_votes"] = 1
    df["agreement"] = 1.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    stats = {
        "rows": len(df),
        "removed_as_duplicates": before - len(df),
        "by_label": df["label"].value_counts().to_dict(),
        "by_lang": df["lang"].value_counts().to_dict(),
        "by_script": df["script"].value_counts().to_dict(),
        "by_branch": df["branch"].value_counts().to_dict(),
        "by_register": df["register"].value_counts().to_dict(),
        "nodes_covered": int(df["node_id"].nunique()),
        "nodes_total": len(NODES),
        "mean_chars": round(float(df["text"].str.len().mean()), 1),
    }
    write_json(ARTIFACTS_DIR / "synth_train.stats.json", stats)
    print(f"\nwrote {len(df)} rows -> {out}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    print("\nsample rows:")
    for _, r in df.sample(min(8, len(df)), random_state=SEED).iterrows():
        print(f"  [{r['label']:7}|{r['node_id']:24}|{r['lang']}/{r['script']:6}] "
              f"{r['text'][:90]}")


if __name__ == "__main__":
    main()
