"""Transductive taxonomy-grounded LLM committee (SETU contribution C4, part 1).

We hold the entire *unlabelled* test set at inference time. Using it is transduction,
not leakage: no gold label is ever read. So instead of predicting each comment in
isolation with one model, we run a **committee** of heterogeneous LLMs over the whole
test set and turn their (dis)agreement into two products:

  1. **soft labels** — a per-comment distribution over {Favour, Against, None},
     weighted by member reliability, used to distil the committee into the compact
     Indic encoder (selftrain.py). Distilling a committee consistently beats both any
     single member and any translate-train model.
  2. **an information-density ranking** — items the committee splits on are exactly the
     items worth spending human annotation on, which is how annotate_dev.py picks its
     ~150-comment sample.

Three design choices that matter:

* **Heterogeneous members, not N samples of one model.** Five temperature-0.3 samples
  from one model agree with themselves for the wrong reasons; five differently
  pretrained models disagree informatively. Members come from llm.MEMBERS.
* **Argument-before-stance.** Each member must first name a node from the SETU taxonomy
  and only then emit a stance. This is chain-of-thought with a *fixed, auditable
  vocabulary* — it curbs the sentiment-vs-stance confusion, and the node predictions
  become auxiliary-head targets for real (non-synthetic) comments.
* **In-language prompting.** The comment is shown in its original script with the
  target claim stated in the same language. No round-trip through English, which is
  where sarcasm dies.

Output: artifacts/committee_<lang>.csv — one row per comment with every member's vote,
plus consensus columns. Raw JSON responses (including rationales) are kept in
artifacts/committee_<lang>.raw.jsonl for the error analysis in the working notes.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from common import (ARTIFACTS_DIR, LABELS, TARGET_CLAIM, LANG_NAMES,
                    normalize_label, read_table, write_json)
from taxonomy import (ANNOTATION_GUIDELINES, NODE_IDS, UNKNOWN_NODE,
                      stance_of_node, taxonomy_prompt_block)

BATCH = 10

_SYSTEM = """\
You are an expert annotator for a stance-detection shared task at FIRE 2026. You label
Indian YouTube comments for their stance towards a fixed claim about climate change.

You are precise, you follow the codebook exactly, and you do not confuse sentiment with
stance. You return only JSON.
"""


# What forensics on the released test files established (see STRATEGY.md §"Test-set
# forensics"): the comments are machine-translated from English YouTube comments under
# climate-science videos by a well-known science communicator (Bill Nye). Telling the
# committee this resolves whole families of otherwise-ambiguous comments: credential
# attacks on the presenter, the 97%-consensus fights, "here from online class", "merch
# link in the description". Override with --context "" to run context-free (ablation).
DEFAULT_CONTEXT = (
    "Context: these comments were originally written in English under YouTube videos "
    "about climate change featuring the science communicator Bill Nye, and were later "
    "machine-translated, so the phrasing may be formal or slightly unnatural. "
    "References to 'he/sir/the presenter', to being 'not a real scientist / a "
    "mechanical engineer', to a '97%' figure, or to watching for a class assignment "
    "refer to that setting.")


def build_prompt(items: list[tuple[str, str]], lang: str,
                 context: str = DEFAULT_CONTEXT) -> str:
    """items = [(id, comment_text), ...] in the comment's original script."""
    claim_local = TARGET_CLAIM.get(lang, TARGET_CLAIM["en"])
    listing = "\n".join(
        f'  "{cid}": {json.dumps(text, ensure_ascii=False)}' for cid, text in items)
    ctx_block = f"{context}\n\n" if context else ""
    return f"""\
{ctx_block}The comments below are {LANG_NAMES.get(lang, 'Indic')} YouTube comments (they may be in
{LANG_NAMES.get(lang, 'Indic')} script, in Roman script, code-mixed with English, or
contain emojis).

TARGET CLAIM (English): "Climate change and global warming is a serious concern."
TARGET CLAIM ({LANG_NAMES.get(lang, 'local')}): "{claim_local}"

{ANNOTATION_GUIDELINES}

For EACH comment, in this order:
  1. pick the single argument node id from the menu that best fits the comment
     (use "{UNKNOWN_NODE}" only if genuinely none fits),
  2. give a stance that is consistent with that node,
  3. give a one-clause rationale in English (max 15 words),
  4. give your confidence in [0,1].

Comments:
{{
{listing}
}}

Return exactly:
{{"labels": {{"<comment id>": {{"node": "<node id>", "stance": "Favour|Against|None",
"rationale": "<...>", "confidence": 0.0}}, ...}}}}

Every comment id above must appear exactly once. Do not merge or skip ids.
"""


# ---------------------------------------------------------------------------
def _parse(res, items) -> dict:
    """-> {id: {stance, node, rationale, confidence}} for whatever parsed cleanly."""
    out = {}
    table = (res or {}).get("labels") if isinstance(res, dict) else None
    if not isinstance(table, dict):
        return out
    valid_nodes = set(NODE_IDS) | {UNKNOWN_NODE}
    for cid, _ in items:
        v = table.get(str(cid), table.get(cid))
        if isinstance(v, str):                     # some models return "Against"
            v = {"stance": v}
        if not isinstance(v, dict):
            continue
        stance = normalize_label(v.get("stance", ""), default=None) \
            if v.get("stance") is not None else None
        node = str(v.get("node") or v.get("node_id") or UNKNOWN_NODE).strip()
        if node not in valid_nodes:
            node = UNKNOWN_NODE
        if stance is None:
            # fall back to the taxonomy pivot: the argument implies the stance
            stance = stance_of_node(node, default=None) if node != UNKNOWN_NODE else None
        if stance is None:
            continue
        try:
            conf = float(v.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        out[str(cid)] = {
            "stance": stance,
            "node": node,
            "rationale": str(v.get("rationale", ""))[:200],
            "confidence": min(max(conf, 0.0), 1.0),
        }
    return out


def run_member(member_id: str, provider: str, model, df: pd.DataFrame, lang: str,
               batch: int, workers: int, temperature: float,
               raw_sink: Path | None = None,
               context: str = DEFAULT_CONTEXT) -> dict:
    from llm import get_client
    cli = get_client(provider, model)
    items = list(zip(df["id"].astype(str), df["text"].astype(str)))
    chunks = [items[i:i + batch] for i in range(0, len(items), batch)]
    prompts = [build_prompt(c, lang, context) for c in chunks]
    results = cli.chat_many(prompts, system=_SYSTEM, temperature=temperature,
                            max_tokens=min(4096, 160 * batch + 300),
                            workers=workers, as_json=True,
                            desc=f"{member_id}/{lang}")

    votes: dict = {}
    for chunk, res in zip(chunks, results):
        votes.update(_parse(res, chunk))
        if raw_sink is not None and res is not None:
            with open(raw_sink, "a", encoding="utf-8") as f:
                f.write(json.dumps({"member": member_id, "lang": lang, "raw": res},
                                   ensure_ascii=False) + "\n")

    missing = [(cid, t) for cid, t in items if cid not in votes]
    if missing:
        print(f"  {member_id}: retrying {len(missing)} unparsed comments singly")
        singles = cli.chat_many([build_prompt([m], lang, context) for m in missing],
                                system=_SYSTEM, temperature=temperature,
                                max_tokens=400, workers=workers, as_json=True,
                                desc=f"{member_id}/retry", cache_salt="single")
        for m, res in zip(missing, singles):
            votes.update(_parse(res, [m]))

    got = len(votes)
    print(f"  {member_id}: {got}/{len(items)} labelled "
          f"({Counter(v['stance'] for v in votes.values())}) | {cli.stats()}")
    return votes


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------
def fleiss_kappa(vote_rows: list[list[str]]) -> float:
    """Fleiss' kappa over rows of categorical votes (variable raters tolerated)."""
    rows = [r for r in vote_rows if len(r) >= 2]
    if not rows:
        return float("nan")
    k = len(LABELS)
    p_bar, n_sum = 0.0, Counter()
    for r in rows:
        n = len(r)
        c = Counter(r)
        p_bar += (sum(v * v for v in c.values()) - n) / (n * (n - 1))
        for lab, v in c.items():
            n_sum[lab] += v / n
    p_bar /= len(rows)
    p_e = sum((n_sum[lab] / len(rows)) ** 2 for lab in LABELS)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def consensus(df: pd.DataFrame, votes_by_member: dict, weights: dict | None = None,
              use_confidence: bool = True) -> pd.DataFrame:
    """Add per-member vote columns plus weighted soft labels and agreement stats."""
    members = list(votes_by_member)
    weights = weights or {m: 1.0 for m in members}
    ids = df["id"].astype(str).tolist()

    for m in members:
        df[f"vote_{m}"] = [votes_by_member[m].get(i, {}).get("stance") for i in ids]
        df[f"node_{m}"] = [votes_by_member[m].get(i, {}).get("node") for i in ids]
        df[f"conf_{m}"] = [votes_by_member[m].get(i, {}).get("confidence")
                           for i in ids]

    soft = {lab: [] for lab in LABELS}
    hard, agree_frac, n_voters, entropy, top_node = [], [], [], [], []
    for i in ids:
        acc = {lab: 0.0 for lab in LABELS}
        cast, nodes = [], []
        for m in members:
            v = votes_by_member[m].get(i)
            if not v:
                continue
            w = weights.get(m, 1.0) * (v["confidence"] if use_confidence else 1.0)
            acc[v["stance"]] += max(w, 1e-6)
            cast.append(v["stance"])
            if v["node"] != UNKNOWN_NODE:
                nodes.append(v["node"])
        total = sum(acc.values()) or 1.0
        probs = {lab: acc[lab] / total for lab in LABELS}
        for lab in LABELS:
            soft[lab].append(round(probs[lab], 5))
        best = max(LABELS, key=lambda l: probs[l])
        hard.append(best if cast else None)
        n_voters.append(len(cast))
        agree_frac.append(
            round(Counter(cast).most_common(1)[0][1] / len(cast), 4) if cast else 0.0)
        entropy.append(round(
            -sum(p * math.log(p + 1e-12) for p in probs.values()) / math.log(len(LABELS)),
            4))
        top_node.append(Counter(nodes).most_common(1)[0][0] if nodes else UNKNOWN_NODE)

    for lab in LABELS:
        df[f"p_{lab.lower()}"] = soft[lab]
    df["committee_label"] = hard
    df["n_voters"] = n_voters
    df["agreement"] = agree_frac        # 1.0 = unanimous
    df["entropy"] = entropy             # 0 = certain, 1 = maximally split
    df["node_id"] = top_node
    return df


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run the SETU LLM committee")
    ap.add_argument("--test", nargs="+", required=True,
                    help="normalised test csv(s) from normalize.py, "
                         "e.g. artifacts/test_hi.csv artifacts/test_bn.csv")
    ap.add_argument("--lang", nargs="+", default=None,
                    help="language per --test file (default: parsed from the filename)")
    ap.add_argument("--members", nargs="+", default=None,
                    help="subset of llm.MEMBERS ids, e.g. gemini deepseek mistral")
    ap.add_argument("--text-col", default="text",
                    help="'text_native' to feed transliterated text instead")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--context", default=DEFAULT_CONTEXT,
                    help="provenance context prepended to every prompt; pass '' to "
                         "run context-free (the ablation row)")
    ap.add_argument("--outdir", default=str(ARTIFACTS_DIR))
    args = ap.parse_args()

    from llm import live_members
    members = live_members(args.members)
    if not members:
        raise SystemExit("no committee members available — check .env keys with "
                         "`python3.12 src/llm.py`")
    print(f"committee: {[m[0] for m in members]}")

    outdir = Path(args.outdir)
    summary = {}
    for k, path in enumerate(args.test):
        p = Path(path)
        lang = (args.lang[k] if args.lang and k < len(args.lang)
                else ("bn" if "bn" in p.stem.lower() else "hi"))
        df = read_table(p)
        if args.text_col not in df.columns:
            raise SystemExit(f"{p}: no column {args.text_col!r} "
                             f"(has {list(df.columns)}) — run normalize.py first")
        df = df.copy()
        df["text"] = df[args.text_col].astype(str)
        print(f"\n=== {p.name}  lang={lang}  rows={len(df)} ===")

        raw_sink = outdir / f"committee_{lang}.raw.jsonl"
        raw_sink.unlink(missing_ok=True)

        votes_by_member = {}
        for mid, prov, model in members:
            votes_by_member[mid] = run_member(
                mid, prov, model, df, lang, args.batch, args.workers,
                args.temperature, raw_sink=raw_sink, context=args.context)

        df = consensus(df, votes_by_member)
        out = outdir / f"committee_{lang}.csv"
        df.to_csv(out, index=False)

        vote_rows = [[v for v in (votes_by_member[m].get(str(i), {}).get("stance")
                                  for m in votes_by_member) if v]
                     for i in df["id"].astype(str)]
        stats = {
            "rows": len(df),
            "members": list(votes_by_member),
            "coverage_per_member": {m: len(v) for m, v in votes_by_member.items()},
            "consensus_distribution": df["committee_label"].value_counts().to_dict(),
            "unanimous": int((df["agreement"] == 1.0).sum()),
            "split": int((df["agreement"] < 0.6).sum()),
            "fleiss_kappa": round(fleiss_kappa(vote_rows), 4),
            "mean_entropy": round(float(df["entropy"].mean()), 4),
            "top_nodes": df["node_id"].value_counts().head(12).to_dict(),
            "pairwise_agreement": {
                f"{a}|{b}": round(float(
                    (df[f"vote_{a}"] == df[f"vote_{b}"]).mean()), 4)
                for i, a in enumerate(votes_by_member)
                for b in list(votes_by_member)[i + 1:]},
        }
        summary[lang] = stats
        print(f"\nwrote {out}")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"  -> {stats['unanimous']}/{len(df)} unanimous; "
              f"{stats['split']} genuinely split (these are the rows worth "
              f"hand-annotating — see annotate_dev.py)")

    write_json(outdir / "committee.stats.json", summary)


if __name__ == "__main__":
    main()
