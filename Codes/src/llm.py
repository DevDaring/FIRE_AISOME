"""Unified multi-provider LLM client: key rotation, disk cache, JSON coercion.

No Google. Every provider here is DeepSeek's own API or a PAID model on
OpenRouter, by project decision — no GCP service and no Gemini is called
anywhere in this pipeline.

The .env holds two keys each for DeepSeek and OpenRouter; rotating across them
doubles the usable rate limit, which is what makes a 5-model committee over
1000 comments plus ~1k synthetic generations comfortable.

Everything is cached to artifacts/cache/llm_<provider>.jsonl keyed by
sha1(model | prompt | system | temperature | seed), so an interrupted run resumes
for free and repeated experiments cost nothing.

Providers are exposed as *committee members* — a stable list of (name, provider,
model) triples used by llm_committee.py so that the working notes can name them.

Usage
-----
    from llm import get_client, MEMBERS
    cli = get_client("deepseek")
    print(cli.chat("Say hi", system="Be terse."))
    print(cli.chat_json('Return {"a":1}'))
    outs = cli.chat_many([...prompts...], workers=8)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from common import CACHE_DIR, env_keys, load_env

DEFAULT_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Disk cache (append-only JSONL per provider)
# ---------------------------------------------------------------------------
class _Cache:
    _locks: dict[str, threading.Lock] = {}

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"llm_{name}.jsonl"
        self.lock = _Cache._locks.setdefault(name, threading.Lock())
        self.mem: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8",
                                            errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                    self.mem[rec["k"]] = rec["v"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, k: str):
        return self.mem.get(k)

    def put(self, k: str, v: str):
        with self.lock:
            self.mem[k] = v
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


def _key(*parts) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


# NOTE the dots in the character classes. Google now issues keys shaped like
# "AQ.Ab8RN6Kxil..." as well as the classic "AIza..." form; a class without `.`
# stops matching at the dot and leaks the rest of the key into the log. Keep this
# permissive — over-redacting a log line costs nothing, under-redacting burns a key.
_SECRET_RE = re.compile(
    r"(key=)[A-Za-z0-9_.\-]{16,}"                      # ?key=... in any URL
    r"|AIza[0-9A-Za-z_.\-]{16,}"                       # Google, classic
    r"|AQ\.[A-Za-z0-9_.\-]{16,}"                       # Google, newer form
    r"|sk-(?:or-v1-|proj-|ant-)?[A-Za-z0-9_.\-]{16,}"  # OpenAI/OpenRouter/DeepSeek
    r"|hf_[A-Za-z0-9]{16,}"                            # HuggingFace
    r"|gh[pousr]_[A-Za-z0-9]{16,}"                     # GitHub
    r"|github_pat_[A-Za-z0-9_]{16,}")


def _redact(text: str) -> str:
    """Strip anything key-shaped out of a string before it is printed."""
    return _SECRET_RE.sub(lambda m: (m.group(1) or "") + "<redacted>", str(text))


# ---------------------------------------------------------------------------
# JSON coercion — LLMs wrap JSON in prose and code fences no matter what you ask
# ---------------------------------------------------------------------------
def extract_json(text: str):
    """Best-effort extraction of the first JSON object or array in `text`."""
    if text is None:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # scan for the first balanced {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(t)):
                c = t[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = t.find(opener, start + 1)
    return None


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------
class LLMClient:
    """One provider, N rotating API keys, cached + retried chat completion."""

    name = "base"

    def __init__(self, keys: list[str], model: str, cache_name: str | None = None):
        if not keys:
            raise SystemExit(f"{self.name}: no API keys found in .env")
        self.keys = keys
        self.model = model
        self._dead: set[str] = set()
        self._i = 0
        self._klock = threading.Lock()
        self.cache = _Cache(cache_name or self.name)
        self.session = requests.Session()
        self.n_calls = 0
        self.n_cached = 0
        self.n_fail = 0

    # -- key rotation --------------------------------------------------------
    def _next_key(self) -> str:
        with self._klock:
            live = [k for k in self.keys if k not in self._dead]
            if not live:                      # every key rejected: surface it
                live = self.keys
            k = live[self._i % len(live)]
            self._i += 1
            return k

    def _mark_dead(self, key: str, why: str):
        """Permanently retire a key that the provider says is invalid.

        A revoked key never recovers, so evict it for the life of the process
        rather than rediscovering it thousands of times. Round-robin over a list
        containing dead keys sends a proportional share of all traffic to certain
        failure, and each one burns the full retry ladder first.
        """
        with self._klock:
            if key not in self._dead:
                self._dead.add(key)
                idx = self.keys.index(key) + 1 if key in self.keys else "?"
                print(f"  [{self.name}] key #{idx} retired for this run: {why}. "
                      f"{len(self.keys) - len(self._dead)}/{len(self.keys)} still live.")

    # -- provider hook -------------------------------------------------------
    def _request(self, prompt: str, system: str | None, temperature: float,
                 max_tokens: int, key: str) -> str:
        raise NotImplementedError

    # -- public API ----------------------------------------------------------
    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.0,
             max_tokens: int = 1024, tries: int = 4, use_cache: bool = True,
             cache_salt: str = "") -> str | None:
        ck = _key(self.name, self.model, system or "", prompt, temperature, cache_salt)
        if use_cache:
            hit = self.cache.get(ck)
            if hit is not None:
                self.n_cached += 1
                return hit
        last = None
        for attempt in range(tries):
            try:
                key_used = self._next_key()
                out = self._request(prompt, system, temperature, max_tokens,
                                    key_used)
                if out:
                    self.n_calls += 1
                    if use_cache:
                        self.cache.put(ck, out)
                    return out
                last = "empty response"
            except Exception as e:  # noqa: BLE001 — retry across keys and back off
                # Some providers echo the request URL (query params included) into
                # their error text. Never let a live key reach stdout or a log.
                last = _redact(f"{type(e).__name__}: {e}")
                low = str(e).lower()
                if "api key not valid" in low or "api_key_invalid" in low:
                    self._mark_dead(key_used, "provider says the key is invalid")
                elif "no longer available" in low or "not found for api version" in low:
                    # a retired model id — no key will fix it, so stop burning retries
                    print(f"  [{self.name}] model {self.model!r} looks retired; "
                          f"re-run `python3.12 src/llm.py` to refresh the roster")
                    break
            if attempt < tries - 1:
                time.sleep(min(2.0 * (2 ** attempt), 30.0))
        self.n_fail += 1
        print(f"  [{self.name}/{self.model}] giving up after {tries} tries: {last}")
        return None

    def chat_json(self, prompt: str, system: str | None = None,
                  temperature: float = 0.0, max_tokens: int = 1024, **kw):
        sys_json = (system or "") + (
            "\n\nRespond with ONLY valid JSON. No prose, no markdown fences.")
        raw = self.chat(prompt, system=sys_json.strip(), temperature=temperature,
                        max_tokens=max_tokens, **kw)
        if not raw:
            return None
        parsed = extract_json(raw)
        if parsed is None:
            # A truncated response is a non-empty string, so `chat` cached it as a
            # success. Every later run then replays the same unparseable text from
            # disk and falls back to slow per-row retries forever — the cache
            # remembers the failure instead of the answer. Evict it so a retry can
            # actually retry.
            self._evict(self.name, self.model, sys_json.strip(), prompt,
                        temperature, kw.get("cache_salt", ""))
        return parsed

    def _evict(self, *key_parts):
        """Drop one poisoned cache entry from memory (the JSONL is append-only)."""
        self.cache.mem.pop(_key(*key_parts), None)

    def chat_many(self, prompts: list[str], system: str | None = None,
                  temperature: float = 0.0, max_tokens: int = 1024,
                  workers: int = 6, as_json: bool = False, desc: str = "",
                  **kw) -> list:
        """Concurrent map over prompts, preserving order. Failures become None."""
        fn = self.chat_json if as_json else self.chat
        out: list = [None] * len(prompts)
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(prompts), desc=desc or f"{self.name}", unit="req")
        except ImportError:
            bar = None

        def work(i):
            out[i] = fn(prompts[i], system=system, temperature=temperature,
                        max_tokens=max_tokens, **kw)
            if bar:
                bar.update(1)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(work, range(len(prompts))))
        if bar:
            bar.close()
        return out

    def stats(self) -> dict:
        return {"provider": self.name, "model": self.model, "keys": len(self.keys),
                "live_calls": self.n_calls, "cache_hits": self.n_cached,
                "failures": self.n_fail}


# ---------------------------------------------------------------------------
# OpenAI-compatible providers (OpenRouter, DeepSeek, NanoGPT, Akash, Mistral)
# ---------------------------------------------------------------------------
class OpenAICompatClient(LLMClient):
    def __init__(self, name, keys, model, base_url, extra_headers=None,
                 cache_name=None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}
        super().__init__(keys, model, cache_name=cache_name or name)

    # Model families that reason before answering. Their hidden reasoning is
    # billed against max_tokens, so a budget sized for the answer alone returns
    # finish_reason="length" with content: null — the model thought until it ran
    # out and never spoke. Detected by name because the API does not advertise it.
    _REASONING = ("gpt-5", "o1-", "o3-", "o4-", "reasoner", "-thinking",
                  "qwq", "deepseek-r1")

    @property
    def is_reasoning(self) -> bool:
        m = self.model.lower()
        return any(t in m for t in self._REASONING)

    def _request(self, prompt, system, temperature, max_tokens, key):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        headers = {"Authorization": f"Bearer {key}",
                   "Content-Type": "application/json", **self.extra_headers}
        body = {"model": self.model, "messages": msgs,
                "temperature": temperature, "max_tokens": max_tokens}
        if self.is_reasoning:
            # Ask for the least reasoning the model will do, and make room for it
            # on top of the answer budget rather than out of it.
            body["reasoning"] = {"effort": "low", "exclude": True}
            body["max_tokens"] = max(max_tokens * 4, 8000)
        r = self.session.post(f"{self.base_url}/chat/completions", headers=headers,
                              json=body, timeout=DEFAULT_TIMEOUT)
        if r.status_code in (429, 500, 502, 503, 504):
            raise RuntimeError(f"HTTP {r.status_code}")
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"no choices: {_redact(str(data))[:200]}")
        msg = choices[0].get("message") or {}
        # Reasoning models (gpt-5*, o*, deepseek-reasoner) legitimately return
        # content: null when the whole token budget went to hidden reasoning.
        # `.get("content", "")` yields None there, not "", so a bare .strip()
        # raises AttributeError and the retry loop burns every key on a bug that
        # more max_tokens would have fixed. Surface it as a real, actionable error.
        content = msg.get("content")
        if content is None:
            reason = choices[0].get("finish_reason")
            raise RuntimeError(
                f"empty content (finish_reason={reason!r}) — if this is a reasoning "
                f"model, raise max_tokens; it spent the budget thinking")
        return content.strip()


def _openrouter(model: str, cache_name: str | None = None) -> OpenAICompatClient:
    """OpenRouter client pinned to one model.

    Refuses ``:free`` variants on purpose. The free pool is shared-quota and
    heavily rate-gated: mid-run it starts returning 429s and empty bodies, which
    silently thins the committee (some comments get 5 votes, others 2) and
    corrupts the agreement statistics that self-training and the dev-set sampler
    both depend on. Paid cheap models cost cents for this workload and behave.
    """
    if model.endswith(":free"):
        raise SystemExit(
            f"refusing OpenRouter free model {model!r} — the shared free quota "
            f"rate-limits mid-run and corrupts committee agreement counts. "
            f"Use the paid variant: {model[:-5]!r}")
    env = load_env()
    return OpenAICompatClient(
        "openrouter", env_keys("OPENROUTER_API_KEY"), model,
        env.get("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1"),
        extra_headers={"HTTP-Referer": "https://aisome2026.my.canva.site/",
                       "X-Title": "AISoMe2026-Nirnay"},
        cache_name=cache_name or f"or_{model.split('/')[-1].replace(':', '_')}")


def _deepseek(model: str) -> OpenAICompatClient:
    return OpenAICompatClient(
        "deepseek", env_keys("DEEPSEEK_API_KEY"), model,
        load_env().get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1"),
        cache_name=f"ds_{model}")


# Model ids below were verified live against the provider catalogues on
# 2026-08-02. They churn — providers rename and retire ids without notice, and a
# stale id fails hours into a run. Re-verify with `python3.12 src/llm.py` before
# any long job; that is what the probe is for.
_FACTORIES = {
    # --- DeepSeek platform (2 keys, round-robin) ---------------------------
    "deepseek":   lambda m=None: _deepseek(m or "deepseek-chat"),
    "deepseek-r": lambda m=None: _deepseek(m or "deepseek-reasoner"),

    # --- OpenRouter, PAID cheap models only (2 keys, round-robin) ----------
    "or-gpt":     lambda m=None: _openrouter(m or "openai/gpt-5-nano"),
    "or-gpt-mini": lambda m=None: _openrouter(m or "openai/gpt-5-mini"),
    "or-llama":   lambda m=None: _openrouter(m or "meta-llama/llama-4-maverick"),
    "or-qwen":    lambda m=None: _openrouter(m or "qwen/qwen3.7-flash"),
    "or-qwen-plus": lambda m=None: _openrouter(m or "qwen/qwen3.7-plus"),
    "or-mistral": lambda m=None: _openrouter(m or "mistralai/mistral-nemo"),
    "or-deepseek": lambda m=None: _openrouter(m or "deepseek/deepseek-v4-flash"),
    "or-claude":  lambda m=None: _openrouter(m or "anthropic/claude-haiku-4.5"),
    "or":         lambda m=None: _openrouter(
        m or load_env().get("OPENROUTER_PRIMARY_MODEL_NAME", "openai/gpt-5-nano")),

    # --- direct provider APIs ----------------------------------------------
    "mistral":   lambda m=None: OpenAICompatClient(
        "mistral", env_keys("MISTRAL_API_KEY"),
        m or load_env().get("MISTRAL_MODEL_NAME", "mistral-small-latest"),
        "https://api.mistral.ai/v1"),
    "nanogpt":   lambda m=None: OpenAICompatClient(
        "nanogpt", env_keys("Nano_GPT_API_KEY", "NanoGPT_API_Key"),
        m or "gpt-4o-mini",
        load_env().get("Nano_GPT_Base_URL", "https://nano-gpt.com/api/v1")),
}

_CLIENTS: dict[str, LLMClient] = {}
_CLIENT_LOCK = threading.Lock()


def get_client(provider: str, model: str | None = None) -> LLMClient:
    """Cached client factory. `provider` is a key of _FACTORIES."""
    ck = f"{provider}:{model or ''}"
    with _CLIENT_LOCK:
        if ck not in _CLIENTS:
            if provider not in _FACTORIES:
                raise SystemExit(f"unknown provider {provider!r}; "
                                 f"choose from {sorted(_FACTORIES)}")
            _CLIENTS[ck] = _FACTORIES[provider](model)
        return _CLIENTS[ck]


def available_providers() -> list[str]:
    """Providers that actually have at least one key present in .env."""
    ok = []
    for p in _FACTORIES:
        try:
            get_client(p)
            ok.append(p)
        except SystemExit:
            continue
    return ok


# ---------------------------------------------------------------------------
# TWO SEPARATE TIERS. Keeping them disjoint is a methodological requirement,
# not a convenience.
#
#   TEACHER  — labels all 1000 test comments. Its labels are distilled into the
#              fine-tuned encoder that we actually submit. Cheap and diverse:
#              heterogeneity is what makes the agreement signal informative,
#              since five samples from one model agree with themselves for the
#              wrong reasons.
#
#   JUDGE    — builds the silver dev set used to calibrate and to choose between
#              runs. It MUST NOT share models with the TEACHER tier. If it did,
#              we would be scoring the teacher against itself, the committee
#              would look perfect, and every calibration constant would be fitted
#              to the teacher's own biases. Stronger + disjoint + reasoning-capable.
#
# All OpenRouter entries are PAID models (never ``:free`` — see _openrouter).
# ---------------------------------------------------------------------------
# Five distinct labs — Meta, Alibaba, OpenAI, Mistral, DeepSeek — so a shared
# blind spot in any one pretraining corpus cannot carry a majority vote.
TEACHER_MEMBERS = [
    # (member id, provider, model override or None)
    ("or-llama",    "or-llama",    None),   # meta-llama/llama-4-maverick
    ("or-qwen",     "or-qwen",     None),   # qwen3.7-flash — cheap, 1M ctx, strong Indic
    ("or-gpt",      "or-gpt",      None),   # openai/gpt-5-nano
    ("or-mistral",  "or-mistral",  None),   # mistralai/mistral-nemo
    ("or-deepseek", "or-deepseek", None),   # deepseek/deepseek-v4-flash
]

# Stronger tier, and no model id here appears above — assert_disjoint() enforces
# it. Anthropic is absent from TEACHER entirely. Two families recur across tiers
# but at very different capability points: OpenAI (gpt-5-nano teaches, gpt-5-mini
# judges) and Qwen (3.7-flash teaches, 3.7-plus judges). State that plainly in the
# working notes rather than claiming full independence.
JUDGE_MEMBERS = [
    ("judge-claude", "or-claude",   None),  # anthropic/claude-haiku-4.5
    ("judge-r1",     "deepseek-r",  None),  # deepseek-reasoner, thinks before answering
    ("judge-gpt",    "or-gpt-mini", None),  # openai/gpt-5-mini
    ("judge-qwen",   "or-qwen-plus", None), # qwen3.7-plus, strongest cheap Indic model
]

MEMBERS = TEACHER_MEMBERS          # backwards-compatible alias

_TIERS = {"teacher": TEACHER_MEMBERS, "judge": JUDGE_MEMBERS}


def live_members(requested: list[str] | None = None,
                 tier: str = "teacher") -> list[tuple]:
    """Roster entries whose provider actually has usable keys."""
    roster = _TIERS.get(tier, TEACHER_MEMBERS)
    want = set(requested) if requested else None
    out = []
    for mid, prov, model in roster:
        if want and mid not in want:
            continue
        try:
            get_client(prov, model)
            out.append((mid, prov, model))
        except SystemExit as e:
            print(f"  skipping {tier} member {mid!r}: {e}")
    return out


def assert_disjoint():
    """Fail loudly if a model appears in both tiers — see the note above."""
    def resolved(roster):
        s = set()
        for mid, prov, model in roster:
            try:
                c = get_client(prov, model)
                s.add(f"{c.name}:{c.model}")
            except SystemExit:
                pass
        return s
    overlap = resolved(TEACHER_MEMBERS) & resolved(JUDGE_MEMBERS)
    if overlap:
        raise SystemExit(
            f"TEACHER and JUDGE tiers share {sorted(overlap)}. The silver dev set "
            f"would be scoring the teacher against itself. Edit the rosters in llm.py.")
    return True


def probe(tier: str | None = None, prompt: str = "Reply with exactly: OK"):
    """Live check of every rostered model. Run this before a long job.

    Model ids on OpenRouter get renamed and retired, so a stale roster fails
    hours into a run. This tells you in 30 seconds which ones answer.
    """
    tiers = [tier] if tier else ["teacher", "judge"]
    ok, bad = [], []
    for t in tiers:
        print(f"\n=== {t.upper()} TIER ===")
        for mid, prov, model in _TIERS[t]:
            try:
                cli = get_client(prov, model)
            except SystemExit as e:
                print(f"  ✗ {mid:14} {str(e)[:70]}")
                bad.append((t, mid))
                continue
            # generous budget: reasoning models spend most of it before emitting
            ans = cli.chat(prompt, max_tokens=2000, tries=2, use_cache=False)
            mark = "✓" if ans else "✗"
            print(f"  {mark} {mid:14} {cli.model:44} {(ans or 'NO RESPONSE')[:24]!r}")
            (ok if ans else bad).append((t, mid))
    print(f"\nworking: {len(ok)}   failing: {len(bad)}")
    if bad:
        print("fix or drop the failing entries in llm.py before a long run:")
        for t, mid in bad:
            print(f"   {t}/{mid}")
    return ok, bad


if __name__ == "__main__":
    import sys
    print("providers with keys:", available_providers())
    for label, prefixes in (("deepseek", ("DEEPSEEK_API_KEY",)),
                            ("openrouter", ("OPENROUTER_API_KEY",))):
        print(f"  {label:11} {len(env_keys(*prefixes))} key(s) in round-robin")
    probe(sys.argv[1] if len(sys.argv) > 1 else None)
    try:
        assert_disjoint()
        print("\ntiers are disjoint ✓")
    except SystemExit as e:
        print(f"\n{e}")


def purge_bad_cache(pattern: str = "*") -> int:
    """Rewrite cache files, dropping entries whose body will not parse as JSON.

    ``chat_json`` evicts poisoned entries from memory, but the JSONL on disk is
    append-only, so a fresh process reloads them and replays the same failure.
    Run this once after fixing whatever produced the truncated output.
    """
    removed = 0
    for path in sorted(CACHE_DIR.glob(f"llm_{pattern}.jsonl")):
        keep, dropped = [], 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = rec.get("v", "")
            # only judge entries that were clearly meant to be JSON
            if v.lstrip().startswith(("{", "[", "```")) and extract_json(v) is None:
                dropped += 1
                continue
            keep.append(line)
        if dropped:
            path.write_text("\n".join(keep) + "\n", encoding="utf-8")
            print(f"  {path.name}: dropped {dropped} unparseable entries, "
                  f"kept {len(keep)}")
            removed += dropped
    print(f"purged {removed} poisoned cache entries")
    return removed
