"""Rent a Vast.ai GPU, train there, pull the results back, destroy the instance.

This machine is CPU-only. Fine-tuning MuRIL and XLM-R on ~15k rows is the one step
that genuinely wants a GPU: roughly 2-4 hours per backbone here versus 10-20 minutes
on a rented RTX 4090.

Money safety, because this bills by the second
----------------------------------------------
  * ``search`` and ``status`` never spend anything. Run them first.
  * ``up`` refuses to rent above ``--max-price`` (default $0.40/hr) and prints the
    projected cost of ``--max-hours`` before asking for ``--yes``.
  * ``up`` writes the instance id to artifacts/vast_instance.json so a later
    ``down`` can always find it, even from a fresh shell.
  * **A forgotten instance bills until the credit is gone.** ``train`` tears the
    instance down automatically when it finishes, including on failure. Always
    finish a session with ``python3.12 src/vast_gpu.py down``.

Typical session
---------------
    python3.12 src/vast_gpu.py search --max-price 0.40
    python3.12 src/vast_gpu.py up --offer <id> --max-hours 4 --yes
    python3.12 src/vast_gpu.py train          # syncs code+data, trains, pulls back
    python3.12 src/vast_gpu.py down           # ALWAYS

``train`` uploads only what training needs — src/, requirements.txt, and the
artifacts the encoder consumes. The .env is uploaded with ONLY the HuggingFace
token, never the other 30-odd keys: the GPU box does not need them, and a rented
machine is not somewhere to leave credentials.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

from common import ARTIFACTS_DIR, BASE_DIR, load_env, write_json

API = "https://console.vast.ai/api/v0"
STATE = ARTIFACTS_DIR / "vast_instance.json"
# CUDA + torch preinstalled; matches the transformers stack in requirements.txt
IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"


def _key() -> str:
    k = load_env().get("VAST_AI_API_KEY")
    if not k:
        raise SystemExit("VAST_AI_API_KEY not found in Codes/.env")
    return k


def _req(method: str, path: str, **kw):
    r = requests.request(method, f"{API}{path}",
                         headers={"Authorization": f"Bearer {_key()}"},
                         timeout=60, **kw)
    r.raise_for_status()
    return r.json() if r.text.strip() else {}


def _offers(max_price: float, min_vram: int, disk: int, limit: int = 20) -> list:
    """Search rentable offers.

    Verified 2026-08-02: this is ``GET /bundles/`` with the filter JSON in a ``q``
    query parameter. The older ``PUT /bundles/`` with a JSON body now 404s.
    """
    q = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": min_vram * 1024},
        "dph_total": {"lte": max_price},
        "disk_space": {"gte": disk},
        "inet_down": {"gte": 100},
        "order": [["dph_total", "asc"]],
        "type": "on-demand",
        "limit": limit,
    }
    return _req("GET", "/bundles/", params={"q": json.dumps(q)}).get("offers", [])


# ---------------------------------------------------------------------------
def cmd_balance(args):
    u = _req("GET", "/users/current/")
    credit = float(u.get("credit") or 0)
    print(f"  account : {u.get('username')}")
    print(f"  credit  : ${credit:.2f}")
    print(f"  at $0.30/hr that is ~{credit / 0.30:.1f} GPU-hours")
    if credit < 2:
        print("  ⚠ under $2 — top up at https://cloud.vast.ai/billing/ before renting")
    return credit


def cmd_search(args):
    offers = _offers(args.max_price, args.min_vram, args.disk)
    if not offers:
        print(f"no offers under ${args.max_price}/hr with >={args.min_vram}GB VRAM. "
              f"Raise --max-price or lower --min-vram.")
        return []
    print(f"{'offer_id':>10}  {'$/hr':>6}  {'GPU':<22} {'VRAM':>6} {'net↓':>7}  loc")
    for o in offers[:12]:
        print(f"{o['id']:>10}  {o['dph_total']:>6.3f}  {o['gpu_name'][:22]:<22} "
              f"{o['gpu_ram'] / 1024:>5.0f}G {o.get('inet_down', 0):>6.0f}M  "
              f"{o.get('geolocation', '?')}")
    cheapest = offers[0]
    print(f"\ncheapest: offer {cheapest['id']} at ${cheapest['dph_total']:.3f}/hr "
          f"→ {args.hours}h would cost ~${cheapest['dph_total'] * args.hours:.2f}")
    return offers


def cmd_up(args):
    credit = cmd_balance(args)
    offers = _offers(args.max_price, args.min_vram, args.disk)
    if not offers:
        raise SystemExit("no offers matched — run `search` and adjust the limits")

    offer = next((o for o in offers if o["id"] == args.offer), offers[0]) \
        if args.offer else offers[0]
    price, projected = offer["dph_total"], offer["dph_total"] * args.max_hours

    print(f"\nabout to rent:")
    print(f"  offer     {offer['id']}  {offer['gpu_name']}  "
          f"{offer['gpu_ram'] / 1024:.0f}GB")
    print(f"  price     ${price:.3f}/hr")
    print(f"  budget    {args.max_hours}h → ~${projected:.2f} of your ${credit:.2f}")
    if price > args.max_price:
        raise SystemExit(f"price ${price:.3f} exceeds --max-price ${args.max_price}")
    if projected > credit:
        raise SystemExit(f"projected ${projected:.2f} exceeds your ${credit:.2f} "
                         f"credit — lower --max-hours or top up")
    if not args.yes:
        raise SystemExit("add --yes to actually rent (nothing spent so far)")

    res = _req("PUT", f"/asks/{offer['id']}/", json={
        "client_id": "me", "image": IMAGE, "disk": args.disk, "runtype": "ssh",
        "label": "nirnay-aisome2026",
        "onstart": "touch /root/.vast_ready",
    })
    iid = res.get("new_contract")
    if not iid:
        raise SystemExit(f"rental failed: {res}")
    write_json(STATE, {"instance_id": iid, "offer": offer["id"],
                       "price_per_hour": price, "started": time.time(),
                       "gpu": offer["gpu_name"]})
    print(f"\ninstance {iid} requested — ${price:.3f}/hr is now running")
    print(f"  wait for it:  python3.12 src/vast_gpu.py status --wait")
    print(f"  ⚠ REMEMBER:   python3.12 src/vast_gpu.py down")
    return iid


def _instance(iid=None):
    if iid is None:
        if not STATE.exists():
            raise SystemExit("no artifacts/vast_instance.json — nothing to act on. "
                             "Check https://cloud.vast.ai/instances/ manually.")
        iid = json.loads(STATE.read_text())["instance_id"]
    for inst in _req("GET", "/instances/").get("instances", []):
        if inst["id"] == iid:
            return inst
    return None


def cmd_status(args):
    deadline = time.time() + (args.wait_timeout if args.wait else 0)
    while True:
        inst = _instance(args.instance)
        if inst is None:
            print("instance not found (already destroyed?)")
            return None
        st, ssh_h, ssh_p = (inst.get("actual_status"), inst.get("ssh_host"),
                            inst.get("ssh_port"))
        elapsed = (time.time() - json.loads(STATE.read_text())["started"]) / 3600 \
            if STATE.exists() else 0
        cost = elapsed * float(inst.get("dph_total") or 0)
        print(f"  id {inst['id']} | {st} | {inst.get('gpu_name')} | "
              f"${inst.get('dph_total', 0):.3f}/hr | up {elapsed:.2f}h "
              f"≈ ${cost:.2f} spent")
        if st == "running" and ssh_h:
            print(f"  ssh -p {ssh_p} root@{ssh_h}")
            return inst
        if not args.wait or time.time() > deadline:
            return inst
        time.sleep(15)


def cmd_down(args):
    inst = _instance(args.instance)
    if inst is None:
        print("nothing running")
        STATE.unlink(missing_ok=True)
        return
    _req("DELETE", f"/instances/{inst['id']}/", json={})
    if STATE.exists():
        s = json.loads(STATE.read_text())
        hrs = (time.time() - s["started"]) / 3600
        print(f"destroyed {inst['id']} after {hrs:.2f}h "
              f"≈ ${hrs * s['price_per_hour']:.2f}")
        STATE.unlink()
    else:
        print(f"destroyed {inst['id']}")


# ---------------------------------------------------------------------------
def _ssh_base(inst):
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-p", str(inst["ssh_port"]),
            f"root@{inst['ssh_host']}"]


def _run_remote(inst, cmd, check=True):
    full = _ssh_base(inst) + [cmd]
    print(f"  remote$ {cmd[:110]}")
    return subprocess.run(full, check=check)


def cmd_train(args):
    inst = _instance(args.instance)
    if inst is None or inst.get("actual_status") != "running":
        raise SystemExit("no running instance — `up` then `status --wait` first")
    host, port = inst["ssh_host"], inst["ssh_port"]
    remote = "/root/nirnay"

    # A rented box gets exactly one credential: the HF token, for pushing weights.
    # None of the other keys in .env are needed there.
    env = load_env()
    slim = ARTIFACTS_DIR / ".env.gpu"
    slim.write_text(f"HUGGINGFACE_TOKEN={env.get('HUGGINGFACE_TOKEN','')}\n"
                    f"RANDOM_SEED={env.get('RANDOM_SEED','42')}\n", encoding="utf-8")
    slim.chmod(0o600)

    print("uploading code and training data ...")
    _run_remote(inst, f"mkdir -p {remote}/src {remote}/artifacts")
    scp = ["scp", "-o", "StrictHostKeyChecking=no", "-o",
           "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-P", str(port)]
    subprocess.run(scp + [str(p) for p in (BASE_DIR / "src").glob("*.py")]
                   + [f"root@{host}:{remote}/src/"], check=True)
    subprocess.run(scp + [str(BASE_DIR / "requirements.txt"),
                          f"root@{host}:{remote}/"], check=True)
    subprocess.run(scp + [str(slim), f"root@{host}:{remote}/.env"], check=True)
    payload = [p for p in ARTIFACTS_DIR.glob("*.csv")
               if p.name.startswith(("train_en", "synth_train", "committee_",
                                     "test_", "dev_gold", "seed_gold"))]
    if not payload:
        raise SystemExit("no training artifacts to upload — run stage0/stage1 first")
    print(f"  {len(payload)} data files ({sum(p.stat().st_size for p in payload)//1024} KB)")
    subprocess.run(scp + [str(p) for p in payload]
                   + [f"root@{host}:{remote}/artifacts/"], check=True)

    _run_remote(inst, f"cd {remote} && pip install -q -r requirements.txt 2>&1 | tail -3")
    _run_remote(inst, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

    pool = " ".join(f"artifacts/{n}" for n in
                    ("train_en.csv", "train_en.to-hi.csv::text_hi",
                     "train_en.to-bn.csv::text_bn", "synth_train.csv")
                    if (ARTIFACTS_DIR / n.split("::")[0]).exists())
    for backbone, out in ((args.backbone, "model_setu"),
                          (args.backbone2, "model_xlmr")):
        if not backbone:
            continue
        print(f"\n--- training {backbone} on the GPU ---")
        _run_remote(inst, (
            f"cd {remote} && python src/train_transformer.py --train {pool} "
            f"--model {backbone} --out artifacts/{out} --epochs {args.epochs} "
            f"--batch {args.batch} --aux-weight 0.3 --soft-alpha 0.4 "
            f"2>&1 | tail -25"), check=False)

    print("\npulling checkpoints back ...")
    for out in ("model_setu", "model_xlmr"):
        subprocess.run(scp + ["-r", f"root@{host}:{remote}/artifacts/{out}",
                              str(ARTIFACTS_DIR) + "/"], check=False)
    slim.unlink(missing_ok=True)
    print("done")
    if not args.keep_up:
        print("tearing the instance down (pass --keep-up to leave it running)")
        cmd_down(args)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Vast.ai GPU helper for Nirnay/SETU")
    ap.add_argument("--instance", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("balance", help="show credit — spends nothing")
    b.set_defaults(func=cmd_balance)

    s = sub.add_parser("search", help="list offers — spends nothing")
    s.add_argument("--max-price", type=float, default=0.40, help="$/hr ceiling")
    s.add_argument("--min-vram", type=int, default=16, help="GB")
    s.add_argument("--disk", type=int, default=40, help="GB")
    s.add_argument("--hours", type=float, default=4, help="for the cost estimate")
    s.set_defaults(func=cmd_search)

    u = sub.add_parser("up", help="RENT a GPU — starts billing")
    u.add_argument("--offer", type=int, default=None)
    u.add_argument("--max-price", type=float, default=0.40)
    u.add_argument("--min-vram", type=int, default=16)
    u.add_argument("--disk", type=int, default=40)
    u.add_argument("--max-hours", type=float, default=4,
                   help="budget check only; does not auto-stop")
    u.add_argument("--yes", action="store_true", help="required to actually rent")
    u.set_defaults(func=cmd_up)

    st = sub.add_parser("status", help="state and money spent so far")
    st.add_argument("--wait", action="store_true", help="poll until running")
    st.add_argument("--wait-timeout", type=float, default=600)
    st.set_defaults(func=cmd_status)

    t = sub.add_parser("train", help="sync, fine-tune both backbones, pull back")
    t.add_argument("--backbone", default="google/muril-base-cased")
    t.add_argument("--backbone2", default="xlm-roberta-base")
    t.add_argument("--epochs", type=float, default=3)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--keep-up", action="store_true",
                   help="do NOT destroy the instance afterwards")
    t.set_defaults(func=cmd_train)

    d = sub.add_parser("down", help="destroy the instance and stop billing")
    d.set_defaults(func=cmd_down)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
