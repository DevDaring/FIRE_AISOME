#!/usr/bin/env bash
# Remote GPU pipeline: dry run in two stages, then the three large-model trainings.
#
# Runs ENTIRELY on the rented box, detached, writing to gpu_pipeline.log. This
# machine's SSH to the box drops mid-command every few minutes, so nothing here
# may depend on a live connection: we launch it once with nohup and afterwards
# only poll a log file. Each stage writes a .done marker so a relaunch resumes.
#
# Why LARGE backbones: every model in the submission so far is base-size
# (110-270M). Large variants are the one reliable untapped lever, typically
# +2-5 macro-F1, and are CPU-infeasible but ~20 min each on a 3090.
#
# Why a DeBERTa-v3-large run on the English view: test-set forensics established
# the comments are machine-translated English, and we already hold
# back-translations. A strong English-only encoder on that view exploits the
# finding a second time and is decorrelated from every native-script model.
set -uo pipefail
cd /root/nirnay
mkdir -p state

# SINGLE-INSTANCE GUARD. Launching this over flaky SSH produced six concurrent
# copies last time: every "failed" ssh actually took, and all six then fought
# over the same HuggingFace cache lock and deadlocked mid-download with the GPU
# at 0% for 50 minutes. flock makes a second copy exit immediately instead.
exec 9>state/pipeline.lock
if ! flock -n 9; then
  echo "$(date -u '+%F %T UTC')  another pipeline holds the lock — exiting" >> gpu_pipeline.log
  exit 0
fi
L=gpu_pipeline.log
say() { echo "$(date -u '+%F %T UTC')  $*" | tee -a $L; }
have() { [ -f "state/$1.done" ]; }
mark() { touch "state/$1.done"; }

POOL_NATIVE="artifacts/train_en.csv artifacts/train_en.to-hi.csv::text_hi \
artifacts/train_en.to-bn.csv::text_bn artifacts/synth_train.csv \
artifacts/distil_hi.csv artifacts/distil_bn.csv"
POOL_EN="artifacts/train_en.csv artifacts/synth_train.csv artifacts/distil_en.csv"

say "=== GPU pipeline start (pid $$) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /' | tee -a $L

# ---------------------------------------------------------------------------
# BANDWIDTH PRE-FLIGHT. The previous box advertised 605 Mb/s but took 10 s for a
# single HEAD to huggingface.co and 50 min for 1.5 GB. Download the three large
# checkpoints ONCE, sequentially, in a single process before anything else, and
# fail loudly if the box is too slow to be worth renting.
# ---------------------------------------------------------------------------
if ! have download; then
  say "[pre-flight] downloading large checkpoints sequentially"
  T0=$(date +%s)
  python - >>$L 2>&1 <<'PY'
import time
from transformers import AutoModel, AutoTokenizer
for m in ('google/muril-large-cased','microsoft/deberta-v3-large','xlm-roberta-large'):
    t=time.time()
    try:
        AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
        print(f'  fetched {m:32} in {time.time()-t:5.0f}s')
    except Exception as e:
        print(f'  FAILED  {m:32} {type(e).__name__}: {str(e)[:90]}')
PY
  ELAPSED=$(( $(date +%s) - T0 ))
  say "[pre-flight] all downloads took ${ELAPSED}s"
  if [ "$ELAPSED" -gt 1500 ]; then
    say "[pre-flight] ABORT: >25 min just to download. This box is too slow to be"
    say "[pre-flight] worth renting — destroy it and pick one with higher inet_down."
    exit 2
  fi
  mark download
fi

# ---------------------------------------------------------------------------
# DRY RUN STAGE A — 2 samples through each backbone. Catches a wrong/gated model
# id, an OOM at large hidden size, or a missing tokenizer dependency in seconds.
# ---------------------------------------------------------------------------
if ! have dryA; then
  say "[dry A] 2-sample forward pass per backbone"
  python - >>$L 2>&1 <<'PY'
import sys; sys.path.insert(0,'src')
import torch
from train_transformer import build_model
from transformers import AutoTokenizer
T=['बिल नाई एक वैज्ञानिक नहीं हैं। वह एक यांत्रिक अभियंता हैं।',
   'Bill Nye is not a scientist. He is a mechanical engineer.']
ok=[]
for m in ('google/muril-large-cased','microsoft/deberta-v3-large','xlm-roberta-large'):
    try:
        tok=AutoTokenizer.from_pretrained(m)
        net=build_model(m,0.3,None,0.4,0.05,attn='auto').cuda().eval()
        enc=tok(T,return_tensors='pt',padding=True,truncation=True,max_length=128).to('cuda')
        with torch.no_grad(): o=net(**enc)
        n=sum(x.numel() for x in net.parameters())/1e6
        print(f'  OK   {m:30} {n:6.0f}M  logits{tuple(o["logits"].shape)}')
        ok.append(m); del net; torch.cuda.empty_cache()
    except Exception as e:
        print(f'  FAIL {m:30} {type(e).__name__}: {str(e)[:100]}')
open('state/usable_backbones.txt','w').write('\n'.join(ok))
PY
  mark dryA; say "[dry A] done"
fi
USABLE=$(cat state/usable_backbones.txt 2>/dev/null || true)
say "usable backbones: ${USABLE:-NONE}"
[ -z "$USABLE" ] && { say "no usable backbone — stopping"; exit 1; }

# ---------------------------------------------------------------------------
# DRY RUN STAGE B — 200 rows / 1 epoch end to end per backbone. Exercises the
# collator, two-head loss, fp16/bf16, checkpoint save and predict path. A bug
# here costs a minute; the same bug 20 minutes into a full run costs the run.
# ---------------------------------------------------------------------------
if ! have dryB; then
  say "[dry B] 200-row / 1-epoch end-to-end per backbone"
  for M in $USABLE; do
    case $M in
      microsoft/deberta-v3-large) TR="$POOL_EN"  ;;
      *)                          TR="$POOL_NATIVE" ;;
    esac
    say "  dry B: $M"
    python src/train_transformer.py --train $TR --model "$M" \
      --out artifacts/_dry_$(basename $M) --max-train 200 --epochs 1 \
      --batch 8 --attn auto --bf16 >>$L 2>&1 \
      && say "  dry B OK: $M" || { say "  dry B FAILED: $M — stopping"; exit 1; }
  done
  mark dryB; say "[dry B] done — all backbones train end to end"
fi

# ---------------------------------------------------------------------------
# FULL RUNS. batch 16 + grad-accum 2 keeps large models inside 24 GB.
# ---------------------------------------------------------------------------
run_full() {
  local key=$1 model=$2 pool=$3 out=$4
  have "$key" && { say "[$key] already done"; return; }
  say "[$key] full train: $model"
  python src/train_transformer.py --train $pool --model "$model" --out "$out" \
    --epochs 3 --batch 16 --grad-accum 2 --lr 1e-5 --aux-weight 0.3 \
    --soft-alpha 0.4 --attn auto --bf16 --dev artifacts/dev_holdout.csv \
    --dev-frac 0.1 >>$L 2>&1
  if [ -f "$out/setu_model.pt" ]; then
    for LG in hi bn; do
      # DeBERTa is English-only, so it must score the back-translated view
      local TEST=artifacts/test_$LG.csv COL=text
      [ "$key" = "deberta_en" ] && { TEST=artifacts/test_${LG}_en.csv; COL=text_en; }
      python src/predict.py --model "$out" --test $TEST --text-col $COL \
        --out artifacts/probs_${key}_$LG.csv --batch 64 >>$L 2>&1
    done
    mark "$key"; say "[$key] done"
  else
    say "[$key] FAILED — see log"; mark "$key"
  fi
}

echo "$USABLE" | grep -q muril-large     && run_full muril_large  google/muril-large-cased   "$POOL_NATIVE" artifacts/model_muril_large
echo "$USABLE" | grep -q deberta-v3-large && run_full deberta_en  microsoft/deberta-v3-large "$POOL_EN"     artifacts/model_deberta_en
echo "$USABLE" | grep -q xlm-roberta-large && run_full xlmr_large xlm-roberta-large   "$POOL_NATIVE" artifacts/model_xlmr_large

say "=== GPU pipeline finished ==="
