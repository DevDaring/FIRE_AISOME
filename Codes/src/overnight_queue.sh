#!/usr/bin/env bash
# Sequential work queue for an unattended window.
#
# Runs the remaining improvements one at a time on CPU. Deliberately sequential:
# two transformer fine-tunes in parallel on the same 32 cores contend and both
# finish slower than either would alone.
#
# NO GPU IS RENTED BY THIS SCRIPT. Everything here runs locally and costs nothing.
#
# Each step writes a marker to artifacts/queue_state/ when it completes, so a
# restart skips finished work instead of redoing hours of it.
#
#   bash src/overnight_queue.sh
set -uo pipefail
cd "$(dirname "$0")/.."

A=artifacts
S=$A/queue_state
mkdir -p "$S" logs
PY=python3.12

log() { echo "$(date -u '+%F %T UTC')  $*" | tee -a logs/overnight.log; }
done_marker() { [ -f "$S/$1.done" ]; }
mark()       { touch "$S/$1.done"; }

log "=== overnight queue started (no GPU, nothing billable) ==="

# ---------------------------------------------------------------------------
# 1. Refit the distilled model on ALL 971 judge labels.
#    Legitimate transduction: the labels are our own panel's predictions on
#    unlabelled test comments, never organizer gold. Standard practice is to
#    refit on everything once the recipe is chosen.
#    NOTE: this model trains on the dev_holdout rows, so its dev score is
#    meaningless. Strategy selection stays with the leak-free variant.
# ---------------------------------------------------------------------------
if ! done_marker refit_all; then
    log "[1/4] refit MuRIL on all 971 judge labels"
    $PY src/split_dev.py --dev-frac 0.0 --outdir "$A/full" > logs/split_full.log 2>&1
    $PY src/train_transformer.py \
        --train $A/train_en.csv $A/train_en.to-hi.csv::text_hi \
                $A/train_en.to-bn.csv::text_bn $A/synth_train.csv \
                $A/full/distil_hi.csv $A/full/distil_bn.csv \
        --model google/muril-base-cased --out $A/model_refit_all \
        --epochs 2 --batch 16 --aux-weight 0.3 --soft-alpha 0.4 --dev-frac 0.08 \
        > logs/refit_all.log 2>&1
    if [ -f $A/model_refit_all/setu_model.pt ]; then
        for L in hi bn; do
            $PY src/predict.py --model $A/model_refit_all --test $A/test_$L.csv \
                --out $A/probs_refit_$L.csv --batch 64 >> logs/refit_all.log 2>&1
        done
        mark refit_all; log "[1/4] done"
    else
        log "[1/4] FAILED — see logs/refit_all.log"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Third backbone for ensemble diversity. IndicBERTv2 is trained on Indian
#    languages by AI4Bharat and is architecturally distinct from both MuRIL and
#    XLM-R, so its errors should decorrelate.
# ---------------------------------------------------------------------------
if ! done_marker indicbert; then
    log "[2/4] distil IndicBERTv2 (third backbone)"
    $PY src/train_transformer.py \
        --train $A/train_en.csv $A/train_en.to-hi.csv::text_hi \
                $A/train_en.to-bn.csv::text_bn $A/synth_train.csv \
                $A/distil_hi.csv $A/distil_bn.csv \
        --model ai4bharat/IndicBERTv2-MLM-only --out $A/model_indicbert \
        --epochs 2 --batch 16 --aux-weight 0.3 --soft-alpha 0.4 \
        --dev $A/dev_holdout.csv --dev-frac 0.1 \
        > logs/indicbert.log 2>&1
    if [ -f $A/model_indicbert/setu_model.pt ]; then
        for L in hi bn; do
            $PY src/predict.py --model $A/model_indicbert --test $A/test_$L.csv \
                --out $A/probs_indic_$L.csv --batch 64 >> logs/indicbert.log 2>&1
            $PY src/calibrate.py fit --probs $A/probs_indic_$L.csv \
                --dev $A/dev_holdout.csv --lang $L --out $A/calib_indic_$L.json \
                >> logs/indicbert.log 2>&1 \
              && $PY src/calibrate.py apply --probs $A/probs_indic_$L.csv \
                    --calib $A/calib_indic_$L.json --out $A/probs_indic_$L.cal.csv \
                    >> logs/indicbert.log 2>&1
        done
        mark indicbert; log "[2/4] done"
    else
        log "[2/4] FAILED (backbone may be gated or unavailable) — non-fatal"
        mark indicbert
    fi
fi

# ---------------------------------------------------------------------------
# 3. Second seed of the winning recipe. Averaging seeds is one of the few
#    reliably positive, zero-risk ensembling tricks on a small dev set.
# ---------------------------------------------------------------------------
if ! done_marker seed2; then
    log "[3/4] second seed of the distilled recipe"
    $PY src/train_transformer.py \
        --train $A/train_en.csv $A/train_en.to-hi.csv::text_hi \
                $A/train_en.to-bn.csv::text_bn $A/synth_train.csv \
                $A/distil_hi.csv $A/distil_bn.csv \
        --model google/muril-base-cased --out $A/model_seed2 \
        --epochs 2 --batch 16 --aux-weight 0.3 --soft-alpha 0.4 \
        --dev $A/dev_holdout.csv --dev-frac 0.1 --seed 1337 \
        > logs/seed2.log 2>&1
    if [ -f $A/model_seed2/setu_model.pt ]; then
        for L in hi bn; do
            $PY src/predict.py --model $A/model_seed2 --test $A/test_$L.csv \
                --out $A/probs_seed2_$L.csv --batch 64 >> logs/seed2.log 2>&1
            $PY src/calibrate.py fit --probs $A/probs_seed2_$L.csv \
                --dev $A/dev_holdout.csv --lang $L --out $A/calib_seed2_$L.json \
                >> logs/seed2.log 2>&1 \
              && $PY src/calibrate.py apply --probs $A/probs_seed2_$L.csv \
                    --calib $A/calib_seed2_$L.json --out $A/probs_seed2_$L.cal.csv \
                    >> logs/seed2.log 2>&1
        done
        mark seed2; log "[3/4] done"
    else
        log "[3/4] FAILED — see logs/seed2.log"; mark seed2
    fi
fi

# ---------------------------------------------------------------------------
# 4. Re-run honest strategy selection over whatever channels now exist.
#    Writes a recommendation only — it does NOT touch the submission. Replacing
#    a validated submission is a judgement call, not something a shell loop
#    should do unsupervised.
# ---------------------------------------------------------------------------
log "[4/4] honest strategy re-selection over all available channels"
$PY src/select_strategy.py > logs/select_strategy.log 2>&1 \
    && log "[4/4] done — see logs/select_strategy.log" \
    || log "[4/4] selection failed — see logs/select_strategy.log"

log "=== overnight queue finished ==="
