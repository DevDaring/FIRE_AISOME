#!/usr/bin/env bash
# ============================================================================
# SETU — AISoMe 2026 end-to-end pipeline.
#
#   bash run_all.sh stage0          # no test data needed — START HERE, it is the
#                                   # long pole (synthetic corpus + translation)
#   bash run_all.sh stage1          # needs the test files: normalise + committee
#   bash run_all.sh annotate        # draw the dev sheet to hand-annotate
#   bash run_all.sh stage2          # train + self-train the encoder
#   bash run_all.sh stage3          # NLI + calibrate + fuse + evaluate
#   bash run_all.sh submit          # build and validate the submission ZIP
#   bash run_all.sh all             # stage0..stage3 (still needs manual annotation)
#
# Put the organizers' two test files in Dataset/Testing_Data/ first (any of
# .csv/.tsv/.txt/.xlsx). Filenames containing "hindi"/"bengali" are detected
# automatically; otherwise the dominant script is sniffed.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.12}"
A=artifacts
STAGE="${1:-all}"

$PY -c "import pandas, torch, transformers, sklearn" 2>/dev/null || {
    echo "ERROR: $PY lacks pandas/torch/transformers/sklearn."
    echo "  $PY -m pip install -r requirements.txt"
    exit 1
}

# ---- knobs -----------------------------------------------------------------
BACKBONE="${BACKBONE:-google/muril-base-cased}"   # MuRIL handles romanised Indic best
BACKBONE2="${BACKBONE2:-xlm-roberta-base}"        # second, error-decorrelated encoder
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-16}"
AUX_WEIGHT="${AUX_WEIGHT:-0.3}"                   # argument-node auxiliary head
SOFT_ALPHA="${SOFT_ALPHA:-0.4}"                   # soft-label KL term
PER_CELL="${PER_CELL:-6}"                         # synthetic comments per taxonomy cell
GENERATORS="${GENERATORS:-or-qwen or-deepseek or-mistral}"  # round-robin, no Google
MEMBERS="${MEMBERS:-}"                            # empty = whole teacher tier (llm.py)
SELFTRAIN_ROUNDS="${SELFTRAIN_ROUNDS:-2}"
TRANSLIT="${TRANSLIT:-none}"                      # none | auto | local | llm
SEED_N="${SEED_N:-40}"                            # hand-labelled anchor set size
SILVER_N="${SILVER_N:-300}"                       # comments the LLM judges adjudicate
TEAM="${TEAM:-Nirnay}"
# ----------------------------------------------------------------------------

banner() { echo; echo "############ $* ############"; }

find_test() { $PY - <<'EOF'
import sys
sys.path.insert(0, "src")
from common import find_test_files
f = find_test_files()
print((str(f["hi"]) if f["hi"] else "") + "|" + (str(f["bn"]) if f["bn"] else ""))
EOF
}

require_test() {
    IFS='|' read -r HI_TEST BN_TEST <<< "$(find_test)"
    if [[ -z "$HI_TEST" && -z "$BN_TEST" ]]; then
        echo "ERROR: no test files in Dataset/Testing_Data/."
        echo "       The organizers email them to registered teams — chase"
        echo "       aisome.fire2026@gmail.com. Deadline is 31 July 2026."
        exit 1
    fi
    echo "  hindi  : ${HI_TEST:-<missing>}"
    echo "  bengali: ${BN_TEST:-<missing>}"
}

# ============================================================================
stage0() {
    banner "STAGE 0 — data that needs NO test file (run this first)"

    echo "--- [0.1] English training pool (GWSD soft labels + SemEval-CC) ---"
    $PY src/prepare_data.py

    echo "--- [0.2] taxonomy sanity check ---"
    $PY src/taxonomy.py | head -8

    echo "--- [0.3] LLM connectivity ---"
    $PY src/llm.py

    echo "--- [0.4] taxonomy-conditioned synthetic corpus (the novelty engine) ---"
    $PY src/synth_generate.py --per-cell "$PER_CELL" --generators $GENERATORS \
        --out "$A/synth_train.csv"

    echo "--- [0.5] translate the English pool -> hi, bn ---"
    $PY src/translate.py --input "$A/train_en.csv" --text-col text --to hi bn \
        --provider auto
}

stage1() {
    banner "STAGE 1 — normalise the test files and run the LLM committee"
    require_test

    echo "--- [1.1] script-aware normalisation ---"
    $PY src/normalize.py ${HI_TEST:+--hi "$HI_TEST"} ${BN_TEST:+--bn "$BN_TEST"} \
        --translit "$TRANSLIT"

    echo "--- [1.2] transductive taxonomy-grounded committee ---"
    local tests=()
    [[ -f "$A/test_hi.csv" ]] && tests+=("$A/test_hi.csv")
    [[ -f "$A/test_bn.csv" ]] && tests+=("$A/test_bn.csv")
    $PY src/llm_committee.py --test "${tests[@]}" ${MEMBERS:+--members $MEMBERS}

    echo "--- [1.3] translate the test comments -> English (for EN-only channels) ---"
    for L in hi bn; do
        [[ -f "$A/test_$L.csv" ]] || continue
        $PY src/translate.py --input "$A/test_$L.csv" --text-col text --to en \
            --from-lang "$L" --provider auto --output "$A/test_${L}_en.csv"
    done
}

committee_files() {
    local cs=()
    [[ -f "$A/committee_hi.csv" ]] && cs+=("$A/committee_hi.csv")
    [[ -f "$A/committee_bn.csv" ]] && cs+=("$A/committee_bn.csv")
    if [[ ${#cs[@]} -eq 0 ]]; then
        echo "ERROR: run stage1 first (no committee_*.csv found)" >&2; exit 1
    fi
    echo "${cs[@]}"
}

# Small hand-labelled ANCHOR set (~25 min). Not optional even in the LLM-judged
# route: it is the only thing that tells us whether the silver labels can be
# trusted, and it turns an unknown risk into a number for the paper.
seed() {
    banner "SEED — hand-label $SEED_N comments (~25 min). This is the anchor."
    local cs=(); read -r -a cs <<< "$(committee_files)"
    $PY src/annotate_dev.py sample --committee "${cs[@]}" --n "$SEED_N" \
        --out "$A/seed_to_annotate.csv"
    cat <<EOF

  NEXT (manual, ~25 minutes):
    1. open $A/seed_to_annotate.csv
    2. read $A/seed_to_annotate_CODEBOOK.txt
    3. fill 'gold' with Favour | Against | None   ('?' if you truly cannot decide)
    4. $PY src/annotate_dev.py finalise --input $A/seed_to_annotate.csv \\
           --out $A/seed_gold.csv
    5. bash run_all.sh silver
EOF
}

# LLM-judged dev set, scored against the human seed.
silver() {
    banner "SILVER — LLM judges label the dev set, anchored to your hand-labelled seed"
    local cs=(); read -r -a cs <<< "$(committee_files)"
    local en=()
    [[ -f "$A/test_hi_en.csv" ]] && en+=("$A/test_hi_en.csv")
    [[ -f "$A/test_bn_en.csv" ]] && en+=("$A/test_bn_en.csv")
    [[ ${#en[@]} -eq 0 ]] && echo "  NOTE: no back-translations found — run stage1 [1.3] first;
        the English pivot is most of the judges' accuracy."
    if [[ ! -f "$A/seed_gold.csv" ]]; then
        echo "  WARNING: no $A/seed_gold.csv. The silver labels will have NO measured"
        echo "           reliability. Run 'bash run_all.sh seed' first — 25 minutes."
    fi
    $PY src/silver_dev.py --committee "${cs[@]}" ${#en[@]:+--test-en "${en[@]}"} \
        --seed-gold "$A/seed_gold.csv" --n "$SILVER_N" --out "$A/dev_gold.csv"
    cat <<EOF

  Read the VERDICT printed above:
    SAFE    (>= 0.85 vs your seed)  proceed to stage2 / stage3
    USABLE  (0.70 - 0.85)           proceed, but label $A/human_queue.csv too
    UNSAFE  (< 0.70)                do NOT calibrate on this — run 'annotate' instead
EOF
}

# Full manual route — the gold standard if you have the two hours.
annotate() {
    banner "ANNOTATE — draw the full dev sheet (~2 hours, the most reliable option)"
    local cs=(); read -r -a cs <<< "$(committee_files)"
    $PY src/annotate_dev.py sample --committee "${cs[@]}" --n 150 \
        --out "$A/dev_to_annotate.csv"
    cat <<EOF

  NEXT (manual):
    1. open $A/dev_to_annotate.csv
    2. read $A/dev_to_annotate_CODEBOOK.txt
    3. fill the 'gold' column with Favour | Against | None  ('?' to skip)
    4. $PY src/annotate_dev.py finalise --input $A/dev_to_annotate.csv \\
           --out $A/dev_gold.csv

  Without dev_gold.csv, calibrate.py and fuse.py --search cannot run, and you are
  choosing between three runs by guesswork.
EOF
}

stage2() {
    banner "STAGE 2 — train and self-train the argument-aware encoder"
    local pool=("$A/train_en.csv")
    [[ -f "$A/train_en.to-hi.csv" ]] && pool+=("$A/train_en.to-hi.csv::text_hi")
    [[ -f "$A/train_en.to-bn.csv" ]] && pool+=("$A/train_en.to-bn.csv::text_bn")
    [[ -f "$A/synth_train.csv" ]]    && pool+=("$A/synth_train.csv")
    echo "training pool: ${pool[*]}"

    local tests=() coms=()
    for L in hi bn; do
        [[ -f "$A/test_$L.csv" ]]      && tests+=("$A/test_$L.csv")
        [[ -f "$A/committee_$L.csv" ]] && coms+=("$A/committee_$L.csv")
    done

    echo "--- [2.1] LaBSE baseline (fast reference row for the paper) ---"
    $PY src/train_baseline.py --train "${pool[@]}" --out "$A/model_baseline" \
        ${DEV_GOLD:+--dev "$DEV_GOLD"} || echo "  (baseline failed — non-fatal)"

    if [[ ${#coms[@]} -gt 0 && ${#tests[@]} -gt 0 ]]; then
        echo "--- [2.2] committee distillation + self-training ($BACKBONE) ---"
        $PY src/selftrain.py --base-train "${pool[@]}" \
            --committee "${coms[@]}" --test "${tests[@]}" \
            --model "$BACKBONE" --out "$A/model_setu" \
            --rounds "$SELFTRAIN_ROUNDS" --epochs "$EPOCHS" --batch "$BATCH" \
            --aux-weight "$AUX_WEIGHT" --soft-alpha "$SOFT_ALPHA"
    else
        echo "--- [2.2] no committee/test files: plain supervised training only ---"
        $PY src/train_transformer.py --train "${pool[@]}" --model "$BACKBONE" \
            --out "$A/model_setu/round0" --epochs "$EPOCHS" --batch "$BATCH" \
            --aux-weight "$AUX_WEIGHT" --soft-alpha "$SOFT_ALPHA"
    fi

    echo "--- [2.3] second backbone for error decorrelation ($BACKBONE2) ---"
    $PY src/train_transformer.py --train "${pool[@]}" --model "$BACKBONE2" \
        --out "$A/model_xlmr" --epochs "$EPOCHS" --batch "$BATCH" \
        --aux-weight "$AUX_WEIGHT" --soft-alpha "$SOFT_ALPHA" \
        ${DEV_GOLD:+--dev "$DEV_GOLD"} || echo "  (xlm-r failed — non-fatal)"

    for L in hi bn; do
        [[ -f "$A/test_$L.csv" ]] || continue
        for M in "$A/model_setu/round$SELFTRAIN_ROUNDS" "$A/model_setu/round0"; do
            [[ -f "$M/setu_model.pt" ]] || continue
            $PY src/predict.py --model "$M" --test "$A/test_$L.csv" \
                --out "$A/probs_setu_$L.csv"
            break
        done
        [[ -f "$A/model_xlmr/setu_model.pt" ]] && \
            $PY src/predict.py --model "$A/model_xlmr" --test "$A/test_$L.csv" \
                --out "$A/probs_xlmr_$L.csv"
    done
}

stage3() {
    banner "STAGE 3 — NLI channel, calibration, fusion, evaluation"
    local tests=()
    for L in hi bn; do [[ -f "$A/test_$L.csv" ]] && tests+=("$A/test_$L.csv"); done
    [[ ${#tests[@]} -eq 0 ]] && { echo "ERROR: run stage1 first"; exit 1; }

    echo "--- [3.1] training-free claim-conditioned NLI ---"
    $PY src/nli_zeroshot.py --test "${tests[@]}" || echo "  (NLI failed — non-fatal)"

    for L in hi bn; do
        [[ -f "$A/test_$L.csv" ]] || continue
        banner "language: $L"

        if [[ -n "${DEV_GOLD:-}" && -f "${DEV_GOLD:-}" ]]; then
            echo "--- [3.2/$L] calibrate each local encoder for macro-F1 ---"
            # Every submitted run gets its own calibration — model2 and model3 are
            # standalone runs, not just fusion inputs, so they each need the
            # metric-aware decision rule.
            for M in setu xlmr; do
                [[ -f "$A/probs_${M}_$L.csv" ]] || continue
                $PY src/calibrate.py fit --probs "$A/probs_${M}_$L.csv" \
                    --dev "$DEV_GOLD" --lang "$L" --out "$A/calib_${M}_$L.json" \
                    && $PY src/calibrate.py apply --probs "$A/probs_${M}_$L.csv" \
                        --calib "$A/calib_${M}_$L.json" \
                        --out "$A/probs_${M}_$L.cal.csv" \
                    || echo "  ($M/$L calibration failed — continuing uncalibrated)"
            done
        fi

        local enc="$A/probs_setu_$L.cal.csv"
        [[ -f "$enc" ]] || enc="$A/probs_setu_$L.csv"

        echo "--- [3.3/$L] fuse the channels ---"
        # model1 fuses LOCAL channels only, so the submitted run stays a
        # self-contained classifier the organizers could re-run. The committee is
        # already baked in via distillation (stage2); adding it here as a live
        # channel would make the run depend on API calls at inference time.
        local enc2="$A/probs_xlmr_$L.cal.csv"; [[ -f "$enc2" ]] || enc2="$A/probs_xlmr_$L.csv"
        local ch=()
        [[ -f "$enc" ]]                && ch+=("encoder=$enc")
        [[ -f "$enc2" ]]               && ch+=("xlmr=$enc2")
        [[ -f "$A/probs_nli_$L.csv" ]] && ch+=("nli=$A/probs_nli_$L.csv")
        if [[ ${#ch[@]} -lt 2 ]]; then
            echo "  only ${#ch[@]} channel(s) — skipping fusion"
        elif [[ -n "${DEV_GOLD:-}" && -f "${DEV_GOLD:-}" ]]; then
            $PY src/fuse.py --channel "${ch[@]}" --search --dev "$DEV_GOLD" \
                --lang "$L" --out "$A/probs_fused_$L.csv" \
                --report "$A/fuse_report_$L.json"
        else
            echo "  no dev_gold.csv — using equal weights (run 'annotate' to do better)"
            $PY src/fuse.py --channel "${ch[@]}" --out "$A/probs_fused_$L.csv" \
                --report "$A/fuse_report_$L.json"
        fi

        echo "--- [3.4/$L] hard-vote ablation ---"
        [[ ${#ch[@]} -ge 3 ]] && $PY src/ensemble.py \
            --preds "$enc" "$A/committee_$L.csv" "$A/probs_nli_$L.csv" \
            --out "$A/probs_hardvote_$L.csv" ${DEV_GOLD:+--dev "$DEV_GOLD"} || true

        if [[ -n "${DEV_GOLD:-}" && -f "${DEV_GOLD:-}" ]]; then
            echo "--- [3.5/$L] scoreboard ---"
            local pr=()
            [[ -f "$A/probs_setu_$L.csv" ]]     && pr+=("encoder=$A/probs_setu_$L.csv")
            [[ -f "$A/probs_setu_$L.cal.csv" ]] && pr+=("encoder_cal=$A/probs_setu_$L.cal.csv")
            [[ -f "$A/committee_$L.csv" ]]      && pr+=("committee=$A/committee_$L.csv")
            [[ -f "$A/probs_nli_$L.csv" ]]      && pr+=("nli=$A/probs_nli_$L.csv")
            [[ -f "$A/probs_hardvote_$L.csv" ]] && pr+=("hardvote=$A/probs_hardvote_$L.csv")
            [[ -f "$A/probs_fused_$L.csv" ]]    && pr+=("fused=$A/probs_fused_$L.csv")
            $PY src/evaluate.py --dev "$DEV_GOLD" --lang "$L" --pred "${pr[@]}" \
                --errors "$A/errors_$L.csv" --report "$A/eval_report_$L.json"
        fi
    done
}

submit() {
    banner "SUBMIT — build and validate the official ZIP"
    require_test
    # ------------------------------------------------------------------------
    # RUN DESIGN — all three submitted runs are LOCAL TRAINED MODELS.
    #
    # The track asks for "a classifier ... designed by the teams", and reserves
    # the right to ask for the classifier plus a README to reproduce a run. A run
    # that is really a live API call to someone else's chat model satisfies
    # neither, and in FIRE's Indic text-classification tracks (HASOC,
    # DravidianCodeMix) the published field is almost entirely fine-tuned
    # encoders — MuRIL topped those leaderboards.
    #
    # So the LLM committee is used ONLY as a training signal (distilled into the
    # encoders in stage2). It is scored and reported in the working notes as an
    # "LLM reference" row, but it is NOT submitted as a run.
    #
    #   model1 = fusion of the local channels   (MuRIL + XLM-R + NLI, calibrated)
    #   model2 = MuRIL encoder alone            (calibrated, distilled, self-trained)
    #   model3 = XLM-R encoder alone            (calibrated) — falls back to LaBSE
    # ------------------------------------------------------------------------
    local hi=() bn=()
    pick() {  # $1 = lang, echoes "modelN=path" specs
        local L=$1 out=()
        local enc="$A/probs_setu_$L.cal.csv";  [[ -f "$enc"  ]] || enc="$A/probs_setu_$L.csv"
        local enc2="$A/probs_xlmr_$L.cal.csv"; [[ -f "$enc2" ]] || enc2="$A/probs_xlmr_$L.csv"
        [[ -f "$enc2" ]] || enc2="$A/probs_baseline_$L.csv"
        [[ -f "$A/probs_fused_$L.csv" ]] && out+=("model1=$A/probs_fused_$L.csv")
        [[ -f "$enc"  ]] && out+=("model$(( ${#out[@]} + 1 ))=$enc")
        [[ -f "$enc2" ]] && out+=("model$(( ${#out[@]} + 1 ))=$enc2")
        echo "${out[@]}"
    }
    read -r -a hi <<< "$(pick hi)"
    read -r -a bn <<< "$(pick bn)"
    [[ ${#hi[@]} -eq 0 && ${#bn[@]} -eq 0 ]] && {
        echo "ERROR: no prediction files — run stage2/stage3 first"; exit 1; }

    $PY src/make_submission.py \
        ${HI_TEST:+--hi-test "$HI_TEST"} ${BN_TEST:+--bn-test "$BN_TEST"} \
        ${#hi[@]:+--hi "${hi[@]}"} ${#bn[@]:+--bn "${bn[@]}"} \
        --team-name "$TEAM" --format csv
}

# ============================================================================
[[ -f "$A/dev_gold.csv" ]] && DEV_GOLD="$A/dev_gold.csv" || DEV_GOLD=""
if [[ -n "$DEV_GOLD" ]]; then
    echo "using dev set: $DEV_GOLD"
else
    cat <<'EOF'
NOTE: no artifacts/dev_gold.csv yet, so calibration and fusion-weight search are
      disabled — and those were worth more than any modelling change. Build one:

        bash run_all.sh seed     # hand-label 40 comments  (~25 min)
        bash run_all.sh silver   # LLM judges label 300 more, scored on your 40
      or
        bash run_all.sh annotate # hand-label 150 yourself  (~2 h, most reliable)
EOF
fi

case "$STAGE" in
    stage0)   stage0 ;;
    stage1)   stage1 ;;
    seed)     seed ;;
    silver)   silver ;;
    annotate) annotate ;;
    stage2)   stage2 ;;
    stage3)   stage3 ;;
    submit)   submit ;;
    all)      stage0; stage1; stage2; stage3
              echo; echo "Now: bash run_all.sh seed  ->  hand-label 40  ->  "
              echo "     bash run_all.sh silver  ->  bash run_all.sh stage3  ->  submit" ;;
    *)        echo "usage: bash run_all.sh {stage0|stage1|seed|silver|annotate|stage2|stage3|submit|all}"
              echo
              echo "  dev-set routes (pick one):"
              echo "    seed + silver   ~25 min of your time, LLM judges do the rest"
              echo "    annotate        ~2 hours of your time, most reliable"
              exit 1 ;;
esac

banner "DONE: $STAGE"
