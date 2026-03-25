#!/bin/bash
# Hyperparameter search for async MLP on neural decoding (indy_20160622_01)
# Uses half the CPU cores (14 out of 28)
# Target: R² >= 0.687 (SNN benchmark on indy_20160622_01)
# Early stopping: kills runs if val R² stagnates for 5 epochs

cd /home/I6256161/AiNed
source venv/bin/activate
export JAX_PLATFORMS=cpu

RESULTS_DIR="network_results/neural_decoding/training/MLP"
LOG_DIR="results/hyperparam_search"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/search_${TIMESTAMP}.log"
STAGNATION_PATIENCE=5

log_msg() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$MASTER_LOG"
}

echo "=== Hyperparameter Search Started at $(date) ===" | tee "$MASTER_LOG"
echo "Target: R² >= 0.687 | Early stopping patience: $STAGNATION_PATIENCE epochs" | tee -a "$MASTER_LOG"

# Monitor a running experiment and kill it if val R² stagnates for STAGNATION_PATIENCE epochs
# Args: PID, LOG_FILE, RUN_NAME
monitor_and_kill_stagnant() {
    local PID=$1
    local LOG_FILE=$2
    local RUN_NAME=$3
    local best_val="-999"
    local stagnant_count=0
    local last_epoch=-1

    while kill -0 "$PID" 2>/dev/null; do
        sleep 30

        # Get the latest epoch number
        local current_epoch=$(grep "^Epoch" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP 'Epoch \d+' | grep -oP '\d+')
        if [ -z "$current_epoch" ]; then
            continue
        fi

        # Skip if no new epoch
        if [ "$current_epoch" = "$last_epoch" ]; then
            continue
        fi

        # Get current val R²
        local current_val=$(grep "^Epoch $current_epoch " "$LOG_FILE" | grep -oP 'Validation R2: [-\d.]+' | grep -oP '[-\d.]+$')
        if [ -z "$current_val" ]; then
            continue
        fi

        # Check improvement using awk for float comparison
        local improved=$(awk "BEGIN {print ($current_val > $best_val + 0.001) ? 1 : 0}")
        if [ "$improved" = "1" ]; then
            best_val="$current_val"
            stagnant_count=0
        else
            stagnant_count=$((stagnant_count + 1))
        fi

        last_epoch="$current_epoch"

        if [ "$stagnant_count" -ge "$STAGNATION_PATIENCE" ]; then
            log_msg "EARLY STOP: $RUN_NAME stagnated at ValR²=$best_val for $STAGNATION_PATIENCE epochs (ep $current_epoch). Killing."
            # Kill the training process and all its children (mpirun spawns workers)
            # setsid gave it its own process group, so kill that group
            kill -- -"$PID" 2>/dev/null
            pkill -P "$PID" 2>/dev/null
            wait "$PID" 2>/dev/null
            return 1
        fi
    done
    return 0
}

# Run an experiment with early stopping monitor
# Args: CONFIG_FILE, CORES, NPROCS, RUN_NAME
run_experiment() {
    local CONFIG_FILE=$1
    local CORES=$2
    local NPROCS=$3
    local RUN_NAME=$4
    local LOG_FILE="$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log"

    log_msg "Starting: $RUN_NAME (cores=$CORES, nprocs=$NPROCS)"

    setsid taskset -c $CORES mpirun -n $NPROCS python async_MLP_neural_decoding.py \
        --config "$CONFIG_FILE" \
        --filename "indy_20160622_01.mat" \
        > "$LOG_FILE" 2>&1 &
    local TRAIN_PID=$!

    # Start monitor in background
    monitor_and_kill_stagnant "$TRAIN_PID" "$LOG_FILE" "$RUN_NAME" &
    local MONITOR_PID=$!

    # Wait for training to finish (either naturally or killed by monitor)
    wait "$TRAIN_PID" 2>/dev/null

    # Kill monitor if still running
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null

    # Extract results
    local BEST_VAL=$(grep "Validation R2" "$LOG_FILE" | grep -oP 'Validation R2: [-\d.]+' | grep -oP '[-\d.]+$' | sort -n | tail -1)
    local TEST_R2=$(grep "test Epoch R2" "$LOG_FILE" | tail -1 | grep -oP 'R2: [-\d.]+' | grep -oP '[-\d.]+$')
    local LAST_EP=$(grep "^Epoch" "$LOG_FILE" | tail -1 | grep -oP 'Epoch \d+' | grep -oP '\d+')
    log_msg "Finished: $RUN_NAME -> ValR²=${BEST_VAL:-N/A} TestR²=${TEST_R2:-N/A} (ep ${LAST_EP:-?})"
    echo "  Log: $LOG_FILE" | tee -a "$MASTER_LOG"
}

make_config() {
    local FILE=$1
    local LAYERS=$2
    local BATCH=$3
    local EPOCHS=$4
    local LR=$5
    local BIAS=$6
    local RESTRICT=$7
    local FIRE=$8
    local WREG=$9
    local SPARSITY=${10}

    # Count layers for sparsity array
    local NLAYERS=$(echo "$LAYERS" | tr ',' '\n' | wc -l)
    local SPARSITY_ARR="["
    for i in $(seq 1 $NLAYERS); do
        if [ $i -gt 1 ]; then SPARSITY_ARR+=", "; fi
        SPARSITY_ARR+="$SPARSITY"
    done
    SPARSITY_ARR+="]"

    cat > "$FILE" << EOF
dataset: neural_decoding
layer_sizes: [$LAYERS]
collapse_units: true
preserve_exact_times: false
mode: training
use_bias: $BIAS
batch_size: $BATCH
num_epochs: $EPOCHS
learning_rate: $LR
optimizer: adam
load_file: false
best: false
rerun: null
restrict: $RESTRICT
init_thresholds: 0.0
shuffle_activations: false
shuffle_input: false
firing_nb: $FIRE
sync_rate: 1
exploration_rate: 0.0
threshold_lr: 0.0
sparsity_impact: $SPARSITY_ARR
w_reg: $WREG
top_weights: -1
history_size: 0
EOF
}

# ============================================================================
# ROUND 2 (restart): Deeper networks + restrict + firing_nb + h=512
# ============================================================================
echo "" | tee -a "$MASTER_LOG"
log_msg "=== ROUND 2: Deeper/wider + restrict + firing_nb ==="

#                    file               layers           batch ep   lr      bias  restrict fire wreg   sparsity
make_config /tmp/nd_r2_a.yaml "96, 128, 64, 2"          64   30   0.001   true  1        1    0.0    0.0
make_config /tmp/nd_r2_b.yaml "96, 128, 2"              64   30   0.001   true  0.5      1    0.0    0.0
make_config /tmp/nd_r2_c.yaml "96, 128, 2"              64   30   0.001   true  1        5    0.0    0.0
make_config /tmp/nd_r2_d.yaml "96, 512, 2"              64   30   0.001   true  1        1    0.0001 0.0

run_experiment /tmp/nd_r2_a.yaml "0-3" 4 "r2a_4layer_128_64" &
P1=$!
run_experiment /tmp/nd_r2_b.yaml "4-6" 3 "r2b_restrict05" &
P2=$!
run_experiment /tmp/nd_r2_c.yaml "7-9" 3 "r2c_fire5" &
P3=$!
run_experiment /tmp/nd_r2_d.yaml "10-12" 3 "r2d_h512_wreg" &
P4=$!
wait $P1 $P2 $P3 $P4
log_msg "=== ROUND 2 COMPLETE ==="

# ============================================================================
# ROUND 3: Best from R1 insights — h=256 is good, lower lr generalizes better
# Try to fix overfitting with w_reg, lower lr, larger batch
# ============================================================================
echo "" | tee -a "$MASTER_LOG"
log_msg "=== ROUND 3: Fix overfitting on h=256 ==="

make_config /tmp/nd_r3_a.yaml "96, 256, 2"              64   50   0.0005  true  1        1    0.0    0.0
make_config /tmp/nd_r3_b.yaml "96, 256, 2"              64   50   0.001   true  1        1    0.001  0.0
make_config /tmp/nd_r3_c.yaml "96, 256, 2"              252  50   0.001   true  1        1    0.0    0.0
make_config /tmp/nd_r3_d.yaml "96, 256, 2"              64   50   0.002   true  1        5    0.0    0.0

run_experiment /tmp/nd_r3_a.yaml "0-2" 3 "r3a_h256_lr0005_50ep" &
P1=$!
run_experiment /tmp/nd_r3_b.yaml "3-5" 3 "r3b_h256_wreg001_50ep" &
P2=$!
run_experiment /tmp/nd_r3_c.yaml "6-8" 3 "r3c_h256_b252_50ep" &
P3=$!
run_experiment /tmp/nd_r3_d.yaml "9-11" 3 "r3d_h256_lr002_fire5_50ep" &
P4=$!
wait $P1 $P2 $P3 $P4
log_msg "=== ROUND 3 COMPLETE ==="

# ============================================================================
# ROUND 4: Higher firing_nb sweep (more sync-like) + h=256/512
# ============================================================================
echo "" | tee -a "$MASTER_LOG"
log_msg "=== ROUND 4: Higher firing_nb + larger hidden ==="

make_config /tmp/nd_r4_a.yaml "96, 256, 2"              64   50   0.0005  true  1        10   0.0    0.0
make_config /tmp/nd_r4_b.yaml "96, 256, 2"              64   50   0.0005  true  1        20   0.0    0.0
make_config /tmp/nd_r4_c.yaml "96, 512, 2"              64   50   0.0005  true  1        1    0.0    0.0
make_config /tmp/nd_r4_d.yaml "96, 256, 2"              64   50   0.0005  true  0.7      5    0.0    0.0

run_experiment /tmp/nd_r4_a.yaml "0-2" 3 "r4a_h256_fire10_50ep" &
P1=$!
run_experiment /tmp/nd_r4_b.yaml "3-5" 3 "r4b_h256_fire20_50ep" &
P2=$!
run_experiment /tmp/nd_r4_c.yaml "6-8" 3 "r4c_h512_lr0005_50ep" &
P3=$!
run_experiment /tmp/nd_r4_d.yaml "9-11" 3 "r4d_h256_restrict07_fire5" &
P4=$!
wait $P1 $P2 $P3 $P4
log_msg "=== ROUND 4 COMPLETE ==="

# ============================================================================
# ROUND 5: Long training on best combos (100 epochs)
# ============================================================================
echo "" | tee -a "$MASTER_LOG"
log_msg "=== ROUND 5: Long training (100ep) on best combos ==="

make_config /tmp/nd_r5_a.yaml "96, 256, 2"              64   100  0.0005  true  1        10   0.0    0.0
make_config /tmp/nd_r5_b.yaml "96, 512, 2"              64   100  0.0005  true  1        5    0.0    0.0
make_config /tmp/nd_r5_c.yaml "96, 256, 2"              64   100  0.0005  true  1        1    0.001  0.0
make_config /tmp/nd_r5_d.yaml "96, 256, 128, 2"         64   100  0.0005  true  1        5    0.0    0.0

run_experiment /tmp/nd_r5_a.yaml "0-2" 3 "r5a_h256_fire10_100ep" &
P1=$!
run_experiment /tmp/nd_r5_b.yaml "3-5" 3 "r5b_h512_fire5_100ep" &
P2=$!
run_experiment /tmp/nd_r5_c.yaml "6-8" 3 "r5c_h256_wreg_100ep" &
P3=$!
run_experiment /tmp/nd_r5_d.yaml "9-12" 4 "r5d_4layer_256_128_fire5_100ep" &
P4=$!
wait $P1 $P2 $P3 $P4
log_msg "=== ROUND 5 COMPLETE ==="

# ============================================================================
# ROUND 6: Extreme configs — very high firing_nb (near synchronous), h=1024
# ============================================================================
echo "" | tee -a "$MASTER_LOG"
log_msg "=== ROUND 6: Near-synchronous + very wide ==="

make_config /tmp/nd_r6_a.yaml "96, 256, 2"              64   50   0.0005  true  1        50   0.0    0.0
make_config /tmp/nd_r6_b.yaml "96, 256, 2"              64   50   0.0005  true  1        128  0.0    0.0
make_config /tmp/nd_r6_c.yaml "96, 1024, 2"             64   50   0.0003  true  1        1    0.001  0.0
make_config /tmp/nd_r6_d.yaml "96, 256, 2"              128  50   0.0005  true  1        10   0.0001 0.0

run_experiment /tmp/nd_r6_a.yaml "0-2" 3 "r6a_h256_fire50" &
P1=$!
run_experiment /tmp/nd_r6_b.yaml "3-5" 3 "r6b_h256_fire128" &
P2=$!
run_experiment /tmp/nd_r6_c.yaml "6-8" 3 "r6c_h1024_wreg" &
P3=$!
run_experiment /tmp/nd_r6_d.yaml "9-11" 3 "r6d_h256_b128_fire10_wreg" &
P4=$!
wait $P1 $P2 $P3 $P4
log_msg "=== ROUND 6 COMPLETE ==="

# ============================================================================
# Summary
# ============================================================================
echo "" | tee -a "$MASTER_LOG"
log_msg "=== ALL ROUNDS COMPLETE at $(date) ==="
echo "" | tee -a "$MASTER_LOG"
echo "=== RESULTS SUMMARY (sorted by Test R²) ===" | tee -a "$MASTER_LOG"
for f in "$LOG_DIR"/r*_${TIMESTAMP}.log; do
    name=$(basename "$f" _${TIMESTAMP}.log)
    best_val=$(grep "Validation R2" "$f" | grep -oP 'Validation R2: [-\d.]+' | grep -oP '[-\d.]+$' | sort -n | tail -1)
    test_r2=$(grep "test Epoch R2" "$f" | tail -1 | grep -oP 'R2: [-\d.]+' | grep -oP '[-\d.]+$')
    last_ep=$(grep "^Epoch" "$f" | tail -1 | grep -oP 'Epoch \d+' | grep -oP '\d+')
    echo "$name | ValR²: ${best_val:-N/A} | TestR²: ${test_r2:-N/A} | Ep: ${last_ep:-?}"
done 2>/dev/null | sort -t'|' -k3 -rn | tee -a "$MASTER_LOG"
