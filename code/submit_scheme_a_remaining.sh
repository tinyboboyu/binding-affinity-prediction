#!/bin/bash
# Submit the remaining Scheme A leave-one-out runs.
# Assumes the 6QLN test run with validation sample 6QLO has already been completed.

set -euo pipefail

cd /lunarc/nobackup/projects/teobio/Xiaofan/binding_affinity_prediction/code

declare -a TEST_SAMPLE_IDS=("6QLO" "6QLP" "6QLR" "6QLT")
declare -a VAL_SAMPLE_IDS=("6QLP" "6QLR" "6QLT" "6QLN")

for index in "${!TEST_SAMPLE_IDS[@]}"; do
  test_sample_id="${TEST_SAMPLE_IDS[$index]}"
  val_sample_id="${VAL_SAMPLE_IDS[$index]}"
  echo "Submitting Scheme A run: test=${test_sample_id}, val=${val_sample_id}"
  sbatch \
    --job-name="pl_${test_sample_id}" \
    --export="ALL,TEST_SAMPLE_ID=${test_sample_id},VAL_SAMPLE_ID=${val_sample_id}" \
    run_train_scheme_a.sbatch
done
