#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

echo "Submitting 5 Baseline 2-PB rotating rounds from $(pwd)"

for split_round in 1 2 3 4 5; do
  echo "Submitting rotating round ${split_round}"
  sbatch --export=ALL,SPLIT_ROUND="${split_round}" run_train_baseline2_pb_rotating.sbatch
done
