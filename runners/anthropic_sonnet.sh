#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# Anthropic Sonnet arm, both rounds. Reproduces results/anthropic_20260810T044952Z.
#
# Round 2 resumes into round 1's tree by run id, so the 1,401 traces from round 1
# are reused from AgentDojo's per-task cache and only repetition 2 is paid for.
# Do not change the run id: a new one starts a new tree and pays for everything.
#
# --temperature 1.0 is NOT a preference. claude-sonnet-5 rejects the temperature
# field outright, so 1.0 is the only value the whole lineage accepts, and the
# configuration hash necessarily diverges from the other arms.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=20260810T044952Z

common=(--arm anthropic --preset sonnet-lineage --suites slack travel banking
        --attack important_instructions_no_names --temperature 1.0
        --allow-config-change --concurrency 6 --run-id $RUN_ID)

echo "### ROUND 1: one repetition"
$PY src/run_sweep.py "${common[@]}" --reps 1 --budget-usd 60

echo "### ROUND 2: adds repetition 2, giving the noise floor"
$PY src/run_sweep.py "${common[@]}" --reps 2 --budget-usd 60

$PY src/export_results.py results/anthropic_$RUN_ID
$PY src/noise_floor.py results_export/$RUN_ID --suites slack travel banking
