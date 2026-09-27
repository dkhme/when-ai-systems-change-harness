#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
set -u
cd "$(dirname "$0")/.."
# GEMINI_API_KEY is read from the environment, or from .env at the repository
# root, which src/run_sweep.py loads. See .env.example.
PY=.venv/bin/python
S3="slack travel banking"

run () { echo "### ITEM $1: $2"; shift 2; $PY src/run_sweep.py --arm google "$@" 2>&1 | tail -14; echo; }

# 2. Close the hole: the "before" model of the one regression, under the name-neutral attack.
run 2 "2.5-flash-lite under no_names" \
  --models gemini-2.5-flash-lite --suites ${=S3} --reps 2 \
  --attack important_instructions_no_names --allow-config-change --budget-usd 6

# 1. Mechanism: was the regression a reasoning-default change rather than a weight change?
run 1 "thinking-normalized, flash-lite lineage" \
  --preset flash-lite-lineage --suites ${=S3} --reps 2 \
  --thinking-mode normalized --allow-config-change --budget-usd 30

# 3. Attack independence.
run 3 "tool_knowledge attack, flash-lite lineage" \
  --preset flash-lite-lineage --suites ${=S3} --reps 2 \
  --attack tool_knowledge --allow-config-change --budget-usd 30

# 4. Suite independence. Same config as the frozen lock, so poolable with the main matrix.
run 4 "workspace suite, flash-lite lineage" \
  --preset flash-lite-lineage --suites workspace --reps 2 --budget-usd 35

# 5. Does a deployed defense carry you through the upgrade?
run 5 "spotlighting defense, flash-lite lineage" \
  --preset flash-lite-lineage --suites ${=S3} --reps 2 \
  --defense spotlighting_with_delimiting --allow-config-change --budget-usd 35

echo "### ALL FIVE COMPLETE"
