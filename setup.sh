#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Create the virtual environment and install the pinned harness.
# Idempotent: safe to re-run.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

echo "=============================================================="
echo "AgentDojo Gemini harness setup"
echo "=============================================================="

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10 or newer is required. Found: $("$PYTHON" --version)"
    exit 1
fi
echo "python      $("$PYTHON" --version)"

if [ ! -d "$VENV" ]; then
    echo "creating    $VENV"
    "$PYTHON" -m venv "$VENV"
else
    echo "reusing     $VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "installing  pinned requirements (a few minutes on a cold cache)"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo
echo "installed:"
pip show agentdojo   2>/dev/null | awk '/^Name|^Version/ {printf "  %s\n", $0}'
pip show google-genai 2>/dev/null | awk '/^Name|^Version/ {printf "  %s\n", $0}'

if [ ! -f .env ]; then
    cp .env.example .env
    echo
    echo "created .env from the template."
fi

echo
echo "=============================================================="
if grep -q '^GEMINI_API_KEY=$' .env 2>/dev/null; then
    echo "NEXT: paste your Gemini API key into .env at the repository root"
    echo "      then run:"
else
    echo "NEXT:"
fi
cat <<'EOF'

  source .venv/bin/activate
  python src/freeze.py write        # freeze the harness configuration, once
  python src/verify_setup.py        # confirm every model id against the live API
  python src/run_sweep.py --preset smoke --suites slack --max-user-tasks 2 --reps 1

The smoke run costs a few cents and exercises the whole path. Read README.md
for what to run after it passes.
EOF
echo "=============================================================="
