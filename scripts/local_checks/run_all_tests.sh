#!/bin/bash
# Run all local tests
# Usage: bash scripts/local_checks/run_all_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================"
echo "Flash-VStream Research — Local Test Suite"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Activate venv if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "Using venv: $(which python3)"
else
    echo "WARNING: No .venv found. Using system Python."
fi

echo ""
echo "--- Syntax Check ---"
bash "$SCRIPT_DIR/syntax_check.sh"

echo ""
echo "--- Import Check ---"
python3 "$SCRIPT_DIR/import_check.py"

echo ""
echo "--- Unit Tests ---"
cd "$PROJECT_ROOT"
python3 -m pytest tests/unit/ -v --tb=short

echo ""
echo "--- Integration Tests ---"
python3 -m pytest tests/integration/ -v --tb=short

echo ""
echo "--- Regression Tests ---"
python3 -m pytest tests/regression/ -v --tb=short

echo ""
echo "============================================"
echo "ALL LOCAL CHECKS PASSED"
echo "============================================"
