#!/bin/bash
# Syntax check all Python files
# Usage: bash scripts/local_checks/syntax_check.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Checking Python syntax..."
errors=0

for f in $(find "$PROJECT_ROOT/research" "$PROJECT_ROOT/tests" "$PROJECT_ROOT/scripts" -name "*.py" 2>/dev/null); do
    if ! python3 -m py_compile "$f" 2>/dev/null; then
        echo "  FAIL: $f"
        errors=$((errors + 1))
    fi
done

if [ $errors -eq 0 ]; then
    echo "  All files passed syntax check."
else
    echo "  $errors file(s) have syntax errors!"
    exit 1
fi
