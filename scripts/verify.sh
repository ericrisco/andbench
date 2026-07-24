#!/usr/bin/env bash
# Local merge gate — mirrors CI. Run as standalone steps (never piped) so a
# failure in one cannot be masked by a later one.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> uv sync"
uv sync --group dev

echo "==> ruff format --check"
uv run ruff format --check .

echo "==> ruff check"
uv run ruff check .

echo "==> mypy --strict"
uv run mypy

echo "==> pytest (unit) + coverage >= 80%"
uv run pytest -m "not integration" --cov --cov-report=term-missing --cov-fail-under=80

echo "==> pytest (integration)"
set +e
uv run pytest -m integration
code=$?
set -e
if [ "$code" -eq 5 ]; then
  echo "No integration tests collected yet — skipping."
elif [ "$code" -ne 0 ]; then
  exit "$code"
fi

echo "All gates green."
