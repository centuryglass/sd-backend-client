#!/bin/bash
# Starts every Claude Code session by refreshing the GitHub issue cache and printing its index as session context (see
# AGENTS.md, "Tracking open work").
#
# A Claude Code on the web session is also prepared to run the test suite and the lint check: a Python 3.13 virtual
# environment with requirements-dev.txt and the package installed in editable mode. Python 3.13 matches CI's lint
# job, since scripts/pylint_baseline.json is only valid for the Python version it was generated with.
set -euo pipefail

cd "${CLAUDE_PROJECT_DIR:-.}"

# Uses gh when it's on PATH and authenticated, and the REST API otherwise; scripts/issues.py covers tokens and rate
# limits. Either fetch is best-effort, since a rate-limited or failed one must not fail session start. When both fail,
# the hook prints no index, and AGENTS.md, "Tracking open work" says what the agent does then.
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  issue_source=--fetch
else
  issue_source=--fetch-api
fi
python3 scripts/issues.py "$issue_source" && cat .claude/cache/issues/index.md || true

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

python_command=python3.13
if ! command -v "$python_command" >/dev/null 2>&1; then
  python_command=python3
  echo "Python 3.13 isn't installed; using $($python_command --version). scripts/pylint_check.py may disagree" \
       "with its baseline." >&2
fi
if [ ! -x .venv/bin/python ]; then
  "$python_command" -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements-dev.txt -e .

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
