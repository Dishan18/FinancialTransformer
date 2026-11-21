#!/usr/bin/env bash
# Full experiment chain (sequential; every stage is resumable because finished runs are skipped).
# PY must point to a Python 3.11 interpreter that has the packages of requirements-repro.txt.
# Stages do not abort the chain: on Windows the interpreter can return a non-zero code at CUDA
# teardown after all work is saved, so each stage's exit code is logged instead.
cd "$(dirname "$0")/.."
PY="${PY:-python}"
mkdir -p results/logs
stage () {  # stage <log-name> <args...>
  local log="results/logs/$1.log"; shift
  "$PY" -u "$@" > "$log" 2>&1
  echo "$(date +%H:%M:%S) $* -> exit $?"
}
[ -f results/hpo/best_composite.json ] || stage hpo_composite scripts/run_hpo.py composite
stage aapl_base       scripts/run_aapl.py base
stage aapl_ablation   scripts/run_aapl.py ablation
[ -f results/hpo/best_mse.json ] || stage hpo_mse scripts/run_hpo.py mse
stage aapl_hpo        scripts/run_aapl.py hpo
stage multiasset      scripts/run_multiasset.py
stage hpo_transfer    scripts/hpo_transfer.py composite
stage analyze         scripts/analyze.py
stage diagnostics     scripts/diagnostics.py
stage analyze         scripts/analyze.py
echo "ALL DONE"
