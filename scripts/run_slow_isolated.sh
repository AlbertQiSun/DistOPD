#!/usr/bin/env bash
# Run each @slow test in its OWN pytest process.
#
#   bash scripts/run_slow_isolated.sh [extra pytest args...]
#
# WHY, and why not more teardown code. The @slow policy gates build a vLLM engine, and since
# tests/vllm_engine.py started building it the way production does -- VLLM_ENABLE_V1_MULTI-
# PROCESSING=0, which sync_from requires -- the EngineCore lives in the pytest process itself.
# Nothing is reclaimed by a child exiting, and vLLM's own destroy_model_parallel /
# destroy_distributed_environment do NOT give the memory back either: measured twice on
# <node> and <node>, both times exactly 35% free after a 180s wait, with and without those
# calls. The state is vLLM's module-level globals, not ours, and chasing it further is a
# guess dressed as a fix.
#
# A process boundary returns the memory unconditionally, and it is the same boundary the real
# arms already have: precompute, the teacher server and each training job are separate
# processes with one engine each. So this makes the test harness match the deployment shape
# instead of asking one process to host several engines in sequence.
#
# pytest-forked would do the same thing and is NOT used: it would be a new dependency
# installed over a compute-node network round-trip, to buy what a loop already provides.
set -uo pipefail

DESELECT=(--deselect tests/test_vllm_weight_hot_update.py)   # gate 5 runs that one on its own

mapfile -t IDS < <(python -m pytest -m slow "${DESELECT[@]}" --collect-only -q 2>/dev/null \
                   | grep '::' || true)
if [ "${#IDS[@]}" -eq 0 ]; then
  echo "[slow-isolated] FATAL: collected no @slow tests. A collection error, or the marker" >&2
  echo "[slow-isolated] moved -- either way this would otherwise report success having run" >&2
  echo "[slow-isolated] nothing, which is the failure mode this repo keeps finding." >&2
  exit 1
fi

echo "[slow-isolated] ${#IDS[@]} test(s), one process each"
FAILED=()
for id in "${IDS[@]}"; do
  echo
  echo "---------------------------------------------------------------- $id"
  # -m slow IS REQUIRED even though the id names one test. pytest.ini sets
  # `addopts = -m "not slow"`, and a marker expression from config still applies to an
  # explicitly named test -- so without it every run reports "1 deselected" and exits 5.
  # That first showed up as 7 of 7 "FAILED", which is the correct verdict for a run that
  # executed nothing, just not the informative one.
  python -m pytest -q -m slow "$id" "$@"
  rc=$?
  if [ "$rc" -eq 5 ]; then
    echo "[slow-isolated] NO TESTS RAN for $id (pytest exit 5) -- deselected or missing." >&2
    FAILED+=("$id (no tests ran)")
  elif [ "$rc" -ne 0 ]; then
    FAILED+=("$id")
  fi
done

echo
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "[slow-isolated] all ${#IDS[@]} passed"
  exit 0
fi
echo "[slow-isolated] ${#FAILED[@]} of ${#IDS[@]} FAILED:" >&2
printf '  %s\n' "${FAILED[@]}" >&2
exit 1
