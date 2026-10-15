# Where a served teacher publishes itself, and which port it listens on -- ONE derivation,
# used by both the server that writes the file and every client that reads it.
#
# WHY THIS FILE EXISTS. teacher_server.sbatch published to a single
# $OPD_RUNSTATE_DIR/opd_teacher_host.txt and listened on a hardcoded 8100. That was correct
# while exactly one teacher existed. The three-teacher comparison runs MiroThinker-1.7-mini,
# Qwen3-30B-A3B-Thinking and Qwen3-32B as three concurrent services, and then:
#
#   - one host file means the last server to start owns it, so which teacher a precompute
#     job scores against is decided by start order;
#   - one port means two teachers scheduled onto the same node (each asks 1 GPU, the nodes
#     have 4) collide at bind, after the queue wait and after weight loading.
#
# The first one does NOT silently corrupt anything: TeacherHTTPScorer._verify_backend GETs
# /health and compares tokenizer_name, which is not an ENGINE_RUNTIME_FIELD and so is really
# compared, and cfg["model"]["teacher"] is ${OPD_TEACHER_MODEL} -- so a client pointed at the
# wrong teacher raises before it scores a single trajectory. That check stays; this file just
# stops the collision from happening in the first place, because failing at hour three of a
# queued job is not the same as not colliding.
#
# BOTH SIDES DERIVE FROM $OPD_TEACHER_MODEL. That is the point: the server and the client
# cannot disagree about the filename, because neither of them chooses it. If they somehow
# ran with different OPD_TEACHER_MODEL values, the client waits on a file that was never
# written and times out loudly, instead of reading a file that someone else wrote.
#
# Both are overridable ($OPD_TEACHER_HOST_FILE, $OPD_TEACHER_PORT) for one-off manual runs.

opd_teacher_host_file() {
  if [ -n "${OPD_TEACHER_HOST_FILE:-}" ]; then
    printf '%s\n' "$OPD_TEACHER_HOST_FILE"
    return 0
  fi
  if [ -z "${OPD_TEACHER_MODEL:-}" ]; then
    echo "FATAL: OPD_TEACHER_MODEL is unset, so the teacher host file cannot be derived." >&2
    echo "       Export it (it is also what configs/opd_qwen3.yaml model.teacher reads), or" >&2
    echo "       set OPD_TEACHER_HOST_FILE explicitly." >&2
    return 1
  fi
  printf '%s/opd_teacher_host_%s.txt\n' \
    "${OPD_RUNSTATE_DIR}" "$(basename "$OPD_TEACHER_MODEL")"
}

opd_teacher_port() {
  if [ -n "${OPD_TEACHER_PORT:-}" ]; then
    printf '%s\n' "$OPD_TEACHER_PORT"
    return 0
  fi
  if [ -z "${OPD_TEACHER_MODEL:-}" ]; then
    echo "FATAL: OPD_TEACHER_MODEL is unset, so the teacher port cannot be derived." >&2
    return 1
  fi
  # CRC32 of the checkpoint's directory name, into [8100, 8189]. Deterministic (cksum is a
  # fixed CRC, not a hash with a per-process seed), so a restarted server lands on the same
  # port and a stale host file stays consistent with the process it names. The three teachers
  # in this study map to 8109 / 8140 / 8135 -- verified distinct, and asserted in
  # tests/test_teacher_endpoint_derivation.py rather than left to chance.
  #
  # The CLIENT never needs this: it reads the full URL out of the host file. Only the server
  # picks a port, so a collision here is a bind error on one node, not a wrong endpoint.
  local crc
  crc=$(printf '%s' "$(basename "$OPD_TEACHER_MODEL")" | cksum | cut -d' ' -f1)
  printf '%s\n' "$((8100 + crc % 90))"
}


# --- service readiness gates -----------------------------------------------------------
# The long-lived services publish their endpoint to a scratch file at job START, but only
# become usable MINUTES later (retriever: 71 GB HNSW index; teacher: 30B MoE + warmup). A
# consumer submitted right after therefore reads an endpoint nothing is listening on yet
# and dies with ConnectionRefused -- and if the service was just restarted onto a DIFFERENT
# node, the file may still hold the PREVIOUS node until the new job writes it. So both
# helpers RE-READ the host file every poll instead of caching it once, and probe with a
# real request (proving the index/model finished loading, not merely that a port is open).
# They echo the resolved endpoint so the job log records which instance it actually used.

opd_pid_still_alive() {                  # $1 = pid, or "" when there is nothing to watch
  # "keep waiting" is the answer when no pid was given -- an EXTERNAL teacher job is not this
  # process's child and cannot be probed this way, and that is the historical caller.
  [ -z "${1:-}" ] && return 0
  kill -0 "$1" 2>/dev/null
}


wait_for_teacher() {                     # $1 = host file, $2 = timeout, $3 = pid to watch (optional)
  # DEFAULT IS DERIVED, not the old shared path: with three teachers served at once, a
  # fixed filename means the last server to start owns it. opd_teacher_host_file() derives
  # the same name the server wrote, from the same $OPD_TEACHER_MODEL. An explicit $1 still
  # wins -- start_private_teacher passes a job-scoped file.
  local file="${1:-$(opd_teacher_host_file)}" timeout="${2:-1800}"
  local waited=0 url health
  echo "[wait] teacher: polling (timeout ${timeout}s)"
  while :; do
    if [ -f "$file" ]; then
      url=$(cat "$file")
      health="${url%/score_logprobs}/health"
      if curl -sf -m 10 "$health" >/dev/null 2>&1; then
        echo "[wait] teacher READY after ${waited}s -> $url"
        return 0
      fi
    fi
    # A DEAD PROCESS IS AN ANSWER, not something to wait out. When $3 names a pid we started,
    # its disappearance ends the wait immediately: on 2026-09-03 the private 32B teacher exited
    # during startup and this loop polled a corpse for 2400 s, burning 40 minutes of a two-A100
    # allocation before the caller's own kill -0 check reported what had been true all along.
    # No pid (the external-teacher callers) keeps the old behaviour exactly.
    if ! opd_pid_still_alive "${3:-}"; then
      echo "[wait] FATAL: the teacher PROCESS (pid ${3}) EXITED after ${waited}s -- it never came up."
      echo "[wait] Its traceback is in THIS job's log, above this line (the server prints it)."
      return 1
    fi
    if [ "$waited" -ge "$timeout" ]; then
      echo "[wait] FATAL: teacher not ready within ${timeout}s (last endpoint: ${url:-<no host file>})."
      echo "[wait] Check its job is RUNNING (squeue -u \$USER -n opd-teacher) and see slurm/logs/teacher-*.out."
      return 1
    fi
    sleep 10; waited=$((waited + 10))
  done
}

# --- private, job-scoped teacher ---------------------------------------------------------
# Starts a teacher INSIDE the calling job's allocation on card $1, waits for it, and leaves
# $TEACHER_ENDPOINT set. The caller must `trap stop_private_teacher EXIT INT TERM`.
#
# It publishes to a JOB-SCOPED file, never the shared opd_teacher_host.txt: writing that one
# would point every other job at a teacher that dies when this job ends -- an endpoint that
# answers for a while and then does not.
#
# Factored here rather than copied per sbatch because it carries a trap and two distinct
# failure branches, and duplicated lifecycles drift.
start_private_teacher() {                # $1 = CUDA device index, $2 = model path, $3 = dtype
  local dev="${1:-0}" model="$2" dtype="${3:-bfloat16}"
  PRIV_TEACHER_PORT=$((8100 + ${SLURM_JOB_ID:-0} % 200))
  PRIV_TEACHER_HOSTFILE="slurm/logs/teacher-sc-${SLURM_JOB_ID:-manual}.host"
  mkdir -p slurm/logs
  echo "http://127.0.0.1:${PRIV_TEACHER_PORT}/score_logprobs" > "$PRIV_TEACHER_HOSTFILE"
  echo "[teacher] private endpoint: $(cat "$PRIV_TEACHER_HOSTFILE") (job ${SLURM_JOB_ID:-manual})"

  # OPD_TEACHER_MAX_LOGPROBS: only a /score_head consumer (delta-OPD step 2) sets it; the
  # scoring path never needs it. Bound with :- before :+ (site test: a bare OPD_* is site vocabulary).
  local max_lp="${OPD_TEACHER_MAX_LOGPROBS:-}" skip_tok="${OPD_TEACHER_SKIP_TOKENIZER:-}"
  CUDA_VISIBLE_DEVICES="$dev" python slurm/teacher_server.py     --model "$model" --dtype "$dtype" --host 127.0.0.1 --port "$PRIV_TEACHER_PORT" \
      ${max_lp:+--max-logprobs "$max_lp"} ${skip_tok:+--skip-tokenizer-init} &
  PRIV_TEACHER_PID=$!

  # Two failure modes, distinguished: the process DIED (its traceback is in THIS log) versus
  # it is merely slow (30B MoE load + worst-case startup probe). "not ready" alone sends you
  # looking in the wrong place.
  if ! wait_for_teacher "$PRIV_TEACHER_HOSTFILE" "${TEACHER_WAIT_TIMEOUT:-2400}" "$PRIV_TEACHER_PID"; then
    if kill -0 "$PRIV_TEACHER_PID" 2>/dev/null; then
      echo "[teacher] FATAL: alive but never answered /health in ${TEACHER_WAIT_TIMEOUT:-2400}s." >&2
    else
      echo "[teacher] FATAL: the PROCESS EXITED during startup -- its traceback is above in" >&2
      echo "[teacher] THIS log (the server prints it), not a separate teacher-*.out." >&2
    fi
    return 1
  fi
  TEACHER_ENDPOINT=$(cat "$PRIV_TEACHER_HOSTFILE")
}

stop_private_teacher() {
  local rc=$?
  if [ -n "${PRIV_TEACHER_PID:-}" ] && kill -0 "$PRIV_TEACHER_PID" 2>/dev/null; then
    echo "[teacher] stopping private teacher (pid $PRIV_TEACHER_PID)"
    kill "$PRIV_TEACHER_PID" 2>/dev/null || true
    wait "$PRIV_TEACHER_PID" 2>/dev/null || true
  fi
  rm -f "${PRIV_TEACHER_HOSTFILE:-/nonexistent}"
  exit $rc
}

wait_for_retriever() {                   # $1 = host file, $2 = timeout seconds (default 3600)
  local file="${1:-$OPD_RUNSTATE_DIR/opd_retriever_host.txt}" timeout="${2:-3600}"
  local waited=0 url
  echo "[wait] retriever: polling (timeout ${timeout}s)"
  while :; do
    if [ -f "$file" ]; then
      url=$(cat "$file")
      # a real POST proves the index finished loading; an open port does not.
      if curl -sf -m 15 -X POST "$url" -H 'Content-Type: application/json' \
           -d '{"queries":["capital of France"],"topk":1,"return_scores":true}' >/dev/null 2>&1; then
        echo "[wait] retriever READY after ${waited}s -> $url"
        return 0
      fi
    fi
    if [ "$waited" -ge "$timeout" ]; then
      echo "[wait] FATAL: retriever not ready within ${timeout}s (last endpoint: ${url:-<no host file>})."
      echo "[wait] Check its job is RUNNING (squeue -u \$USER) and see slurm/logs/retriever-wiki18-*.out."
      return 1
    fi
    sleep 10; waited=$((waited + 10))
  done
}
