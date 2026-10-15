#!/usr/bin/env python3
"""MiroThinker teacher, served by vLLM as an independent HTTP PROCESS (HPC only).

WHY A SERVED PROCESS (not in-process vLLM):
  The teacher must be numerically isolated from the policy's vLLM engine. Two live
  vLLM V1 engines in ONE process interfere via CUDA graphs / global state, so the
  ONLY clean isolation is a separate PROCESS on its OWN card (own CUDA_VISIBLE_DEVICES)
  -- exactly the retriever pattern. Serving also decouples the offline precompute into
  "call the service, store to disk", so online and offline hit the SAME backend by
  construction (gate 1).

THE PURE-ID CONTRACT (load-bearing, mirrors the retriever contract shape):
    POST /score_logprobs  {"input_ids": [int,...], "score_positions": [int,...]?}
      -> {"logprobs": [null, float, ...], "n_input": N, "n_prompt_logprobs": M}
    GET  /health          -> {"status":"ok", "backend": {...fingerprint...}}

  input_ids are fed to vLLM VERBATIM (TokensPrompt) -- the server NEVER tokenizes.
  Re-tokenizing here would be a detok->retok at the service boundary: the server's
  re-encoded ids could differ from the client's frozen ids (Qwen3 boundary-merge
  hazard), silently desyncing teacher from student. Pure-id transport extends the
  "never detok->retok" invariant across the HTTP boundary; it is a hard requirement.

DETERMINISM (gate 3a "eliminate the source"):
  This process PINS the batch-composition source of non-determinism so the SAME token
  gets the SAME teacher logprob regardless of batch-mates:
    - VLLM_BATCH_INVARIANT=1  (forced, not defaulted -- an operator override to 0 would
      silently break gate 3a; the determinism probe MUST run against THIS server)
    - enforce_eager=True      (no CUDA-graph variance)
    - enable_prefix_caching=False
  A single served instance also auto-pins same-HW / same-version / same-TP for both
  online and offline. Whether prompt_logprobs is bit-invariant under the flag is the
  UNVERIFIED third-party behavior the HPC determinism probe exists to settle.

Two server-side guards make the contract self-checking:
  (BOS guard) assert len(prompt_logprobs) == len(input_ids). If vLLM prepends a BOS,
    the returned array is length N+1, every index shifts, and align-to-t collapses.
    We refuse (400) rather than silently mis-align -- the reader trusts the length.
  (span guard) if score_positions is given, assert each is a scoreable position
    (1 <= p < N, non-null logprob). This is the cross-service form of the M2 masking-
    parity check: it catches a client whose score span has drifted from its ACTION
    definition before the bad request is silently scored.

Run (see slurm/teacher_server.sbatch, added in Step 3):
    CUDA_VISIBLE_DEVICES=0 python slurm/teacher_server.py \
      --model /path/to/scratch/.cache/models/MiroThinker-1.7-mini \
      --dtype bfloat16 --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

# repo root on path so the server reuses the ONE alignment convention + fingerprint.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from teacher.engine import (  # noqa: E402  (after the sys.path insert above)
    TEACHER_MAX_MODEL_LEN,
    checkpoint_footprint,
    config_summary,
    pin_determinism_env,
    resolve_teacher_utilization,
    teacher_engine_kwargs,
)


class ScoreError(ValueError):
    """A malformed / mis-aligned scoring request -> HTTP 400 (not a 500 crash)."""


def _aligned_logprobs(prompt_logprobs, input_ids, score_positions):
    """vLLM prompt_logprobs -> JSON-ready align-to-t list, with the two guards.

    Reuses opd.masking.vllm_prompt_logprobs_to_aligned as the SINGLE alignment
    source (its length + [0]-is-None asserts ARE the BOS guard). Returns a plain
    list where position 0 is null (no prefix) and every other entry is a float.
    """
    import torch

    from opd.masking import vllm_prompt_logprobs_to_aligned

    n = len(input_ids)
    m = len(prompt_logprobs)
    if m != n:
        # BOS guard: a length mismatch means vLLM injected/dropped a prefix token.
        raise ScoreError(
            f"prompt_logprobs length {m} != input length {n} -- vLLM likely prepended "
            "a BOS/prefix token; align-to-t would be off by one. Refusing."
        )
    ids_t = torch.tensor(input_ids, dtype=torch.long)
    try:
        aligned = vllm_prompt_logprobs_to_aligned(prompt_logprobs, ids_t)  # asserts [0] None
    except AssertionError as exc:  # convert alignment asserts into a clean 400
        raise ScoreError(f"alignment failed: {exc}") from exc

    if score_positions is not None:
        for p in score_positions:
            if not (isinstance(p, int) and 1 <= p < n):
                raise ScoreError(
                    f"score position {p!r} out of range [1,{n}) -- a position-0 or "
                    "out-of-bounds ACTION token means the client's score span has "
                    "drifted from its token_type (cross-service masking-parity guard)."
                )
            if not bool(torch.isfinite(aligned[p])):
                raise ScoreError(f"score position {p} has no finite logprob")

    # nan (position 0) -> None; JSON has no NaN. The client rebuilds nan on read.
    return [None if not bool(torch.isfinite(x)) else float(x) for x in aligned]


def make_handler(scorer_fn, backend_dict):
    # ONE SCORING CALL AT A TIME. ThreadingHTTPServer gives every request its own thread, and
    # they all reach the SAME in-process vLLM engine. LLM.generate() is not safe under that:
    # each call drains whatever outputs the engine's step() produces, so two concurrent calls
    # steal each other's results. Observed 2026-08-17, the first time TWO clients existed --
    # precompute and the online arm started together and died in the same second, one
    # receiving 4 outputs for its 3 sequences and the other 2 for its 3. The pairing guard
    # below refused, which is why crossed logprobs went into nobody's cache.
    #
    # The lock also preserves what gate 2 certified: batch-equivalence was measured for one
    # client's batch scored as ONE generate call. Interleaving two clients' requests inside a
    # single engine step is a batch composition no gate has ever seen. Threading still buys
    # concurrency for everything that does not touch the engine (health checks, request
    # parsing, response writing); the engine itself was always serial -- one GPU, one queue.
    _score_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("/health", "/healthz"):
                # publish the backend fingerprint so the client can enforce gate 1
                # (same template / dtype / tokenizer) ACROSS the service boundary.
                self._send(200, {"status": "ok", "backend": backend_dict})
            else:
                self._send(404, {"error": "GET /health only"})

        def do_POST(self):  # noqa: N802
            route = self.path.rstrip("/")
            if route not in ("/score_logprobs", "/score_head"):
                self._send(404, {"error": "POST /score_logprobs or /score_head only"})
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
                if route == "/score_head":
                    # THE TEACHER HEAD (delta-OPD step 2): for each requested position the
                    # top-K_q (cover/cap chosen by the client) and log q at the client's extra
                    # ids, from ONE prompt_logprobs=k_wide pass. Same lock, same pairing guard.
                    items = req.get("batch")
                    if not isinstance(items, list) or not items:
                        raise ScoreError("batch must be a non-empty list")
                    parsed = []
                    for it in items:
                        ids, pos, ex = it.get("input_ids"), it.get("positions"), it.get("extra_ids")
                        if not isinstance(ids, list) or not ids or not isinstance(pos, list) or not isinstance(ex, list) \
                                or len(ex) != len(pos):
                            raise ScoreError("each batch item needs input_ids, positions and extra_ids (one list per position)")
                        parsed.append((ids, pos, ex))
                    head_fn = getattr(scorer_fn, "head", None)
                    if head_fn is None:
                        raise ScoreError("this server has no head scorer")
                    with _score_lock:
                        scored = head_fn(parsed, int(req.get("k_wide", 128)), int(req.get("k_cap", 32)), float(req.get("cover", 0.99)))
                    self._send(200, {"results": scored})
                    return
                if "batch" in req:
                    items = req["batch"]
                    if not isinstance(items, list) or not items:
                        raise ScoreError("batch must be a non-empty list")
                    parsed = []
                    for it in items:
                        ids = it.get("input_ids")
                        if not isinstance(ids, list) or not ids:
                            raise ScoreError("each batch item needs a non-empty input_ids")
                        parsed.append((ids, it.get("score_positions")))
                    with _score_lock:
                        scored = getattr(scorer_fn, "batch")(parsed)
                    self._send(200, {"results": [
                        {"logprobs": lp, "n_input": len(ids), "n_prompt_logprobs": m}
                        for (ids, _sp), (lp, m) in zip(parsed, scored)]})
                    return
                input_ids = req.get("input_ids")
                if not isinstance(input_ids, list) or not input_ids:
                    raise ScoreError("input_ids must be a non-empty list of ints")
                score_positions = req.get("score_positions")  # optional
                with _score_lock:
                    logprobs, m = scorer_fn(input_ids, score_positions)
                self._send(200, {"logprobs": logprobs, "n_input": len(input_ids),
                                 "n_prompt_logprobs": m})
            except ScoreError as exc:
                print(f"[teacher_server] 400 ScoreError: {exc}", file=sys.stderr, flush=True)
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 -- surface, don't crash the server
                # ALSO print the traceback: the 500 body is the only other copy of this
                # reason, and a client that does not read the error body off its HTTPError
                # would otherwise lose the cause entirely (it happened -- a precompute died
                # with a bare "HTTP Error 500" and no diagnosable trace on either side).
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                self._send(500, {"error": repr(exc)})

        def log_message(self, *args):  # noqa: ARG002 -- silence per-request stderr spam
            pass

    return Handler


def _build_scorer(model: str, engine_kwargs: dict):
    """Load MiroThinker under vLLM with the determinism pins, return a scorer_fn
    (input_ids, score_positions) -> (aligned_logprobs_list, n_prompt_logprobs).

    Takes ALREADY-RESOLVED engine kwargs (from teacher/engine.py, the same definition
    scripts/determinism_probe.py builds with) rather than resolving them itself, so main()
    can publish the EXACT dict the engine was built from. Resolving in two places is how
    /health came to advertise max_num_batched_tokens=null while the engine ran 8192."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    max_model_len = engine_kwargs["max_model_len"]
    llm = LLM(model=model, **engine_kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    # We score a PROMPT of length N and still sample 1 token (prompt_logprobs needs a
    # generate call), and vLLM counts prompt + sampled against max_model_len -- so the
    # longest SCOREABLE trajectory is max_model_len - 1, NOT max_model_len. If the rollout
    # budget (config rollout.max_len) is set equal to this server's --max-model-len, the
    # longest legal trajectory is exactly one token too long to score, and the failure is
    # data-dependent: it stays invisible until some question finally produces a
    # cap-length trajectory. Check it explicitly and say so, instead of letting vLLM raise
    # into the 500 catch-all with no numbers attached.
    budget = max_model_len - 1

    def scorer_batch_fn(items):
        """Score a LIST of (input_ids, score_positions) in ONE generate call.

        This is THE scoring path; the single-sequence entry point below is this at size
        one, so batched and unbatched requests cannot drift apart in parsing or guards.

        Submitting many sequences does NOT raise the memory peak: the prompt_logprobs
        log_softmax is materialised per PREFILL STEP, and max_num_batched_tokens caps a
        step at 8192 tokens (4.64 GiB), independent of how many sequences are in flight or
        how long they are. Batch composition and chunk boundaries were both certified
        numerically inert by the determinism probe, so batching here changes throughput
        only.
        """
        for input_ids, _sp in items:
            if len(input_ids) > budget:
                raise ScoreError(
                    f"trajectory length {len(input_ids)} exceeds this teacher's scoreable "
                    f"budget {budget} (= --max-model-len {max_model_len} minus the 1 sampled "
                    "token). Raise --max-model-len ABOVE the rollout's max_len (they must not "
                    "be equal), or lower rollout.max_len. Refusing rather than scoring a "
                    "truncated prompt."
                )
        outs = llm.generate([TokensPrompt(prompt_token_ids=list(ids)) for ids, _ in items],
                            params)
        # vLLM preserves request order, but results are consumed POSITIONALLY -- a reorder
        # would hand trajectory A the logprobs of B, silently. Refuse a count mismatch for
        # the same reason the retriever client does.
        if len(outs) != len(items):
            raise ScoreError(
                f"engine returned {len(outs)} outputs for {len(items)} scored sequences -- "
                "results are paired to requests by POSITION, so this would cross logprobs "
                "between trajectories. Refusing.")
        results = []
        for (input_ids, score_positions), out in zip(items, outs):
            pl = out.prompt_logprobs
            results.append((_aligned_logprobs(pl, input_ids, score_positions), len(pl)))
        return results

    def scorer_fn(input_ids, score_positions):
        return scorer_batch_fn([(input_ids, score_positions)])[0]

    def head_batch_fn(items, k_wide, k_cap, cover):
        """(input_ids, positions, extra_ids per position) x batch -> per item {"heads": [...], "n_input"}.
        One generate call with prompt_logprobs=k_wide: the engine hands back the top-k_wide
        tokens (+ the realised one) at EVERY prompt position; teacher.head.head_from_entry cuts
        K_q and looks the extras up in that list. The values are the same log_softmax the
        scoring path reads -- k_wide only widens what is returned -- and the client's alignment
        gate (lp_y here vs the cached teacher_logp) is what certifies that, per line."""
        from teacher.head import head_from_entry
        params_head = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=int(k_wide))
        for input_ids, _p, _e in items:
            if len(input_ids) > budget:
                raise ScoreError(f"trajectory length {len(input_ids)} exceeds this teacher's scoreable budget {budget}")
        outs = llm.generate([TokensPrompt(prompt_token_ids=list(ids)) for ids, _, _ in items], params_head)
        if len(outs) != len(items):
            raise ScoreError(f"engine returned {len(outs)} outputs for {len(items)} sequences -- positional pairing; refusing")
        results = []
        for (input_ids, positions, extras), out in zip(items, outs):
            pl = out.prompt_logprobs
            if len(pl) != len(input_ids):
                raise ScoreError(f"prompt_logprobs length {len(pl)} != input length {len(input_ids)}")
            heads = []
            for p, ex in zip(positions, extras):
                p = int(p)
                if not (1 <= p < len(input_ids)) or pl[p] is None:
                    raise ScoreError(f"head position {p} outside [1, {len(input_ids)})")
                heads.append(head_from_entry(pl[p], int(input_ids[p]), ex, k_cap, cover))
            results.append({"heads": heads, "n_input": len(input_ids)})
        return results

    scorer_fn.batch = scorer_batch_fn
    scorer_fn.head = head_batch_fn
    return scorer_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="MiroThinker teacher -> /score_logprobs HTTP server")
    ap.add_argument("--model", required=True, help="LOCAL MiroThinker path (offline node)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 so other nodes can reach it")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="Chunked-prefill token budget. None = let vLLM derive it from "
                         "max-model-len. It is RECORDED in the backend fingerprint because "
                         "it decides chunk boundaries, and VLLM_BATCH_INVARIANT pins batch "
                         "COMPOSITION, not chunking -- the determinism probe covered "
                         "co-batching within one prefill step, not a chunk boundary landing "
                         "inside a scored sequence. Changing it mid-experiment could shift "
                         "every logprob, so the fingerprint now refuses that silently.")
    ap.add_argument("--gpu-memory-utilization", type=float,
                    default=None,
                    help="DEFAULT RESOLVES FROM THE CHANNEL (teacher.engine."
                         "resolve_teacher_utilization): $OPD_TEACHER_UTIL, else "
                         "configs/channels/$OPD_CHANNEL.yaml. It used to default to the "
                         "module constant 0.72, which is the H20 value -- on an 80 GiB card "
                         "that is less than the teacher's weights, so the engine never "
                         "started and certify gate 2 misreported it as a batch-equivalence "
                         "failure. Pass a value explicitly only when PROBING for one. "
                         "Deliberately LOW. prompt_logprobs makes vLLM materialise a "
                         "[n_prompt, vocab] fp32 tensor in sampler.compute_logprobs "
                         "(logits.log_softmax(dtype=float32)) -- a TRANSIENT allocation that "
                         "vLLM's memory profiling does NOT reserve for, because the KV pool is "
                         "sized to fill gpu_memory_utilization. At 0.90 the engine kept only "
                         "6.4 GiB free and died with CUDA OOM trying to allocate 7.78 GiB while "
                         "scoring a 13750-token trajectory. Budget: n_prompt * 151936 (Qwen3 "
                         "vocab) * 4 B, i.e. ~9.3 GiB at rollout.max_len 16384, and the logits "
                         "input is a second tensor of the same shape -- so leave ~20 GiB free. "
                         "Lowering this only shrinks the KV pool (fewer CONCURRENT sequences); "
                         "it does NOT change any computed value, unlike chunking or TP.")
    ap.add_argument("--max-logprobs", type=int, default=None,
                    help="vLLM's cap on prompt_logprobs=k (engine default 20). /score_head asks for k_wide "
                         "(128-256) per position, so a server meant to serve the teacher head must be started "
                         "with this >= k_wide (OPD_TEACHER_MAX_LOGPROBS in lib_teacher_endpoint.sh). It changes "
                         "what is RETURNED, not what is computed: the scoring path's numbers are unaffected, "
                         "and the client's alignment gate re-checks that on every line.")
    ap.add_argument("--skip-tokenizer-init", action="store_true",
                    help="engine kwarg skip_tokenizer_init=True. The scoring paths take token ids in and hand "
                         "log-probs out; the only thing the tokenizer does here is DETOKENIZE every returned "
                         "logprob entry (vLLM V1 fills Logprob.decoded_token per entry), which at "
                         "prompt_logprobs=256 is 256 string conversions per prompt position -- the 2026-09-05 "
                         "head smoke ran at 443 prompt tok/s with it. Numerically inert by construction (no "
                         "kernel changes); the head client's alignment gate re-checks lp_y against the cache.")
    ap.add_argument("--max-model-len", type=int, default=TEACHER_MAX_MODEL_LEN,
                    help="prompt+response ceiling. Must be STRICTLY GREATER than the rollout's "
                         "max_len (configs/opd_qwen3.yaml rollout.max_len, currently 16384): a "
                         "cap-length trajectory needs max_len+1 slots here because scoring also "
                         "samples 1 token. The original default was EQUAL to it, which made a "
                         "cap-length trajectory unscoreable and only failed once the data "
                         "happened to produce one. Kept just 1024 above max_len on purpose: the "
                         "startup probe scores max_model_len-1 tokens, so an inflated value "
                         "would demand worst-case log_softmax memory for prompts the rollout can "
                         "never actually produce. Raise this and rollout.max_len TOGETHER.")
    args = ap.parse_args()

    # FORCE the determinism source off before importing vllm. setdefault would let a
    # stale env VLLM_BATCH_INVARIANT=0 silently kill gate 3a -- pin it.
    pin_determinism_env()

    from dataclasses import asdict

    from teacher.backend import BackendConfig, runtime_fingerprint

    # Publish the engine runtime too: these are the settings that can change the NUMBERS
    # (chunk budget in particular) yet were invisible to gate 1, so a teacher restarted
    # under different settings could silently shift every logprob. The client adopts them
    # from /health and stamps them onto the precompute cache, which is where
    # assert_same_backend can then catch the drift.
    # Resolve the engine configuration ONCE, then both BUILD from it and PUBLISH it. The
    # fingerprint must describe what the engine ACTUALLY runs, not what the CLI happened to
    # be given: --max-num-batched-tokens defaults to None while the shared default is 8192,
    # so reading the arg advertised a null the engine never used -- a fingerprint that
    # misreports its configuration is worse than none, because it would later flag a false
    # drift (null -> 8192) and could never say what a cache was really scored under.
    engine_kwargs = teacher_engine_kwargs(
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=resolve_teacher_utilization(args.gpu_memory_utilization),
        tensor_parallel_size=args.tensor_parallel_size,
        **({} if args.max_num_batched_tokens is None
           else {"max_num_batched_tokens": args.max_num_batched_tokens}),
        **({} if args.max_logprobs is None else {"max_logprobs": args.max_logprobs}),
        **({"skip_tokenizer_init": True} if args.skip_tokenizer_init else {}),
    )
    # SIZING PREFLIGHT -- refuse BEFORE 61 GiB of weights are read, not after.
    #
    # Job 27001269 (2026-09-03): this server was pointed at Qwen3-32B and inherited the channel
    # value 0.80, which configs/channels/a100.yaml certified by MEASURING it against
    # MiroThinker's 56.93 GiB of weights. Qwen3-32B is 61.03 GiB; the 4.1 GiB difference IS the
    # KV pool that certification left, so vLLM loaded for 64 s, found 0.79 GiB of KV against the
    # 4.25 GiB one max-length request needs, and died. The information needed to say that was on
    # disk before anything loaded.
    #
    # WHY THIS REFUSES RATHER THAN RAISING THE VALUE ITSELF. gpu_memory_utilization is a
    # BackendConfig ENGINE_RUNTIME_FIELD: a cache scored at one value and an arm run at another
    # have different fingerprints and gate 1 refuses the pairing. resolve_generation_utilization
    # exists for the SAMPLING path and deliberately does not touch this one. So the operator
    # sets $OPD_TEACHER_UTIL, and this prints the exact value to set.
    #
    # NO FALSE REFUSALS FROM THE ESTIMATE'S KNOWN BIAS: device_total_gib reports nvidia-smi's
    # BOARD total (80.00 GiB), larger than what CUDA addresses (79.25), so required_utilization
    # is UNDER-stated. An understated requirement that still exceeds the resolved value means
    # the true requirement exceeds it by more. The bias can only hide a real problem, never
    # invent one.
    try:
        fp = checkpoint_footprint(args.model, engine_kwargs["max_model_len"])
    except Exception as exc:                      # unreadable layout: say so and let vLLM try
        print(f"[teacher_server] could not size {args.model} ({exc}); starting unchecked", flush=True)
        fp = None
    if fp and fp["required_utilization"] is not None:
        print(f"[teacher_server] sizing {os.path.basename(args.model.rstrip('/'))}: weights "
              f"{fp['weights_gib']:.2f} + overhead {fp['overhead_gib']:.2f} + KV "
              f"{fp['kv_gib_per_request']:.2f} (1 x {fp['max_model_len']} tok) = "
              f"{fp['required_gib_one_request']:.2f} of {fp['device_total_gib']:.2f} GiB "
              f"-> needs util >= {fp['required_utilization']:.3f}, running at "
              f"{engine_kwargs['gpu_memory_utilization']}", flush=True)
        if fp["required_utilization"] > engine_kwargs["gpu_memory_utilization"]:
            raise SystemExit(
                f"FATAL: {args.model} cannot start at gpu_memory_utilization="
                f"{engine_kwargs['gpu_memory_utilization']} -- it needs at least "
                f"{fp['required_utilization']:.3f} to hold ONE {fp['max_model_len']}-token "
                f"request. The channel value was certified against a smaller checkpoint. "
                f"Re-run with OPD_TEACHER_UTIL set (the three lines use 0.80 / 0.80 / 0.858 "
                f"for miro / moe / dense -- README.md), and remember it is a "
                f"BackendConfig engine-runtime field: a cache scored under a different value "
                f"will not pair with this one.")

    cfg = BackendConfig(engine="vllm_http", dtype=engine_kwargs["dtype"],
                        tokenizer_name=args.model,
                        max_model_len=engine_kwargs["max_model_len"],
                        max_num_batched_tokens=engine_kwargs["max_num_batched_tokens"],
                        gpu_memory_utilization=engine_kwargs["gpu_memory_utilization"],
                        # WHICH MACHINE. Everything above is identical across execution
                        # channels (same model, same template, same vLLM settings), so
                        # without these an A800-scored cache and an H20-scored one carry the
                        # same fingerprint and gate 1 accepts either -- while their teacher
                        # logprobs are NOT the same numbers. The client adopts these from
                        # /health and stamps them onto whatever it writes.
                        **runtime_fingerprint())
    # publish the FULL config (incl. thinking flags + template_fp) so the client's
    # cross-service gate-1 check compares every field, not a subset.
    backend_dict = asdict(cfg)

    print(f"[teacher_server] loading MiroThinker model={args.model}", flush=True)
    # Log the EXACT built configuration, so a job log always records which settings produced
    # its numbers -- and so /health and the log can be compared without reading the code.
    print(f"[teacher_server] engine: {config_summary(**engine_kwargs)}", flush=True)
    scorer_fn = _build_scorer(args.model, engine_kwargs)
    # fail-fast: prove a WORST-CASE score works NOW, not 13 trajectories into a precompute.
    # The old probe was [1,2,3,4] -- it proved the engine answers, which is exactly the thing
    # that was never in doubt: a fully-loaded, healthy server served 13 trajectories and THEN
    # died of CUDA OOM on a long one, because prompt_logprobs' [n_prompt, vocab] fp32
    # log_softmax scales with PROMPT LENGTH and a 4-token probe never touches that ceiling.
    # So probe at the longest prompt this server claims to accept: if it cannot score that,
    # its --max-model-len is a promise it cannot keep, and we must find out at startup while
    # the only cost is this job, not after a consumer has burned an allocation.
    probe_n = engine_kwargs["max_model_len"] - 1
    print(f"[teacher_server] worst-case probe: scoring a {probe_n}-token prompt "
          f"(peak transient ~{probe_n * 151936 * 4 / 2**30:.1f} GiB for the fp32 log_softmax) ...",
          flush=True)
    probe_logp, m = scorer_fn([1] * probe_n, None)
    print(f"[teacher_server] worst-case probe scored {len(probe_logp)} tokens "
          f"(n_prompt_logprobs={m}, logp[0]={probe_logp[0]}).", flush=True)

    # ---- BATCH-EQUIVALENCE GATE: refuse to serve if batching moves a number -----------
    # Everything downstream assumes a trajectory's teacher logprobs do not depend on what
    # it was scored alongside: offline precomputes in whatever batches the client forms,
    # online scores live in different ones, and gate 1 declares the two identical. The
    # determinism probe established that on this engine, so this is REGRESSION protection,
    # not exploration -- if it trips, the batching implementation is wrong (an ordering or
    # pairing bug), because the numerics question is already settled. Checked HERE, at
    # startup, rather than in a test file: a test proves the code was right once, this
    # proves THIS server process is right before anything depends on it.
    a = [7] * 1024
    b = [9] * 2048                       # different length -> genuinely different tiling
    alone, _ = scorer_fn(a, None)
    batched = scorer_batch_verify = getattr(scorer_fn, "batch")([(b, None), (a, None)])
    co = batched[1][0]                   # `a` sitting SECOND, behind a different-length peer
    if len(co) != len(alone):
        raise SystemExit(f"[teacher_server] FATAL batch gate: length {len(co)} != {len(alone)}")
    diffs = [i for i, (x, y) in enumerate(zip(alone, co))
             if (x is None) != (y is None) or (x is not None and x != y)]
    if diffs:
        worst = max(abs(alone[i] - co[i]) for i in diffs
                    if alone[i] is not None and co[i] is not None)
        raise SystemExit(
            f"[teacher_server] FATAL batch gate: scoring a sequence ALONE vs inside a BATCH "
            f"differs at {len(diffs)} of {len(alone)} positions (max |delta| {worst:.3e}). "
            "Batched and unbatched teacher logprobs must be bit-identical -- otherwise "
            "offline (precomputed in one batching) and online (scored live in another) "
            "silently disagree and gate 1's claim that they share a backend is false. "
            "The determinism probe already certified this engine, so this is an "
            "implementation bug (ordering/pairing), not a numerics surprise. Refusing to serve."
        )
    print(f"[teacher_server] batch-equivalence gate PASS -- {len(alone)} positions "
          "bit-identical alone vs co-batched.", flush=True)
    print("[teacher_server] READY.", flush=True)

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(scorer_fn, backend_dict))
    print(f"[teacher_server] serving POST /score_logprobs on http://{args.host}:{args.port}",
          flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
