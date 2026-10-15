"""THE served teacher's vLLM engine configuration -- one definition, two importers.

`slurm/teacher_server.py` (what actually scores every trajectory) and
`scripts/determinism_probe.py` (the gate that certifies those scores are batch-invariant)
used to construct their engines INDEPENDENTLY. They had drifted: the probe defaulted to
gpu_memory_utilization 0.90 and derived max_model_len from its own synthetic sequence
lengths, while production runs 0.72 / 17408. A probe that passes under settings production
never runs certifies nothing about production -- the same defect as certifying the weight
hot-update with prefix caching off while the real arm ran with it on.

So the engine kwargs live HERE and both import them. Divergence stops being something to
notice in review and becomes something you have to write code to cause.

max_num_batched_tokens IS pinned here now (it was not, while the probe still had to be
free to choose it in order to force a co-batch at all). The probe PASSED at 8192 -- twice,
the second time with a chunk boundary cutting a scored sequence -- so 8192 became the
certified value, and production must run it rather than vLLM's derivation. max_num_seqs
stays caller-supplied: it caps concurrency, not tiling, and the probe sizes it to its own
batch.
"""
from __future__ import annotations

import os

# Prompt+sampled ceiling. MUST exceed rollout.max_len (configs/opd_qwen3.yaml, 16384):
# scoring samples 1 token on top of the prompt, so a cap-length trajectory needs max_len+1.
TEACHER_MAX_MODEL_LEN = 17408

# Deliberately low. prompt_logprobs materialises a [n_prompt, vocab] fp32 tensor in
# sampler.compute_logprobs, a TRANSIENT allocation vLLM's profiling does not reserve for
# (the KV pool is sized to fill this fraction). At 0.90 the engine kept 6.4 GiB free and
# died with CUDA OOM demanding 7.78 GiB on a 13750-token trajectory.
TEACHER_GPU_MEMORY_UTILIZATION = 0.72

TEACHER_DTYPE = "bfloat16"

# Chunked-prefill token budget, set EXPLICITLY rather than left to vLLM's derivation.
# Two reasons it is pinned to 8192: (1) it is the value the determinism probe PASSED under
# -- twice, including with a chunk boundary cutting a scored sequence -- and running
# production on any other value would put it back on an uncertified configuration, the
# exact defect this project keeps finding; (2) it BOUNDS the prompt_logprobs log_softmax
# transient at budget * 151936 * 4 B = 4.64 GiB regardless of how long a trajectory is or
# how many are submitted at once, which is a stronger guarantee than the
# gpu_memory_utilization headroom alone (that only made the old OOM unlikely; this makes
# the peak independent of the input).
TEACHER_MAX_NUM_BATCHED_TOKENS = 8192


def resolve_teacher_utilization(explicit: float | None = None, *, verbose: bool = True) -> float:
    """The utilization THIS CHANNEL certified, not the one another card needed.

    WHY THIS EXISTS. gate 0.5 searches for a workable gpu_memory_utilization and gate 1 is
    handed it, but the served teacher took the module constant -- 0.72, the H20 value. On an
    80 GiB card that is 57.1 GiB against 56.93 GiB of weights, so the engine dies at KV
    allocation and never serves. Certify gate 2 then reported "the served teacher scores a
    sequence differently alone vs co-batched", which was not what happened at all: the server
    never started. The measured value existed, in the operator's shell and in the channel
    file, and the one process that needed it read neither.

    Resolution order, most specific first, and the source is PRINTED because a silent
    resolution is exactly how the above went unnoticed for a whole gate:

      1. explicit         -- a probe searching for a value must not be told the answer
      2. $OPD_TEACHER_UTIL -- the certify override, needed before the channel file is updated
      3. configs/channels/$OPD_CHANNEL.yaml : teacher.gpu_memory_utilization
      4. TEACHER_GPU_MEMORY_UTILIZATION -- the h20/dry-run default

    This value is a BackendConfig ENGINE_RUNTIME_FIELD, so resolving it differently between
    the precompute and the arms produces mismatched fingerprints and gate 1 refuses the
    pairing. That is the safety net under this function, not a substitute for it.
    """
    def _say(v, src):
        if verbose:
            print(f"[teacher-engine] gpu_memory_utilization={v} (from {src})", flush=True)
        return v

    if explicit is not None:
        return _say(float(explicit), "explicit argument")

    env = os.environ.get("OPD_TEACHER_UTIL", "").strip()
    if env:
        return _say(float(env), "$OPD_TEACHER_UTIL")

    channel = os.environ.get("OPD_CHANNEL", "").strip()
    if channel:
        import pathlib

        import yaml

        path = (pathlib.Path(__file__).resolve().parents[1]
                / "configs" / "channels" / f"{channel}.yaml")
        if path.exists():
            overlay = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            val = (overlay.get("teacher") or {}).get("gpu_memory_utilization")
            if val is not None:
                return _say(float(val), f"configs/channels/{channel}.yaml")
        # A named channel whose file does not answer is a mistake worth refusing, not a
        # reason to fall back to another card's number and fail 70 seconds into a weight load.
        raise ValueError(
            f"OPD_CHANNEL={channel!r} but {path} does not set teacher.gpu_memory_utilization. "
            "Run certify gate 0.5 on this channel and write what it reports into that file; "
            "falling back to another channel's value would start an uncertified engine.")

    return _say(TEACHER_GPU_MEMORY_UTILIZATION, "TEACHER_GPU_MEMORY_UTILIZATION default")


# --- how much of the card THIS checkpoint actually needs ------------------------------------
#
# Non-KV, non-weight resident memory at engine start: CUDA graphs, the compiled backbone,
# activation workspace. TWO measurements, both from this project's own logs on the 79.2 GiB
# A100, both at util 0.80 (budget 63.36 GiB):
#
#   Qwen3-32B, 2026-08-21   63.36 - 61.03 weights - 0.65 KV (vLLM's number)   = 1.68 GiB
#   MiroThinker, gate 0.5   63.36 - 56.93 weights - 5.10 KV (channel comment) = 1.33 GiB
#
# 1.8 sits above both. It is deliberately the conservative end: this constant feeds a REFUSAL
# threshold and the value chosen when the channel's is too small, so erring high costs a little
# KV and erring low costs an engine that dies 70 seconds into a weight load. It is NOT a
# promise that a run will fit -- two points on two models say nothing about a third.
_ENGINE_OVERHEAD_GIB = 1.8

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "half": 2, "float32": 4, "float": 4, "int8": 1,
                "fp8": 1, "float8": 1}


def device_total_gib(index: int = 0) -> float | None:
    """Total memory of the GPU, WITHOUT creating a CUDA context in this process.

    torch.cuda.get_device_properties would answer too, but it initialises CUDA here, and vLLM
    V1 runs EngineCore in a SUBPROCESS that profiles free memory at start-up -- a context held
    by the parent is memory the child then does not see, which would make this function's own
    advice wrong by a few hundred MiB. nvidia-smi asks the driver instead.

    KNOWN AND DELIBERATE OPTIMISM, measured 2026-08-22 on a100-4004: nvidia-smi reports the
    BOARD total (81920 MiB = 80.00 GiB) while CUDA reports what is actually addressable
    (79.25 GiB) -- ECC and driver reservations. So required_utilization is computed against a
    denominator ~0.94% larger than the allocator will see, i.e. it UNDER-states the fraction
    needed by about that much. The 0.02 margin in resolve_generation_utilization absorbs it:
    Qwen3-32B needs 67.07 GiB, 0.858 x 79.25 = 68.0, leaving 0.93 GiB rather than the 1.5 the
    arithmetic implies. Not corrected here because switching denominators would change a
    number that has already run successfully several times, on the strength of arithmetic
    rather than a measurement -- but a checkpoint that fits only by less than 1% of the card
    is fitting by luck, and this is where to look when one does not.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        return float(out.splitlines()[0]) / 1024.0        # MiB -> GiB
    except Exception:
        return None


def checkpoint_footprint(model_dir: str, max_model_len: int) -> dict:
    """What a vLLM engine for this checkpoint needs, read off the checkpoint itself.

    WHY THIS EXISTS. resolve_teacher_utilization returns "the utilization THIS CHANNEL
    certified" -- and configs/channels/a100.yaml certified 0.80 by MEASURING it against
    MiroThinker's 56.93 GiB of weights. Qwen3-32B is 61.03 GiB. The 4.1 GiB difference is
    almost exactly the KV pool that certification left, so the engine refuses to start with
    "estimated maximum model length is 2656" -- and it refuses for a reason that has nothing
    to do with the card and everything to do with the model. Inheriting one model's certified
    number for a different model is the SAME defect the resolver was written to stop, one
    level up: a value that was measured somewhere becoming a value that is assumed everywhere.

    Returns GiB figures and the utilization that would be needed. Everything is read from
    disk -- safetensors file sizes and config.json -- so this runs before anything is loaded
    and costs nothing.
    """
    import glob
    import json
    import os as _os

    idx = _os.path.join(model_dir, "model.safetensors.index.json")
    if _os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            total = (json.load(f).get("metadata") or {}).get("total_size")
        weight_b = float(total) if total else 0.0
    else:
        weight_b = 0.0
    if not weight_b:
        weight_b = float(sum(_os.path.getsize(p) for p in
                             glob.glob(_os.path.join(model_dir, "*.safetensors"))))
    if not weight_b:
        raise FileNotFoundError(f"no *.safetensors under {model_dir}")

    with open(_os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        c = json.load(f)
    c = c.get("text_config", c)
    layers = int(c["num_hidden_layers"])
    kv_heads = int(c.get("num_key_value_heads") or c["num_attention_heads"])
    head_dim = int(c.get("head_dim")
                   or (int(c["hidden_size"]) // int(c["num_attention_heads"])))
    kvb = _DTYPE_BYTES.get(str(c.get("torch_dtype", "bfloat16")).replace("torch.", ""), 2)
    # K and V, per layer, per token.
    kv_b_per_token = layers * kv_heads * head_dim * 2 * kvb

    weights = weight_b / 2 ** 30
    kv_one = kv_b_per_token * max_model_len / 2 ** 30
    total_gib = device_total_gib()
    need_gib = weights + _ENGINE_OVERHEAD_GIB + kv_one
    return {
        "model_dir": model_dir,
        "weights_gib": weights,
        "overhead_gib": _ENGINE_OVERHEAD_GIB,
        "kv_gib_per_request": kv_one,
        "kv_bytes_per_token": kv_b_per_token,
        "max_model_len": max_model_len,
        "device_total_gib": total_gib,
        "required_gib_one_request": need_gib,
        # the utilization at which vLLM can hold exactly ONE max-length request. Below this
        # the engine does not start at all.
        "required_utilization": (need_gib / total_gib) if total_gib else None,
    }


def resolve_generation_utilization(model_dir: str, max_model_len: int, *,
                                   margin: float = 0.02, ceiling: float = 0.94) -> float:
    """gpu_memory_utilization for a SAMPLING-only engine, sized to the model in front of it.

    This is the SFT-generation path, not the scoring path, and the distinction is the whole
    justification for computing rather than inheriting:

      * the scoring teacher's 0.80 is part of the channel's IDENTITY (a BackendConfig
        ENGINE_RUNTIME_FIELD; gate 1 refuses a cache and an arm that disagree). Nothing here
        touches it -- precompute and the arms keep reading resolve_teacher_utilization.
      * SFT generation never computes a logprob. It samples, one request at a time
        (sft/datagen.py uses generate_trajectory, SINGULAR), so the KV pool cannot change
        batch composition and there is no co-batching for it to perturb. That is an argument
        from the call graph, not a measurement: no two-utilization A/B has been run, and this
        sentence is not a claim that one would come out identical.

    An explicit $OPD_TEACHER_UTIL still wins outright -- an operator overriding by hand must
    not be second-guessed by arithmetic. Otherwise: take the channel's value unless the
    checkpoint provably cannot start under it, and print the whole derivation either way,
    because a silent resolution is how the certified-for-another-model value got here.
    """
    env = os.environ.get("OPD_TEACHER_UTIL", "").strip()
    if env:
        print(f"[gen-sft] gpu_memory_utilization={float(env)} ($OPD_TEACHER_UTIL, "
              f"operator override -- sizing check skipped)", flush=True)
        return float(env)

    channel_util = resolve_teacher_utilization()
    try:
        fp = checkpoint_footprint(model_dir, max_model_len)
    except Exception as e:                       # unreadable checkpoint layout: say so, proceed
        print(f"[gen-sft] could not size {model_dir} ({e}); using the channel value "
              f"{channel_util} unchecked", flush=True)
        return channel_util
    if fp["required_utilization"] is None:
        print("[gen-sft] nvidia-smi unavailable; using the channel value "
              f"{channel_util} unchecked", flush=True)
        return channel_util

    print(f"[gen-sft] sizing {os.path.basename(model_dir.rstrip('/'))}: "
          f"weights {fp['weights_gib']:.2f} + overhead {fp['overhead_gib']:.2f} + KV "
          f"{fp['kv_gib_per_request']:.2f} (1 x {max_model_len} tok @ "
          f"{fp['kv_bytes_per_token'] / 1024:.0f} KiB/tok) = "
          f"{fp['required_gib_one_request']:.2f} GiB of {fp['device_total_gib']:.2f} GiB "
          f"-> needs util >= {fp['required_utilization']:.3f}", flush=True)

    # The channel value stands unless the checkpoint PROVABLY cannot start under it. The
    # comparison is against the bare requirement, with no margin added: margin is headroom on
    # a value this function chooses for itself, not a threshold for overriding a value that
    # was measured on real hardware. Applying it to the comparison instead nudged MiroThinker
    # -- the very model 0.80 was certified against -- from 0.800 to 0.802, changing a certified
    # number on the strength of an estimate, for 0.16 GiB. A test caught it; the reasoning is
    # written here so it does not come back.
    if fp["required_utilization"] <= channel_util:
        print(f"[gen-sft] channel value {channel_util} is sufficient "
              f"({(channel_util - fp['required_utilization']) * fp['device_total_gib']:.2f} "
              f"GiB of slack); using it unchanged", flush=True)
        return channel_util
    need = fp["required_utilization"] + margin
    if need > ceiling:
        raise SystemExit(
            f"FATAL: {model_dir} cannot be served at max_model_len={max_model_len} on this "
            f"card.\n  weights {fp['weights_gib']:.2f} GiB + overhead "
            f"{fp['overhead_gib']:.2f} + KV {fp['kv_gib_per_request']:.2f} = "
            f"{fp['required_gib_one_request']:.2f} GiB of {fp['device_total_gib']:.2f} GiB "
            f"total, i.e. utilization {fp['required_utilization']:.3f} before any margin.\n"
            f"  Do NOT lower max_model_len to make this fit: it is rollout.max_len + 2048, "
            f"and cutting it silently caps how long a trajectory may get -- the parameter "
            f"would end up in the paper and the cause would be this message.\n"
            f"  Use a larger card, or tensor-parallel across two.")
    print(f"[gen-sft] channel value {channel_util} is BELOW what this checkpoint needs "
          f"({need:.3f} incl. {margin} margin) -- raising it for this SAMPLING engine only. "
          f"The scoring teacher's {channel_util} is untouched.", flush=True)
    return round(need, 3)


def pin_determinism_env() -> None:
    """FORCE the determinism pins before vllm is imported.

    VLLM_BATCH_INVARIANT is set, never defaulted: a stale env of 0 would silently turn off
    the batch-invariance the teacher's whole online==offline guarantee rests on, and the
    probe would then certify a configuration nobody runs. The allocator hint is a belt
    against fragmentation eating the log_softmax headroom; it affects no computed value.
    """
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def teacher_engine_kwargs(*, dtype: str = TEACHER_DTYPE,
                          max_model_len: int = TEACHER_MAX_MODEL_LEN,
                          gpu_memory_utilization: float = TEACHER_GPU_MEMORY_UTILIZATION,
                          max_num_batched_tokens: int = TEACHER_MAX_NUM_BATCHED_TOKENS,
                          tensor_parallel_size: int = 1,
                          **extra) -> dict:
    """vLLM LLM(...) kwargs for the served teacher.

    enforce_eager: no CUDA-graph variance. enable_prefix_caching=False: the KV cache is
    keyed on tokens, so a cached prefill would return logprobs computed under a previous
    batch (and, in the probe, hide the very co-batch effect being measured).
    """
    kwargs = dict(
        dtype=dtype,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        tensor_parallel_size=tensor_parallel_size,
    )
    kwargs.update(extra)
    return kwargs


def config_summary(**overrides) -> str:
    """One-line record of what was actually built -- print it from every entry point so a
    job log always says which configuration produced its numbers."""
    kw = teacher_engine_kwargs(**overrides)
    return (f"dtype={kw['dtype']} max_model_len={kw['max_model_len']} "
            f"gpu_mem_util={kw['gpu_memory_utilization']} tp={kw['tensor_parallel_size']} "
            f"enforce_eager={kw['enforce_eager']} prefix_caching={kw['enable_prefix_caching']} "
            f"VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT')} "
            f"max_num_batched_tokens={kw.get('max_num_batched_tokens')} "
            f"max_num_seqs={kw.get('max_num_seqs')}")
