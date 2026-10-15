#!/usr/bin/env python3
"""Is the GPU on THIS node actually usable by THIS torch? Fail in seconds, not minutes.

    python scripts/gpu_preflight.py            # preflight: exit 1 if the node is broken
    python scripts/gpu_preflight.py --survey   # just print, always exit 0

WHY THIS EXISTS. `srun --gres=gpu:a800:1` landed on <node>, whose driver is CUDA 12.2 --
too old for this torch build. The GPU is physically there and SLURM happily allocated it;
torch simply cannot initialise. That burned a queue slot to learn one line of information,
and a real job (precompute, training) would burn its allocation the same way.

Driver version is a property of the NODE, not of the GPU model. So pinning `--gres=gpu:h20`
buys hardware uniformity, not a working driver -- the two have to be checked separately, and
this is the check.

IT ALSO CLOSES A GATE HOLE. GPU tests guard themselves with

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

which cannot tell "no GPU here" (a legitimate skip on a laptop) from "GPU present but the
driver is too old" (a broken node that should shout). On <node> the whole slow suite reported
`1 skipped` for exactly the gate we were there to run. Nodes that HAVE a GPU must prove they
can use it; a machine with no GPU at all is free to pass.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time


def smi() -> list[tuple[str, str]]:
    """[(gpu name, driver version)] straight from nvidia-smi. Empty if there is no GPU."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.splitlines():
        if "," in line:
            name, drv = line.split(",", 1)
            rows.append((name.strip(), drv.strip()))
    return rows


def who_holds_the_gpus() -> str:
    """nvidia-smi's view of WHO is using each card, for the failure path only.

    WHY IT EXISTS. Seven jobs died to cudaErrorDevicesUnavailable across six nodes before
    anyone asked this question, and two rounds of theory were built without it. The first --
    "it is the previous tenant's context still tearing down" -- predicted seconds and was
    killed by a 5-minute retry that did not clear. The second -- "it is always device 0
    because SLURM hands out the lowest index first" -- was killed by a failure on device 1.

    Both were inferences from the outside. This asks the driver directly: which processes hold
    which card, and how much memory each is using. A card held by a PID that is not ours is a
    scheduler/cleanup problem on that node; a card held by nobody while the allocation still
    fails is a driver problem. Those need opposite responses, and no amount of retrying tells
    them apart.
    """
    if shutil.which("nvidia-smi") is None:
        return "  (no nvidia-smi)"
    out = []
    for title, args in (
            ("per-card memory",
             ["--query-gpu=index,uuid,memory.used,memory.total,compute_mode",
              "--format=csv,noheader"]),
            ("processes holding a card",
             ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
              "--format=csv,noheader"])):
        try:
            p = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
            body = p.stdout.strip() or "(none)"
        except (OSError, subprocess.SubprocessError) as e:
            body = f"(nvidia-smi failed: {e})"
        out.append(f"  {title}:")
        out.extend("    " + l for l in body.splitlines())
    out.append(f"  this job: SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '?')} "
               f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")
    return "\n".join(out)


def retry(fn, tries: int, delay: float, what: str):
    """Call `fn` until it stops raising, up to `tries` times, `delay` seconds apart.

    Returns (result, None) on success and (None, last_exception) if it never succeeded.

    WHY A RETRY AND NOT A FAILURE. cudaErrorDevicesUnavailable -- "CUDA-capable device(s)
    is/are busy or unavailable" -- fires when a fresh allocation lands on a GPU whose previous
    tenant has not finished releasing it. It clears on its own within tens of seconds. On
    2026-08-24 it killed the dense teacher twice and then BOTH dense arms; slurm/gen_line.sh
    and slurm/sft_train.sbatch already retry for exactly this, and this script did not.
    """
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn(), None
        except Exception as e:                        # noqa: BLE001 -- report anything
            last = e
            if attempt < tries:
                print(f"[gpu-preflight] {what}: attempt {attempt}/{tries} failed "
                      f"({type(e).__name__}: {str(e).splitlines()[0][:100]}); "
                      f"retrying in {delay:g}s")
                time.sleep(delay)
    return None, last


def probe_all_devices(torch, n: int, tries: int, delay: float) -> bool:
    """TOUCH every visible device, not just device 0. Returns True if all of them answered.

    THE FAILURE THIS ENCODES. This script used to call `torch.cuda.mem_get_info()` with no
    argument -- which reads DEVICE 0 -- and never allocated anything anywhere. It therefore
    printed "OK" on a node where cuda:1 was unusable, and the online arm discovered that only
    after loading the vLLM policy on cuda:0 (60 s of engine start-up, torch.compile and CUDA
    graph capture included) and then calling model.to("cuda:1") for the student. The channel
    puts the student on cuda:1 by design, so the one device this check skipped is the one the
    online arm needs second.

    A tiny allocation is the point: torch.cuda.is_available() is a driver handshake and
    succeeds while a device is still busy. The error only surfaces when memory is actually
    requested, which is what this does -- in the preflight process, where the retry is cheap
    and the context is freed on exit.
    """
    ok = True
    for i in range(n):
        def _touch(idx=i):
            t = torch.zeros(8, device=f"cuda:{idx}")
            torch.cuda.synchronize(idx)
            free, total = torch.cuda.mem_get_info(idx)
            del t
            return free, total

        got, err = retry(_touch, tries, delay, f"device {i}")
        if err is not None:
            print(f"[gpu-preflight] FAIL: device {i} never became usable after {tries} "
                  f"attempts -- {type(err).__name__}: {str(err).splitlines()[0][:160]}",
                  file=sys.stderr)
            # ASK THE DRIVER WHO HAS IT. Retrying harder cannot distinguish "a context is
            # still closing" from "another process owns this card", and those need opposite
            # responses. Printed once, on the failure path, so a healthy run pays nothing.
            print("[gpu-preflight] who holds the cards, per nvidia-smi:", file=sys.stderr)
            print(who_holds_the_gpus(), file=sys.stderr)
            ok = False
            continue
        free, total = got
        print(f"[gpu-preflight] device {i} free {free / 2**30:.1f} / {total / 2**30:.1f} GiB"
              "  (allocation succeeded)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Prove this node's GPU is usable by this torch.")
    ap.add_argument("--survey", action="store_true",
                    help="report only; never fail (for sweeping nodes)")
    ap.add_argument("--compute", action="store_true",
                    help="also RUN kernels, not just initialise CUDA")
    ap.add_argument("--device-tries", type=int, default=6,
                    help="attempts per device before declaring it unusable (default 6). "
                         "Matches slurm/sft_train.sbatch, which has retried this transient "
                         "since 2026-08-22.")
    ap.add_argument("--device-delay", type=float, default=10.0,
                    help="seconds between attempts (default 10)")
    args = ap.parse_args()

    host = platform.node()
    gpus = smi()
    print(f"[gpu-preflight] node={host}")
    if gpus:
        for i, (name, drv) in enumerate(gpus):
            print(f"[gpu-preflight]   gpu{i}: {name}   driver={drv}")
    else:
        print("[gpu-preflight]   nvidia-smi reports no GPU")

    try:
        import torch
    except ImportError:
        print("[gpu-preflight] torch is not importable")
        return 0 if args.survey else 1

    print(f"[gpu-preflight] torch={torch.__version__} built for CUDA {torch.version.cuda}")
    # A CPU-ONLY BUILD IS NOT A BROKEN NODE. Without this branch the script blames the
    # driver ("too old") on any machine with a `+cpu` torch -- a wrong diagnosis from the
    # tool whose whole job is to stop wrong diagnoses. Separate "this torch cannot do CUDA
    # at all" from "this torch does CUDA but not with THIS driver".
    if torch.version.cuda is None:
        print("[gpu-preflight] this is a CPU-ONLY torch build -- no CUDA expected, fine")
        return 0

    usable, why = False, ""
    try:
        usable = torch.cuda.is_available()
        if usable:
            n = torch.cuda.device_count()
            caps = [f"{torch.cuda.get_device_name(i)} sm_{'%d%d' % torch.cuda.get_device_capability(i)}"
                    for i in range(n)]
            print(f"[gpu-preflight] torch sees {n} device(s): {', '.join(caps)}")
            if not probe_all_devices(torch, n, tries=args.device_tries,
                                     delay=args.device_delay):
                return 0 if args.survey else 1
        else:
            why = "torch.cuda.is_available() is False"
    except RuntimeError as e:                       # the too-old-driver path raises here
        why = str(e).strip().splitlines()[0]

    if usable and args.compute:
        # INITIALISING CUDA IS NOT THE SAME AS RUNNING ON IT. torch.cuda.is_available() only
        # says the driver handshake succeeded; the kernels this project actually needs are
        # compiled for a compute capability and dispatched through cuBLAS, and a
        # driver/toolkit combination can pass the handshake and then fail (or silently produce
        # garbage) on the first real launch. That distinction is the whole question when
        # asking whether a card of a different architecture can host this environment.
        try:
            a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            b = (a @ a.T).float()                        # cuBLAS bf16 GEMM
            lsm = torch.log_softmax(b[:64].float(), dim=-1)   # the op the teacher scores with
            torch.cuda.synchronize()
            ok = bool(torch.isfinite(b).all() and torch.isfinite(lsm).all())
            print(f"[gpu-preflight] compute: bf16 GEMM + log_softmax ran, finite={ok}, "
                  f"checksum={float(b.abs().mean()):.6f}")
            if not ok:
                print("[gpu-preflight] FAIL: kernels ran but produced non-finite values",
                      file=sys.stderr)
                return 0 if args.survey else 1
        except Exception as e:                            # noqa: BLE001 - report anything
            print(f"[gpu-preflight] FAIL: CUDA initialised but a real kernel could not run "
                  f"-- {type(e).__name__}: {str(e).splitlines()[0][:160]}", file=sys.stderr)
            return 0 if args.survey else 1

    if usable:
        print("[gpu-preflight] OK")
        return 0

    # No GPU at all is a legitimate CPU box. A GPU that torch cannot drive is a BROKEN NODE,
    # and saying so here costs seconds where discovering it downstream costs an allocation.
    if not gpus:
        print("[gpu-preflight] no GPU on this node -- CPU-only run, that is fine")
        return 0
    print(f"[gpu-preflight] FAIL: {host} HAS a GPU ({gpus[0][0]}, driver {gpus[0][1]}) and "
          f"torch is a CUDA build (for {torch.version.cuda}), but it cannot use it -- {why}")
    print("[gpu-preflight] this node's driver is too old for this torch build. Driver version "
          "is per-NODE, so another node of the SAME gpu type may well work; re-submit "
          "elsewhere rather than concluding the gpu type is unusable.")
    return 0 if args.survey else 1


if __name__ == "__main__":
    sys.exit(main())
