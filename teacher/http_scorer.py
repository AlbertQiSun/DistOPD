"""Real teacher client: HTTP to the served vLLM MiroThinker (slurm/teacher_server.py).

Drop-in for TeacherScorer / TeacherVLLMScorer -- same score_online / precompute_and_save
surface, and `.config` carries engine="vllm_http" so gate 1 (assert_same_backend) treats
online and offline as the identical served backend. The in-process TeacherVLLMScorer stays
a CI/dry-run tool ONLY; the real (non-dry-run) path is forced onto THIS served client by
the guard in scripts/train.py::make_teacher.

Pure-id contract (see slurm/teacher_server.py): we send the client's FROZEN input_ids
verbatim and never detok->retok. The server returns align-to-t logprobs (position 0 =
null); we rebuild the nan-at-0 tensor every other scorer emits, so downstream masking /
loss are byte-identical to the in-process backends.

Uses stdlib urllib (no new dependency), mirroring envs/http_retriever.py.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, replace

import torch

from opd.masking import build_score_mask, scored_indices
from opd.trajectory import Trajectory

from .backend import ENGINE_RUNTIME_FIELDS, BackendConfig
from .scorer import _save_precomputed


class TeacherHTTPError(RuntimeError):
    """A non-2xx from the served teacher, WITH the server's stated reason attached.

    Replaces leaking a bare urllib HTTPError: the server puts its reason in the response
    body for both the 400 contract violations and the 500 catch-all, but urllib raises
    before anyone reads the body, so the cause was being discarded. `.code` is preserved
    so callers can still branch on status (400 = the client violated the contract,
    500 = the server failed) without parsing text."""

    def __init__(self, code: int, detail: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


class TeacherHTTPScorer:
    """Scores a FIXED trajectory by calling the served teacher over HTTP."""

    def __init__(self, endpoint: str, dtype: str, tokenizer_name: str,
                 timeout: float = 120.0, verify: bool = True) -> None:
        # endpoint e.g. http://<node>:8100/score_logprobs
        self.endpoint = endpoint
        self.timeout = timeout
        self.config = BackendConfig(engine="vllm_http", dtype=dtype,
                                    tokenizer_name=tokenizer_name)
        if verify:
            self._verify_backend()

    # -- gate 1 across the service boundary --------------------------------------
    def _health_url(self) -> str:
        base = self.endpoint.rsplit("/", 1)[0]      # drop /score_logprobs
        return f"{base}/health"

    def _verify_backend(self) -> None:
        """Fail-fast if the SERVED backend fingerprint != this client's expected
        BackendConfig. The teacher's frozen template / dtype / tokenizer must match
        the student's, or the same token gets a different teacher logprob -- gate 1,
        now enforced where the two live in different processes."""
        req = urllib.request.Request(self._health_url(), method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        served = body.get("backend", {})
        want = asdict(self.config)
        # Compare only what the client DECLARES. The engine-runtime fields describe the
        # server's vLLM setup, which a client has no way to know in advance -- asserting
        # on them here would just fail every startup.
        diffs = [k for k in want
                 if k not in ENGINE_RUNTIME_FIELDS and str(served.get(k)) != str(want[k])]
        if diffs:
            lines = "\n".join(
                f"  {k}: client={want[k]!r}  served={served.get(k)!r}" for k in diffs)
            raise AssertionError(
                "served teacher backend != client expectation -- a cross-service "
                f"backend confound (template/dtype/tokenizer drift):\n{lines}"
            )
        # ADOPT the served engine runtime, so everything this client writes is stamped with
        # the configuration that ACTUALLY produced the numbers. That is what makes the
        # disk-boundary assert_same_backend able to catch a teacher restarted under
        # different settings -- the check the fingerprint could not make before.
        adopted = {k: served[k] for k in ENGINE_RUNTIME_FIELDS if k in served}
        if adopted:
            self.config = replace(self.config, **adopted)
            print(f"[teacher] adopted served engine runtime: "
                  + " ".join(f"{k}={v}" for k, v in adopted.items()))

    # -- core scoring ------------------------------------------------------------
    def logprobs(self, input_ids: torch.Tensor, score_positions=None) -> torch.Tensor:
        n = int(input_ids.shape[0])
        payload = {"input_ids": input_ids.tolist()}
        if score_positions is not None:
            payload["score_positions"] = [int(p) for p in score_positions]
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # The server puts the REASON in the response body ({"error": ...}) for both the
            # 400 contract violations and the 500 catch-all -- but urllib raises on >=400 and
            # the body is discarded unless we read it off the exception. Without this, a
            # server-side failure reaches the operator as a bare "HTTP Error 500: Internal
            # Server Error" with the actual cause thrown away on BOTH sides (the server's
            # handler does not re-raise). Surface it, and include the trajectory length --
            # a length-dependent failure (e.g. exceeding the teacher's max_model_len) is
            # otherwise indistinguishable from a random server fault.
            try:
                detail = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:                                   # noqa: BLE001
                detail = "<no response body>"
            raise TeacherHTTPError(
                exc.code, detail,
                f"teacher /score_logprobs -> HTTP {exc.code} for a {n}-token trajectory "
                f"({len(payload.get('score_positions') or [])} scored positions). "
                f"Server said: {detail}"
            ) from exc
        aligned = body["logprobs"]
        assert len(aligned) == n, (
            f"served logprobs length {len(aligned)} != input length {n} "
            "-- BOS/length guard should have caught this server-side")
        out = torch.tensor(
            [float("nan") if v is None else float(v) for v in aligned],
            dtype=torch.float32)
        return out

    def logprobs_batch(self, items, token_budget: int = None) -> list:
        """Score many trajectories, grouped into requests by TOKEN COUNT.

        Grouping is by tokens, NOT by trajectory count: the cost and the memory of scoring
        are set by how many TOKENS a prefill step handles, and trajectory lengths here span
        ~700 to ~16k. A fixed count of trajectories per request would therefore mean wildly
        different amounts of work per request -- and the OOM that killed this server came
        from exactly that confusion (a single 13,750-token sequence, not from too many).

        The default budget is the engine's own certified chunked-prefill budget, so a
        request carries about one prefill step of work. A trajectory longer than the budget
        is sent ALONE and chunked by the engine, which the determinism probe certified as
        numerically inert.

        `items` is [(input_ids, score_positions|None), ...]; returns aligned tensors in the
        SAME ORDER -- pairing is positional throughout, never by content.
        """
        from teacher.engine import TEACHER_MAX_NUM_BATCHED_TOKENS
        budget = token_budget or TEACHER_MAX_NUM_BATCHED_TOKENS
        items = list(items)
        if not items:
            return []

        groups, cur, cur_tokens = [], [], 0
        for it in items:
            n = int(it[0].shape[0])
            if cur and cur_tokens + n > budget:
                groups.append(cur); cur, cur_tokens = [], 0
            cur.append(it); cur_tokens += n
        if cur:
            groups.append(cur)

        out = []
        for g in groups:
            out.extend(self._post_batch(g))
        assert len(out) == len(items), "logprobs_batch dropped a trajectory"
        return out

    def score_head(self, items, k_wide: int = 128, k_cap: int = 32, cover: float = 0.99,
                   token_budget: int = None) -> list:
        """The teacher HEAD (delta-OPD step 2) at chosen positions, via POST /score_head.
        `items` is [(input_ids tensor, positions list, extra_ids list-of-lists), ...]; returns,
        in the SAME order, one list of head dicts per item (teacher.head.head_from_entry's
        shape: kq, top_ids, top_lp, extra_ids, extra_lp (None = outside top-k_wide), lp_y,
        floor_lp, n_entry, cover_reached). Grouped by token budget like logprobs_batch."""
        from teacher.engine import TEACHER_MAX_NUM_BATCHED_TOKENS
        budget = token_budget or TEACHER_MAX_NUM_BATCHED_TOKENS
        items = list(items)
        if not items:
            return []
        groups, cur, cur_tokens = [], [], 0
        for it in items:
            n = int(it[0].shape[0])
            if cur and cur_tokens + n > budget:
                groups.append(cur); cur, cur_tokens = [], 0
            cur.append(it); cur_tokens += n
        if cur:
            groups.append(cur)
        url = self.endpoint.rsplit("/", 1)[0] + "/score_head"
        out = []
        for g in groups:
            payload = {"k_wide": int(k_wide), "k_cap": int(k_cap), "cover": float(cover),
                       "batch": [{"input_ids": ids.tolist(), "positions": [int(p) for p in pos],
                                  "extra_ids": [[int(t) for t in ex] for ex in exs]} for ids, pos, exs in g]}
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:2000]
                except Exception:                                   # noqa: BLE001
                    detail = "<no response body>"
                raise TeacherHTTPError(exc.code, detail, f"teacher /score_head -> HTTP {exc.code} for a batch of "
                                                          f"{len(g)} trajectories. Server said: {detail}") from exc
            results = body["results"]
            if len(results) != len(g):
                raise TeacherHTTPError(200, "", f"teacher returned {len(results)} head results for {len(g)} sequences "
                                                "-- positional pairing; refusing")
            for (ids, pos, _e), r in zip(g, results):
                if r.get("n_input") != int(ids.shape[0]) or len(r["heads"]) != len(pos):
                    raise TeacherHTTPError(200, "", "head result does not match the request's length / position count")
                out.append(r["heads"])
        assert len(out) == len(items), "score_head dropped a trajectory"
        return out

    def _post_batch(self, group) -> list:
        payload = {"batch": [
            {"input_ids": ids.tolist(),
             **({} if sp is None else {"score_positions": [int(p) for p in sp]})}
            for ids, sp in group]}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=data, headers={"Content-Type": "application/json"})
        n_tok = sum(int(ids.shape[0]) for ids, _ in group)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:                                   # noqa: BLE001
                detail = "<no response body>"
            raise TeacherHTTPError(
                exc.code, detail,
                f"teacher /score_logprobs -> HTTP {exc.code} for a batch of {len(group)} "
                f"trajectories ({n_tok} tokens). Server said: {detail}") from exc

        results = body["results"]
        # Positional pairing again: a short or reordered response would give a trajectory
        # another one's teacher logprobs, and nothing downstream could detect it.
        if len(results) != len(group):
            raise TeacherHTTPError(
                200, "",
                f"teacher returned {len(results)} results for {len(group)} scored sequences "
                "-- results are paired by POSITION, so this would cross logprobs between "
                "trajectories. Refusing.")
        out = []
        for (ids, _sp), r in zip(group, results):
            aligned = r["logprobs"]
            n = int(ids.shape[0])
            assert len(aligned) == n, (
                f"served logprobs length {len(aligned)} != input length {n} "
                "-- BOS/length guard should have caught this server-side")
            out.append(torch.tensor(
                [float("nan") if v is None else float(v) for v in aligned],
                dtype=torch.float32))
        return out

    # -- TeacherScorer surface ---------------------------------------------------
    def score_online(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logprobs(input_ids)

    def precompute_and_save_batch(self, trajs, paths, store_dtype: str = "float32") -> list:
        """Score and store MANY trajectories, grouped into token-budgeted requests.

        Same span construction and same _save_precomputed as the single path -- this only
        changes how many trajectories ride in one request. Scoring one at a time is what
        made the offline precompute serial: at 38,400 trajectories that is the difference
        between ~16h and ~60h, i.e. between fitting the budget and not.
        """
        trajs, paths = list(trajs), list(paths)
        assert len(trajs) == len(paths), "one path per trajectory"
        items = [(t.input_ids, scored_indices(build_score_mask(t.token_type))) for t in trajs]
        logps = self.logprobs_batch(items)
        for traj, path, logp in zip(trajs, paths, logps):
            _save_precomputed(traj, logp, self.config, path, store_dtype)
        return logps

    def precompute_and_save(self, traj: Trajectory, path: str,
                            store_dtype: str = "float32") -> torch.Tensor:
        # send the ACTION positions too, so the server's span guard checks the
        # client's mask against the frozen ids on EVERY offline precompute (a live
        # cross-service masking-parity guard, not just a test).
        positions = scored_indices(build_score_mask(traj.token_type))
        logp = self.logprobs(traj.input_ids, score_positions=positions)
        _save_precomputed(traj, logp, self.config, path, store_dtype)
        return logp
