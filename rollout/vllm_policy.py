"""Real student policy: vLLM sampling. HPC only (lazy-imports vllm).

Implements the SAME `act(prefix_ids, turn) -> ActionStep` contract as MockPolicy.
The student conditions on `prefix_ids` -- the exact ids appended to the trajectory
so far -- and returns the ids it SAMPLED. Those ids are appended verbatim; `text`
is decode(ids) for control-flow parsing only. This is the whole point of the M6
seam: real generation flows through the id-passthrough path, so the online/offline
per-position identity proven on the mock survives on real weights.

Stop strings </search> and </answer> are kept in the output (include_stop_str_in_output)
so the closing tag -- and therefore the real ACTION/OBSERVATION boundary -- lands
inside the sampled ids, exactly where the boundary-merge hazard would bite if we ever
re-encoded.
"""
from __future__ import annotations

import torch

from configs import template as T
from rollout.policy import ActionStep


class VLLMPolicy:
    def __init__(self, llm, tokenizer, *, temperature: float = 0.8, top_p: float = 1.0,
                 max_action_tokens: int, seed: int = 0) -> None:
        """`max_action_tokens` HAS NO DEFAULT ON PURPOSE.

        It used to default to 512, and three of the four production call sites --
        train.py, stage3_eval.py, policy_invariance_probe.py -- never passed it, so they
        silently sampled at 512 while gen_sft_data.py (the only one that did pass it) ran
        the configured 2560. That is not a small discrepancy: measured on the production
        shards, the fraction of teacher actions longer than 512 is 1.29% for MiroThinker
        and 24.68% for Qwen3-30B-A3B (runs/prod_{miro,moe}_sft/manifest.jsonl, 2026-08-23).
        A truncated action carries no closing tag, so rollout routes it into the search
        branch and Gate A scores it a format failure -- which would have made the MoE line
        look ~19x worse than the MiroThinker line for reasons that have nothing to do with
        either teacher.

        A default is what let that happen: a call site that forgets the argument inherits a
        number nobody chose. Without a default it cannot compile. Pass
        cfg["rollout"]["max_action_tokens"] -- and note that indexing rather than .get() is
        also deliberate, so a config missing the key fails loudly instead of falling back.
        """
        self.llm = llm
        self.tok = tokenizer                    # HFTokenizerAdapter (for decode)
        self.temperature = temperature
        self.top_p = top_p
        self.max_action_tokens = max_action_tokens
        self.seed = seed

    def _sampling_params(self):
        """ONE definition of how this policy samples.

        `act` and `act_batch` must not construct these separately: a divergence would make
        a trajectory's tokens depend on whether the driver happened to serve it alone or in
        a gathered round, which is precisely the batching-changes-semantics failure the
        coroutine design exists to prevent -- reintroduced one layer down.
        """
        from vllm import SamplingParams

        return SamplingParams(
            temperature=self.temperature, top_p=self.top_p,
            max_tokens=self.max_action_tokens, seed=self.seed,
            stop=[T.SEARCH_CLOSE, T.ANSWER_CLOSE],
            include_stop_str_in_output=True,
        )

    def act(self, prefix_ids: torch.Tensor, turn: int,
            forced: bool = False) -> ActionStep:  # noqa: ARG002
        # `forced` is accepted for contract parity with MockPolicy but is a NO-OP here: on the
        # forced turn the env has already PREFILLED <answer> into prefix_ids, so the real model
        # simply continues from inside the answer span and stop=[</search>, </answer>] makes it
        # sample the body then halt at </answer>. No special sampling is needed.
        from vllm.inputs import TokensPrompt

        params = self._sampling_params()
        out = self.llm.generate(
            TokensPrompt(prompt_token_ids=prefix_ids.tolist()), params
        )[0].outputs[0]
        ids = torch.tensor(list(out.token_ids), dtype=torch.long)
        # decode from the SAMPLED ids (never re-encoded back into the trajectory)
        text = self.tok.decode(ids)
        return ActionStep(ids=ids, text=text)

    def act_batch(self, requests) -> list:
        """Serve a whole rollout round in ONE llm.generate so vLLM schedules the requests
        concurrently. This is what makes global_batch=256 feasible: without it the driver
        gathers a round and then hands vLLM one prompt at a time, so the batching exists in
        the scheduler and nowhere in the engine.

        `forced` needs no special handling -- on the forced turn the env has already
        PREFILLED <answer> into the prefix, so the model simply continues from inside the
        answer span (see `act`). Every request therefore shares one SamplingParams.

        Results are paired to requests POSITIONALLY, so a reordered or short return would
        give a trajectory another one's sampled tokens -- silently, since the ids are
        well-formed and the rollout continues. Refuse, as the teacher and retriever clients
        do at their own boundaries.
        """
        from vllm.inputs import TokensPrompt
        from rollout.policy import ActionStep

        requests = list(requests)
        if not requests:
            return []
        params = self._sampling_params()
        prompts = [TokensPrompt(prompt_token_ids=r[0].tolist()) for r in requests]
        outs = self.llm.generate(prompts, params)
        if len(outs) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(outs)} outputs for {len(requests)} prompts -- outputs "
                "are paired to requests by POSITION, so this would cross sampled tokens "
                "between trajectories. Refusing.")
        steps = []
        for out in outs:
            c = out.outputs[0]
            ids = torch.tensor(list(c.token_ids), dtype=torch.long)
            steps.append(ActionStep(ids=ids, text=self.tok.decode(ids)))
        return steps

    def sync_from(self, student) -> None:
        """Hot-update the vLLM engine's weights from the current training student so
        the next rollouts are truly on-policy. This is THE thing that makes online
        online; without it the engine keeps sampling from stale weights and the
        online arm silently degrades into a frozen (offline-like) run.

        HPC-ONLY / real-weight path: whether vLLM's live weight update actually lands
        (and its latency) is verified on GPU -- the mock engine in
        tests/test_online_resync.py proves the resync SEMANTICS (bit-equality after
        sync, staleness on skipped steps) without a GPU; tests/test_vllm_weight_hot_update.py
        (@slow) proves the real push LANDS + is REVERSIBLE. Run that @slow test to CERTIFY
        this method before trusting any online number.

        REQUIRES the engine IN-PROCESS (launch with VLLM_ENABLE_V1_MULTIPROCESSING=0). With
        vLLM V1's EngineCore in a SUBPROCESS the sampled model lives in another process and no
        in-process weight handle exists; `_locate_live_model` then fails LOUD rather than
        letting online silently sample from stale weights (the offline-vs-online confound)."""
        state = student.lm.state_dict()
        model = _locate_live_model(self.llm)
        model.load_weights((name, w) for name, w in state.items())


def _locate_live_model(llm):
    """Return the nn.Module the vLLM engine actually SAMPLES with, so sync_from can
    overwrite its weights in place. vLLM's internal layout is version- AND process-specific,
    so we probe the known layouts and FAIL LOUD if none resolves -- a silent miss would leave
    'online' sampling from STALE weights, precisely the false-poisoning confound this phase
    exists to eliminate.

    Layouts covered:
      * V0:            llm_engine.model_executor.driver_worker.model_runner.model
      * V1 in-process: llm_engine.engine_core[.engine_core].model_executor.driver_worker
                       [.worker].model_runner.model   (VLLM_ENABLE_V1_MULTIPROCESSING=0)
    """
    eng = getattr(llm, "llm_engine", llm)
    executors = []
    me = getattr(eng, "model_executor", None)          # V0
    if me is not None:
        executors.append(me)
    core = getattr(eng, "engine_core", None)           # V1 InprocClient (or MP client, no model)
    for c in (core, getattr(core, "engine_core", None)):
        cme = getattr(c, "model_executor", None)
        if cme is not None:
            executors.append(cme)
    for ex in executors:
        dw = getattr(ex, "driver_worker", None)
        for worker in (dw, getattr(dw, "worker", None)):   # driver_worker may wrap the Worker
            mr = getattr(worker, "model_runner", None)
            model = getattr(mr, "model", None)
            if model is not None and hasattr(model, "load_weights"):
                return model
    raise RuntimeError(
        "sync_from: could not locate the live vLLM model to hot-update. The V1 EngineCore is "
        "most likely in a SUBPROCESS -- launch the online job with VLLM_ENABLE_V1_MULTIPROCESSING=0 "
        "so the model stays in-process (or switch to a collective_rpc weight push). Refusing to "
        "continue: a silent miss here makes 'online' sample from STALE weights, the exact "
        "offline-vs-online confound this phase exists to eliminate."
    )
