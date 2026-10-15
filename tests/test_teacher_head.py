"""delta-OPD step 2 (the teacher head) and its consumers, end to end on synthetic data:
teacher.head's selection rule; the REAL /score_head handler + the REAL client; the dump on a
synthetic cache (parts, gates, report); the DeltaCacheReader join; and the OPDTrainer with
loss=delta stepping through the unchanged offline loop."""
import importlib.util
import json
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opd import delta_loss as dl  # noqa: E402
from opd.delta_cache_reader import DeltaCacheReader  # noqa: E402
from scripts import teacher_head_dump as thd  # noqa: E402
from teacher.backend import BackendConfig  # noqa: E402
from teacher.head import head_from_entry, residual_log_mass, select_kq  # noqa: E402
from teacher.http_scorer import TeacherHTTPScorer  # noqa: E402


def _load_teacher_server():
    spec = importlib.util.spec_from_file_location("teacher_server_head", ROOT / "slurm" / "teacher_server.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


TS = _load_teacher_server()


# ---------------------------------------------------------------- 1. the selection rule
def test_head_from_entry_cuts_kq_dedupes_extras_and_flags_unknowns():
    lp = {10: np.log(0.6), 11: np.log(0.3), 12: np.log(0.05), 13: np.log(0.03), 14: np.log(0.01), 15: np.log(0.005)}
    assert select_kq(sorted(lp.values(), reverse=True), cover=0.99, k_cap=32) == 5      # 0.6+.3+.05+.03+.01 = 0.99
    assert select_kq(sorted(lp.values(), reverse=True), cover=0.99, k_cap=3) == 3
    h = head_from_entry(lp, y=12, extra_ids=[11, 99, 12, 15, 99], k_cap=32, cover=0.99)
    assert h["kq"] == 5 and h["top_ids"] == [10, 11, 12, 13, 14]
    assert h["extra_ids"] == [99, 15] and h["extra_lp"][0] is None and abs(h["extra_lp"][1] - np.log(0.005)) < 1e-12
    assert abs(h["lp_y"] - np.log(0.05)) < 1e-12 and abs(h["floor_lp"] - np.log(0.005)) < 1e-12 and h["cover_reached"]
    r = residual_log_mass(h["top_lp"], h["extra_lp"])
    assert abs(np.exp(r) - (1 - 0.99 - 0.005)) < 1e-9                     # unknown 99 counted as 0
    with pytest.raises(ValueError):
        head_from_entry(lp, y=42, extra_ids=[])


# ---------------------------------------------------------------- the fake engine behind the REAL handler
class _LP:
    def __init__(self, v): self.logprob = v


def _teacher_rows(V, seed):
    """A deterministic synthetic teacher: full log-prob rows keyed by (traj, pos)."""
    g = torch.Generator().manual_seed(seed)
    return lambda t, p: torch.log_softmax(torch.randn(V, generator=torch.Generator().manual_seed(seed * 100003 + t * 1009 + p)) * 2.5, -1)


def _fake_scorer(V, row_of, ids_of):
    """scorer_fn with .head: builds vLLM-shaped entries (top-k_wide + the realised token) from the table."""
    def head_fn(items, k_wide, k_cap, cover):
        out = []
        for input_ids, positions, extras in items:
            t = ids_of(tuple(input_ids))
            heads = []
            for p, ex in zip(positions, extras):
                row = row_of(t, p)
                order = torch.argsort(row, descending=True)[:k_wide].tolist()
                entry = {int(j): _LP(float(row[j])) for j in order}
                entry[int(input_ids[p])] = _LP(float(row[int(input_ids[p])]))
                heads.append(head_from_entry(entry, int(input_ids[p]), ex, k_cap, cover))
            out.append({"heads": heads, "n_input": len(input_ids)})
        return out

    def scorer_fn(input_ids, score_positions):
        raise AssertionError("not used")
    scorer_fn.head = head_fn
    return scorer_fn


@contextmanager
def _serve(scorer_fn, backend_dict):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), TS.make_handler(scorer_fn, backend_dict))
    th = threading.Thread(target=httpd.serve_forever, daemon=True); th.start()
    try:
        host, port = httpd.server_address
        yield f"http://{host}:{port}/score_logprobs"
    finally:
        httpd.shutdown(); httpd.server_close(); th.join(timeout=5)


# ---------------------------------------------------------------- 2. endpoint + client
def test_score_head_roundtrip_through_the_real_handler_and_client():
    V = 64
    row_of = _teacher_rows(V, seed=1)
    ids = torch.randint(0, V, (12,), generator=torch.Generator().manual_seed(3))
    scorer = _fake_scorer(V, row_of, lambda key: 0)
    backend = asdict(BackendConfig(engine="vllm_http", dtype="bfloat16", tokenizer_name="mock-teacher"))
    with _serve(scorer, backend) as url:
        client = TeacherHTTPScorer(endpoint=url, dtype="bfloat16", tokenizer_name="mock-teacher")
        heads = client.score_head([(ids, [3, 7], [[1, 2, int(ids[3])], [5, int(ids[7])]])], k_wide=16, k_cap=4, cover=0.99)
    assert len(heads) == 1 and len(heads[0]) == 2
    h = heads[0][0]
    row = row_of(0, 3)
    assert h["top_ids"] == torch.argsort(row, descending=True)[:h["kq"]].tolist() and h["kq"] <= 4
    assert abs(h["lp_y"] - float(row[int(ids[3])])) < 1e-6
    for t, v in zip(h["extra_ids"], h["extra_lp"]):
        assert t not in h["top_ids"] and (v is None or abs(v - float(row[t])) < 1e-6)
    # a wrong-shaped request is a 400, not a crash
    with _serve(scorer, backend) as url:
        client = TeacherHTTPScorer(endpoint=url, dtype="bfloat16", tokenizer_name="mock-teacher")
        with pytest.raises(Exception):
            client.score_head([(ids, [3, 7], [[1]])])


# ---------------------------------------------------------------- 3-5. the dump, the reader, the trainer
def _synthetic_world(tmp_path, n_traj=5, V=64, kp=6, seed=7):
    """A precompute cache (records + manifest, mock backend) plus ref_topk parts written in the
    step-1 layout from a synthetic pi_ref; the teacher table serves the fake /score_head AND
    fills the records' teacher_logp (so the alignment gate is a genuine identity)."""
    from opd.trajectory import TokenType
    from teacher.scorer import MOCK_CONFIG
    cache = tmp_path / "cache"; (cache / "ref_topk").mkdir(parents=True)
    row_of = _teacher_rows(V, seed)
    g = torch.Generator().manual_seed(seed)
    def pi(t, p):                                   # every third position saturated (hard: rank-2 below -8)
        lp = torch.randn(V, generator=torch.Generator().manual_seed(seed * 7 + t * 31 + p)) * 1.2
        if p % 3 == 0:
            lp[p % V] += 15.0
        return torch.log_softmax(lp, -1)
    names, table = [], {}
    ref = {f: [] for f in ("traj", "pos", "y", "top_ids", "top_lp", "lp_y", "y_rank", "topk_mass", "row_sum")}
    for t in range(n_traj):
        L = 14 + 2 * t
        ids = torch.randint(0, V, (L,), generator=g)
        tt = torch.zeros(L, dtype=torch.long); tt[5:] = int(TokenType.ACTION)
        tl = torch.full((L,), float("nan"))
        for p in range(5, L):
            tl[p] = float(row_of(t, p)[int(ids[p])])
            lp = pi(t, p); top = torch.topk(lp, kp)
            ref["traj"].append(t); ref["pos"].append(p); ref["y"].append(int(ids[p]))
            ref["top_ids"].append(top.indices.numpy().astype(np.int32)); ref["top_lp"].append(top.values.numpy().astype(np.float32))
            ref["lp_y"].append(float(lp[int(ids[p])])); ref["y_rank"].append(int((lp > lp[int(ids[p])]).sum()) + 1)
            ref["topk_mass"].append(float(top.values.exp().sum())); ref["row_sum"].append(1.0)
        torch.save({"input_ids": ids, "token_type": tt, "teacher_logp": tl, "logp_dtype": "float32", "backend": MOCK_CONFIG,
                    "meta": {}, "terminated": "answered", "num_search_calls": 1}, cache / f"traj_{t:04d}.pt")
        names.append(f"traj_{t:04d}.pt"); table[tuple(ids.tolist())] = t
    json.dump({"backend": asdict(MOCK_CONFIG), "paths": names, "n": n_traj, "sampler": {}}, open(cache / "manifest.json", "w"))
    Z = {k: (np.stack(v) if k in ("top_ids", "top_lp") else np.array(v)) for k, v in ref.items()}
    summ = dict(start=0, end=n_traj, n_traj=n_traj, n_empty=0, n_pos=len(Z["pos"]), k=kp, wall_s=0.0)
    np.savez(cache / "ref_topk" / "part_000000.npz", **Z, summary=np.array(json.dumps(summ)))
    np.savez(cache / "ref_topk.npz", meta=np.array("{}"), gates_passed=np.bool_(True), usable=np.bool_(True), report=np.array("synthetic"))
    scorer = _fake_scorer(V, row_of, lambda key: table[key])
    return cache, scorer, row_of, pi


def test_dump_reader_and_delta_trainer_on_a_synthetic_cache(tmp_path, capsys):
    from opd.trainer import OPDConfig, OPDTrainer, OfflineSource
    from teacher.scorer import MOCK_CONFIG, MockLM, Scorer
    cache, scorer, row_of, pi = _synthetic_world(tmp_path)
    backend = asdict(BackendConfig(engine="vllm_http", dtype="bfloat16", tokenizer_name="mock-teacher",
                                   template_fp=asdict(MOCK_CONFIG)["template_fp"]))
    with _serve(scorer, backend) as url:
        rc = thd.main(["--line", "x", "--cache", str(cache), "--teacher", "mock-teacher", "--teacher-endpoint", url,
                       "--k-wide", "16", "--k-cap", "4", "--shard-size", "1000"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "COMPLETE: 1/1 parts" in out and "GATES PASSED" in out and "soft positions" in out
    z = np.load(cache / "teacher_head" / "part_000000.npz", allow_pickle=True)
    assert set(thd.FIELDS) <= set(z.files) and json.loads(str(z["summary"]))["align"]["passed"]
    assert (cache / "teacher_head.npz").exists() and bool(np.load(cache / "teacher_head.npz", allow_pickle=True)["usable"])
    # a --n inside the shard is a smoke: it filters the part's trajectories and lands in teacher_head_smoke/
    with _serve(scorer, backend) as url:
        assert thd.main(["--cache", str(cache), "--teacher", "mock-teacher", "--teacher-endpoint", url, "--k-wide", "16",
                         "--k-cap", "4", "--n", "2"]) == 0
    zs = np.load(cache / "teacher_head_smoke" / "part_000000.npz", allow_pickle=True)
    assert set(np.unique(zs["traj"]).tolist()) <= {0, 1} and (cache / "teacher_head" / "part_000000.npz").exists()
    # only soft rows were scored; every one of them is a soft row of the ref part
    ref = np.load(cache / "ref_topk" / "part_000000.npz", allow_pickle=True)
    soft = ref["top_lp"][:, 1] > thd.SOFT_X2
    assert len(z["pos"]) == int(soft.sum()) > 0 and int((~soft).sum()) > 0
    # the reader joins both parts into a DeltaCache whose S holds K', K_q and y, with the teacher known on S at soft rows
    reader = DeltaCacheReader(str(cache), tau=10.0)
    rec = torch.load(cache / "traj_0002.pt", weights_only=False)
    c = reader.get(2, rec["teacher_logp"])
    assert c.n == int((ref["traj"] == 2).sum()) and torch.equal(c.pos, torch.tensor(ref["pos"][ref["traj"] == 2]).long())
    for i in range(c.n):
        S = c.s_ids[i][c.s_valid[i]]
        assert int(c.s_ids[i, c.y_col[i]]) == int(rec["input_ids"][c.pos[i]])
        if c.is_soft[i]:
            assert torch.isfinite(c.log_q[i][c.s_valid[i]]).all() and c.in_J[i][c.s_valid[i]].all()
            row = row_of(2, int(c.pos[i]))
            known = ~c.q_floored[i][c.s_valid[i]]
            assert torch.allclose(c.log_q[i][c.s_valid[i]][known].double(), row[S][known].double(), atol=1e-6)
        else:
            assert int(c.in_J[i].sum()) == 1
    # the trainer: loss=delta through the UNCHANGED offline loop (mask identity asserted inside the step)
    student = Scorer(MockLM(64, seed=5678), requires_grad=True)
    cfg = OPDConfig(mode="offline", steps=2, lr=1e-2, precompute_dir=str(cache), global_batch=2, loss="delta", beta=1.0)
    src = OfflineSource(questions=[], policy=None, retriever=None, tokenizer=None, teacher=None, cfg=cfg, expected_backend=MOCK_CONFIG)
    before = {k: v.clone() for k, v in student.lm.state_dict().items()}
    tr = OPDTrainer(student, src, cfg)
    outs = list(tr.train())
    assert len(outs) == 2 and all(torch.isfinite(o.loss) for o in outs) and all("loss_soft" in o.info for o in outs)
    assert any(not torch.equal(before[k], v) for k, v in student.lm.state_dict().items())
    # the gate: an incomplete / failed teacher_head refuses at construction
    np.savez(cache / "teacher_head.npz", meta=np.array("{}"), gates_passed=np.bool_(False), usable=np.bool_(False), report=np.array("x"))
    with pytest.raises(RuntimeError):
        DeltaCacheReader(str(cache))
    with pytest.raises(ValueError):
        OPDTrainer(student, src, OPDConfig(mode="online", loss="delta"))
