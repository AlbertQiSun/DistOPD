"""ref_topk_dump: the PRODUCER on a tiny Qwen3 against a longhand full-logits reference that
shares no code with it; the first-100 gate; the alignment gate; the sharded, resumable run
end to end on a synthetic cache."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import ref_topk_dump as rd  # noqa: E402

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return Qwen3ForCausalLM(cfg).eval()


def _longhand(model, ids):
    with torch.no_grad():
        return torch.log_softmax(model(input_ids=ids[None]).logits[0].double(), -1).numpy()


def test_producer_emits_every_field_and_matches_longhand_logits():
    from opd.trajectory import TokenType
    model = _tiny()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 64, (30,), generator=g)
    tt = torch.zeros(30, dtype=torch.long)
    tt[10:20] = int(TokenType.ACTION); tt[25:28] = int(TokenType.ACTION)
    d = rd.dump_trajectory(model, ids, tt, k=5, device="cpu", row_chunk=4)
    assert set(d) == set(rd.FIELDS) - {"traj"}
    pos = np.concatenate([np.arange(10, 20), np.arange(25, 28)])
    assert d["pos"].tolist() == pos.tolist() and d["y"].tolist() == ids[pos].tolist()
    full = _longhand(model, ids)
    for i, p in enumerate(pos):
        row = full[p - 1]; y = int(ids[p])
        assert abs(d["lp_y"][i] - row[y]) < 1e-4
        order = np.argsort(-row)[:5]
        assert d["top_ids"][i].tolist() == order.tolist()
        assert np.allclose(d["top_lp"][i], row[order], atol=1e-4)
        assert d["y_rank"][i] == int((row > row[y]).sum()) + 1
        assert abs(d["topk_mass"][i] - np.exp(row[order]).sum()) < 1e-4
        assert abs(d["row_sum"][i] - 1.0) < 1e-4
    assert d["top_lp"].dtype == np.float32 and d["top_ids"].dtype == np.int32
    assert rd.dump_trajectory(model, ids, torch.zeros(30, dtype=torch.long), 5, "cpu") is None


def test_gate_first_100_as_written():
    n = 150
    top_lp = np.log(np.tile(np.array([0.5, 0.3, 0.1]), (n, 1)))
    y_rank = np.ones(n, int); row_sum = np.ones(n)
    g = rd.gates(top_lp, y_rank, row_sum, k=3)
    assert g["passed"] and g["n_first"] == 100 and g["y_in_k_first"] == 1.0
    y_rank[3] = 7                                   # one of the first 100 outside K' -> 0.99 < 0.999
    assert not rd.gates(top_lp, y_rank, row_sum, k=3)["passed"]
    y_rank[3] = 1; y_rank[120] = 7                  # outside the first 100: reported, not gated
    g = rd.gates(top_lp, y_rank, row_sum, k=3)
    assert g["passed"] and g["y_in_k_part0"] < 1.0
    top_lp[0, 0] = np.log(0.7)                      # Σ exp(top-K') = 1.1 > 1
    assert not rd.gates(top_lp, y_rank, row_sum, k=3)["passed"]


def test_alignment_gate_token_mismatch_is_fatal_and_tolerance_is_strict():
    traj, pos, y = np.array([0, 0, 1]), np.array([5, 9, 3]), np.array([11, 12, 13])
    lp_y = np.array([-0.5, -1.0, -2.0], np.float32)
    tail_traj, tail_pos, tail_y = np.array([0, 1, 4]), np.array([9, 3, 1]), np.array([12, 13, 99])
    tail_lp = np.array([[-0.1, -1.0004, -3.0], [-2.0, -2.5, -3.0], [-0.2, -1.0, -3.0]], np.float32)
    tail_tok = np.array([[7, 12, 8], [13, 5, 6], [99, 1, 2]])
    tail_rank = np.array([2, 1, 1])
    al = rd.align(traj, pos, y, lp_y, tail_traj, tail_pos, tail_y, tail_lp, tail_tok, tail_rank, traj_max=2)
    assert al["n_in_range"] == 2 and al["n_shared"] == 2 and al["n_token_mismatch"] == 0 and al["passed"] and al["max_abs"] < 1e-3
    al = rd.align(traj, pos, y, lp_y, tail_traj, tail_pos, tail_y, tail_lp, tail_tok, tail_rank, tol=1e-4, traj_max=2)
    assert not al["passed"] and al["n_over_tol"] == 1
    # without traj_max, tail traj 4 is a position this dump does not have -> a mismatch
    assert rd.align(traj, pos, y, lp_y, tail_traj, tail_pos, tail_y, tail_lp, tail_tok, tail_rank)["n_token_mismatch"] == 1
    tail_y2 = tail_y.copy(); tail_y2[0] = 77
    assert rd.align(traj, pos, y, lp_y, tail_traj, tail_pos, tail_y2, tail_lp, tail_tok, tail_rank, traj_max=2)["n_token_mismatch"] == 1
    tail_tok2 = tail_tok.copy(); tail_tok2[1] = [5, 6, 8]          # y absent from the dump's top-K: skipped, not compared
    assert rd.align(traj, pos, y, lp_y, tail_traj, tail_pos, tail_y, tail_lp, tail_tok2, tail_rank, traj_max=2)["n_skipped_y_below_topK"] == 1


def _synthetic_cache(tmp_path, model, n_traj=5, seed=2):
    """traj_*.pt records + manifest in the cache's own layout, and a tail_slope-shaped npz whose
    pi_ref numbers come from LONGHAND full logits (not from the code under test)."""
    from opd.trajectory import TokenType
    cache = tmp_path / "cache"; cache.mkdir()
    g = torch.Generator().manual_seed(seed)
    names, tail = [], dict(traj=[], pos=[], y=[], lp_ref=[], tok_ids=[], y_rank_ref=[])
    for ti in range(n_traj):
        L = 20 + 4 * ti
        ids = torch.randint(0, 64, (L,), generator=g)
        tt = torch.zeros(L, dtype=torch.long); tt[8:L - 2] = int(TokenType.ACTION)
        torch.save({"input_ids": ids, "token_type": tt, "teacher_logp": torch.zeros(L)}, cache / f"traj_{ti:04d}.pt")
        names.append(f"traj_{ti:04d}.pt")
        full = _longhand(model, ids)
        for p in (9, 12):                                   # two scored positions per trajectory, tail-style
            row = full[p - 1]; order = np.argsort(-row)[:40]          # top-40 of 64: y sometimes below it, mostly inside
            tail["traj"].append(ti); tail["pos"].append(p); tail["y"].append(int(ids[p]))
            tail["lp_ref"].append(row[order].astype(np.float32)); tail["tok_ids"].append(order.astype(np.int32))
            tail["y_rank_ref"].append(int((row > row[int(ids[p])]).sum()) + 1)
    json.dump({"paths": names, "n": n_traj, "backend": {"template_fp": "deadbeef"}}, open(cache / "manifest.json", "w"))
    tail_npz = tmp_path / "tail.npz"
    np.savez(tail_npz, **{k: np.array(v) if k not in ("lp_ref", "tok_ids") else np.stack(v) for k, v in tail.items()})
    return cache, tail_npz


def test_sharded_run_writes_parts_gates_part0_resumes_and_assembles(tmp_path, monkeypatch, capsys):
    model = _tiny()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(lambda *a, **k: model.to(k.get("device_map", {"": "cpu"})[""])))
    cache, tail_npz = _synthetic_cache(tmp_path, model)
    args = ["--line", "x", "--cache", str(cache), "--pi-ref", "/fake", "--k", "64", "--align", str(tail_npz), "--shard-size", "2"]
    rc = rd.main(args + ["--n", "3"])                       # 2 parts: [0,2) and [2,3)
    assert rc == 0
    out = cache / "ref_topk"
    assert sorted(os.listdir(out)) == ["gates.json", "part_000000.npz", "part_000002.npz"]
    G = json.load(open(out / "gates.json"))
    assert G["passed"] and G["align"]["n_in_range"] == 4 and G["align"]["n_token_mismatch"] == 0 and G["align"]["max_abs"] < 1e-4
    assert G["align"]["n_shared"] + G["align"]["n_skipped_y_below_topK"] == 4 and G["align"]["n_shared"] >= 1
    assert (cache / "ref_topk.npz").exists() and "COMPLETE: 2/2 parts" in open(cache / "ref_topk_report.txt").read()
    # resume over the FULL cache: the two parts are kept, only [4,5) is new, gates.json is not recomputed
    m0 = os.path.getmtime(out / "part_000000.npz"); mg = os.path.getmtime(out / "gates.json")
    rc = rd.main(args)
    assert rc == 0
    assert os.path.getmtime(out / "part_000000.npz") == m0 and os.path.getmtime(out / "gates.json") == mg
    assert sorted(existing := rd.existing_parts(str(out))) == [0, 2, 4]
    log = capsys.readouterr().out
    assert "2 done, 1 to do" in log and "COMPLETE: 3/3 parts" in log
    # the reader sees every part, in order, with the arrays the writer emitted
    parts = list(rd.iter_parts(str(cache)))
    assert [s["start"] for s, _ in parts] == [0, 2, 4] and all(set(z) == set(rd.FIELDS) for _, z in parts)
    assert sum(s["n_pos"] for s, _ in parts) == sum(len(z["pos"]) for _, z in parts) > 0
    # --report-only re-assembles without a model
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model"))))
    assert rd.main(["--cache", str(cache), "--report-only"]) == 0


def test_continuation_refuses_a_failed_gate_and_diagnose_lists_the_flagged_positions(tmp_path, monkeypatch, capsys):
    model = _tiny()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(lambda *a, **k: model.to(k.get("device_map", {"": "cpu"})[""])))
    cache, tail_npz = _synthetic_cache(tmp_path, model)
    args = ["--line", "x", "--cache", str(cache), "--pi-ref", "/fake", "--k", "64", "--align", str(tail_npz), "--shard-size", "2"]
    assert rd.main(args + ["--n", "2"]) == 0                              # part 0 only, gates pass
    out = cache / "ref_topk"
    G = json.load(open(out / "gates.json")); G["passed"] = False; G["align"]["passed"] = False
    json.dump(G, open(out / "gates.json", "w"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("must not load a model"))))
    assert rd.main(args) == 1                                            # refused before the model is touched
    assert "records a failed gate" in capsys.readouterr().out and rd.existing_parts(str(out)) == [0]
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(lambda *a, **k: model.to(k.get("device_map", {"": "cpu"})[""])))
    # --ignore-gates scores on; the exit code still reports the recorded failure (0 needs complete AND passed)
    assert rd.main(args + ["--ignore-gates", "--n", "4"]) == 1 and rd.existing_parts(str(out)) == [0, 2]
    # diagnose: perturb the tail's value at one shared position so it is flagged, and read the fresh re-scores
    T = dict(np.load(tail_npz, allow_pickle=True))
    j, hit = next((j, h) for j in range(len(T["y"])) if int(T["traj"][j]) < 2
                  for h in [np.where(T["tok_ids"][j] == T["y"][j])[0]] if h.size)   # first shared row with y inside the tail's top-K
    T["lp_ref"][j, hit[0]] += 0.5
    np.savez(tail_npz, **T)
    assert rd.main(args + ["--diagnose-align"]) == 0
    D = json.load(open(out / "gates_diag.json"))
    assert len(D) == 1 and D[0]["traj"] == int(T["traj"][j]) and D[0]["pos"] == int(T["pos"][j]) and abs(D[0]["diff"] + 0.5) < 1e-4
    fresh = D[0]["fresh"]
    assert set(fresh) == {"chunk32_rep1", "chunk32_rep2", "chunk64_rep1", "chunk64_rep2"}
    assert all(abs(v["lp_y"] - D[0]["lp_dump"]) < 1e-4 for v in fresh.values())   # this node reproduces the dump


def test_adjudication_is_recorded_beside_the_failed_gate_not_over_it(tmp_path, monkeypatch, capsys):
    model = _tiny()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(lambda *a, **k: model.to(k.get("device_map", {"": "cpu"})[""])))
    cache, tail_npz = _synthetic_cache(tmp_path, model)
    args = ["--line", "x", "--cache", str(cache), "--pi-ref", "/fake", "--k", "64", "--align", str(tail_npz), "--shard-size", "2"]
    assert rd.main(args) == 0
    out = cache / "ref_topk"
    G = json.load(open(out / "gates.json")); G["passed"] = False; G["align"]["passed"] = False; json.dump(G, open(out / "gates.json", "w"))
    assert rd.main(["--cache", str(cache), "--report-only"]) == 1
    assert rd.main(["--cache", str(cache), "--adjudicate", "3 positions, dump reproduces on a second node; tail dump is the odd one"]) == 0
    G = json.load(open(out / "gates.json"))
    assert G["passed"] is False and G["adjudication"]["note"].startswith("3 positions")
    z = np.load(cache / "ref_topk.npz", allow_pickle=True)
    assert bool(z["gates_passed"]) is False and bool(z["adjudicated"]) and bool(z["usable"])
    rep = str(z["report"]); assert "FAILED, ADJUDICATED: 3 positions" in rep and "line=x" in rep and "pi_ref=fake" in rep


def test_no_alignment_reference_skips_the_gate_instead_of_failing_it(tmp_path, monkeypatch):
    model = _tiny()
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(lambda *a, **k: model.to(k.get("device_map", {"": "cpu"})[""])))
    cache, _tail = _synthetic_cache(tmp_path, model)
    assert rd.main(["--line", "x", "--cache", str(cache), "--pi-ref", "/fake", "--k", "64", "--align", "none", "--shard-size", "2"]) == 0
    G = json.load(open(cache / "ref_topk" / "gates.json"))
    assert G["passed"] and G["align"].get("skipped") and bool(np.load(cache / "ref_topk.npz", allow_pickle=True)["usable"])
