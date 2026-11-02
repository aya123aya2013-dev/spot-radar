"""Journal integrity and agent bookkeeping invariants on synthetic data."""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_journal_chain_detects_tampering():
    import journal
    d = tempfile.mkdtemp()
    p = os.path.join(d, "j.jsonl")
    journal.append([dict(a=1), dict(a=2)], p)
    journal.append([dict(a=3)], p)
    assert journal.verify(p) == (True, 3, None)
    lines = open(p).read().splitlines()
    rec = json.loads(lines[1])
    rec["a"] = 99
    lines[1] = json.dumps(rec)
    open(p, "w").write("\n".join(lines) + "\n")
    ok, n, bad = journal.verify(p)
    assert not ok and bad == 2


def run_agent(data_dir, *args):
    env = dict(os.environ, SPOT_LAB_DATA=data_dir)
    return subprocess.run([sys.executable, os.path.join(ROOT, "agent.py"), "--synthetic", *args], env=env,
                          capture_output=True, text=True, timeout=600)


def test_agent_cycle_bookkeeping():
    d = tempfile.mkdtemp()
    for _ in range(2):
        r = run_agent(d)
        assert r.returncode == 0, r.stderr
    st = json.load(open(os.path.join(d, "paper_G.json")))
    assert st["control"]["mode"] == "paper"
    assert st["cash"] >= -1e-6, "cash can never go negative"
    eq = st["cash"] + sum(h["value"] for h in st["holdings"].values())
    assert abs(eq - st["equity"]) < 1e-6
    for h in st["holdings"].values():
        assert h["risk_usd"] <= 0.01 * st["equity"] * 1.5 + 1.0, "open risk above the per-trade cap"
        assert h["basis"] <= 0.25 * 5000 + 1.0, "position above max_position"
    assert st["journal"]["intact"]


def test_emergency_stop_blocks_agent_entries():
    d = tempfile.mkdtemp()
    ctl = os.path.join(ROOT, "control.json")
    orig = open(ctl).read()
    try:
        c = json.loads(orig)
        c["emergency_stop"] = True
        open(ctl, "w").write(json.dumps(c))
        r = run_agent(d)
        assert r.returncode == 0, r.stderr
        st = json.load(open(os.path.join(d, "paper_G.json")))
        assert not st["pos"], "an entry was made during emergency stop"
        assert "emergency_stop" in st["health"]["gate_reasons"]
    finally:
        open(ctl, "w").write(orig)


if __name__ == "__main__":
    test_journal_chain_detects_tampering()
    test_agent_cycle_bookkeeping()
    test_emergency_stop_blocks_agent_entries()
    print("3 journal/agent tests passed")
