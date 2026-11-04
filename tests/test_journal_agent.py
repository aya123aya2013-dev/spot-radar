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


def test_backtest_resumes_after_drawdown_halt():
    """Regression: a drawdown halt used to re-trigger forever (breach checked before the halt expired)."""
    sys.argv.append("--synthetic")
    import importlib, market
    importlib.reload(market)
    import agent
    importlib.reload(agent)
    data = {c: market.klines(c, "1d", 1500) for c in ("BTC", "ETH", "SOL", "AVAX", "LINK", "DOGE")}
    days = [r[0] // agent.DAY for r in data["BTC"]]
    cfg = dict(agent.risk.load_control(), max_drawdown=0.02, halt_days=5)       # tiny limit: forces many halts
    curve, trades, _, acc = agent.backtest_run(data, days[260], days[-1], risk_cfg=cfg, edge_table={})
    last_trade_day = max((days[260] + t["days"]) for t in trades) if trades else None
    vals = [p[1] for p in curve]
    late = vals[len(vals) * 3 // 4:]
    assert len(set(round(v, 6) for v in late)) > 1, "equity frozen in the last quarter: halt never released"
    sys.argv.remove("--synthetic")


def test_live_drawdown_not_overwritten_by_backtest():
    """Regression: the cost-support block once reused `dd`, so the live max drawdown showed the backtest's 23.75%."""
    import accounting
    d = tempfile.mkdtemp()
    csv_p = os.path.join(d, "paper_equity_G.csv")
    open(csv_p, "w").write("time,equity,btc\n2026-11-02 10:00,5000,77000\n2026-11-03 10:00,4950,76000\n")
    json.dump(dict(promotion=dict(challenger=True), challenger=dict(full=dict(cagr=0.147, max_dd=0.2375))),
              open(os.path.join(d, "backtest_G.json"), "w"))
    costs_p = os.path.join(d, "costs.json")
    json.dump(dict(monthly=[dict(name="x", usd=1500)]), open(costs_p, "w"))
    a = accounting.summarize(dict(equity=4950, cash=4950, trades=[], holdings={}), csv_p, costs_p)
    assert abs(a["max_drawdown"] - 0.01) < 1e-9
    assert a["operating"]["support"]["hist_max_dd"] == 0.2375


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} journal/agent tests passed")
