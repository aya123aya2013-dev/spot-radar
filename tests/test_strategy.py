"""Strategy parameters, the ML filter and the research helpers behave as documented."""
import os, random, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SPOT_LAB_DATA", tempfile.mkdtemp())
if "--synthetic" not in sys.argv:
    sys.argv.append("--synthetic")
import agent, market as mk, risk


def S(c=110.0, e20=105.0, e50=100.0, e200=90.0, breakout=True, resume=False, ret63=0.3, near_high=-0.01):
    return dict(c=c, e20=e20, e50=e50, e200=e200, atr=3.0, vol=0.03, ret63=ret63, near_high=near_high, breakout=breakout,
                resume=resume, low=c - 2, high=c + 2)


def snaps():
    return {"BTC": S(ret63=0.1), "AAA": S(), "BBB": S(breakout=False, resume=True), "CCC": S()}


def test_defaults_are_the_original_rules():
    p = agent.DEFAULT_PARAMS
    assert (p["watch_n"], p["max_pos"], p["slot"], p["init_atr"], p["trail_atr"], p["partial_r"], p["breakout_n"]) == (8, 6, 4, 2.5, 3.0, 2.0, 20)
    assert p["resume"] and p["exit"] == "trail" and not p["pyramid"] and not p["ml"] and p["time_stop_days"] == 0


def test_resume_entries_can_be_switched_off():
    on = agent.plan(snaps(), {}, 5000, {}, prm=dict(agent.DEFAULT_PARAMS))
    off = agent.plan(snaps(), {}, 5000, {}, prm=dict(agent.DEFAULT_PARAMS, resume=False))
    assert "BBB" in [e[0] for e in on["entries"]] and "BBB" not in [e[0] for e in off["entries"]]


def test_filter_blocks_and_reports():
    d = agent.plan(snaps(), {}, 5000, {}, prm=dict(agent.DEFAULT_PARAMS), entry_filter=lambda f: False)
    assert d["entries"] == [] and set(d["filtered"]) >= {"AAA", "CCC"}


def test_untradable_coins_never_lead_but_count_for_breadth():
    d = agent.plan(snaps(), {}, 5000, {}, prm=dict(agent.DEFAULT_PARAMS), tradable={"BTC", "AAA", "BBB"})
    assert "CCC" not in d["watch"] and "CCC" not in [e[0] for e in d["entries"]] and d["breadth"] == 1.0


def test_entry_features_have_the_documented_length():
    d = agent.plan(snaps(), {}, 5000, {}, prm=dict(agent.DEFAULT_PARAMS))
    assert d["feats"] and all(len(f) == len(agent.FEATURES) for f in d["feats"].values())


def test_logit_learns_a_clear_pattern_and_filter_uses_base_rate():
    rnd = random.Random(1)
    X = [[rnd.gauss(0, 1) for _ in range(len(agent.FEATURES))] for _ in range(400)]
    y = [1 if x[0] > 0 else 0 for x in X]
    m = agent.logit_fit(X, y)
    acc = sum((agent.logit_prob(m, x) >= 0.5) == bool(t) for x, t in zip(X, y)) / len(X)
    assert acc > 0.85
    f = agent.EntryFilter([(100, m)])
    good, bad = [2.0] + [0.0] * (len(agent.FEATURES) - 1), [-2.0] + [0.0] * (len(agent.FEATURES) - 1)
    assert f(50, bad) is True                      # before the model exists: no verdict, the signal passes
    assert f(150, good) is True and f(150, bad) is False


def test_filter_never_trains_on_too_few_or_unfinished_trades():
    tr = [dict(feat=[0.0] * len(agent.FEATURES), ret=0.1, why="end") for _ in range(100)]
    assert agent.train_filter(tr) is None


def test_edge_tables_are_walk_forward():
    tabs = [(10, {"bull": {"n": 30, "mean": 0.01}}), (20, {"bull": {"n": 40, "mean": -0.01}})]
    assert agent.edge_at(tabs, 5) == {} and agent.edge_at(tabs, 15)["bull"]["mean"] == 0.01 and agent.edge_at(tabs, 25)["bull"]["mean"] == -0.01


def _run(**prm):
    data = {c: mk.synthetic(c, "1d", 1200) for c in agent.COINS[:12]}
    days = [r[0] // agent.DAY for r in data["BTC"]]
    return agent.backtest_run(data, days[260], days[-1], prm=dict(agent.DEFAULT_PARAMS, **prm), risk_cfg=risk.load_control(), edge_table={})


def test_partial_off_means_no_partial_sales():
    _, tr, _, acc = _run(partial_r=0.0)
    assert tr and all(t["why"] != "partial" for t in tr)


def test_time_stop_and_pyramid_and_donchian_run():
    _, tr, _, _ = _run(time_stop_days=5, time_stop_r=3.0)
    assert any(t["why"] == "time" for t in tr)
    _, _, _, acc = _run(pyramid=True, pyramid_r=0.5)
    assert acc["adds"] > 0
    c, tr, _, _ = _run(exit="donchian", exit_low_n=10)
    assert tr and c[-1][1] > 0


def test_monte_carlo_is_a_range():
    curve = [(i, 1000 * (1.0005 ** i) * (1 + 0.01 * ((i % 7) - 3) / 3), 0) for i in range(800)]
    mc = agent.monte_carlo(curve, paths=200)
    assert mc["cagr_p5"] <= mc["cagr_p50"] <= mc["cagr_p95"] and 0 <= mc["dd_p50"] <= mc["dd_p95"] <= 1


def test_default_rules_are_frozen():
    """Guard for every future change: the original rules (DEFAULT_PARAMS) must give exactly these results on fixed
    synthetic data.  A new parameter has to keep its default behaviour identical, or this fails."""
    coins = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC", "DOT", "TRX", "ATOM"]
    data = {c: mk.synthetic(c, "1d", 1500) for c in coins}
    days = [r[0] // agent.DAY for r in data["BTC"]]
    cfg = dict(risk.DEFAULTS, max_stop=0.20)
    c, tr, _, _ = agent.backtest_run(data, days[260], days[-1], prm=dict(agent.DEFAULT_PARAMS), risk_cfg=cfg, edge_table={})
    c2, tr2, _, _ = agent.backtest_run(data, days[260], days[-1], prm=dict(agent.DEFAULT_PARAMS))
    assert (round(c[-1][1], 4), len(tr)) == (9622.6335, 307)
    assert (round(c2[-1][1], 4), len(tr2)) == (9003.3516, 329)


def test_regime_band_keeps_the_previous_side_near_the_ema200():
    btc = dict(c=99.0, e50=101.0, e200=100.0)            # 1% under EMA200, EMA50 above it
    assert agent.regime(btc, 0.6)[0] == "weak"            # no band: below EMA200 means weak
    assert agent.regime(btc, 0.6, band=0.02, prev="bull")[0] == "bull"   # inside the 2% band: stays bull
    assert agent.regime(btc, 0.6, band=0.02, prev="weak")[0] == "weak"
    far = dict(c=95.0, e50=101.0, e200=100.0)             # 5% under: outside the band, flips
    assert agent.regime(far, 0.6, band=0.02, prev="bull")[0] == "weak"


def test_no_weekend_entries_when_asked():
    sat = 2 + 7 * 2900                                     # day 2 was a Saturday (1970-01-03)
    assert agent.weekday(sat) == 5 and agent.weekday(sat + 2) == 0
    on = agent.plan(snaps(), {}, 5000, {}, prm=dict(agent.DEFAULT_PARAMS, no_weekend_entry=True), exec_day=sat)
    off = agent.plan(snaps(), {}, 5000, {}, prm=dict(agent.DEFAULT_PARAMS, no_weekend_entry=True), exec_day=sat + 2)
    assert on["entries"] == [] and off["entries"]


def test_volume_confirmation():
    sn = snaps()
    sn["AAA"] = dict(sn["AAA"], vol_ratio=2.0)
    sn["CCC"] = dict(sn["CCC"], vol_ratio=1.0)
    d = agent.plan(sn, {}, 5000, {}, prm=dict(agent.DEFAULT_PARAMS, vol_confirm=1.5))
    got = [e[0] for e in d["entries"]]
    assert "AAA" in got and "CCC" not in got

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} strategy tests passed")
