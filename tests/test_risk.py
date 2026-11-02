"""Risk invariants, tested directly.  Run: python -m pytest -q tests  (or python tests/test_risk.py)."""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import risk

DAY = risk.DAY
NOW = 1_800_000_000_000
CFG = dict(risk.DEFAULTS)


def acct(**kw):
    a = dict(equity=5000.0, peak=5000.0, day_start_eq=5000.0, week_start_eq=5000.0, halt_until=0, streak_until=0)
    a.update(kw)
    return a


def ctx(**kw):
    c = dict(equity=5000.0, cash=5000.0, budget=1.0, held_values={}, n_positions=0, rets={}, cooldowns={}, now_ms=NOW)
    c.update(kw)
    return c


def cand(**kw):
    c = dict(coin="ETH", price=100.0, stop=95.0, weight=0.25, spread_bps=1.0, slip_bps=1.0, min_qty=0.001, lot=0.001)
    c.update(kw)
    return c


# ---------------------------------------------------------------- global gates
def test_clean_state_allows_entries():
    ok, why = risk.global_gates(CFG, acct(), NOW, 2.0, 1.0)
    assert ok and why == []


def test_live_mode_is_refused():
    ok, why = risk.global_gates(dict(CFG, mode="live"), acct(), NOW, 2.0, 1.0)
    assert not ok and "live_refused" in why


def test_emergency_stop_blocks_entries():
    ok, why = risk.global_gates(dict(CFG, emergency_stop=True), acct(), NOW, 2.0, 1.0)
    assert not ok and "emergency_stop" in why


def test_no_entry_during_drawdown_halt():
    ok, why = risk.global_gates(CFG, acct(halt_until=NOW + DAY), NOW, 2.0, 1.0)
    assert not ok and "drawdown_halt" in why


def test_daily_and_weekly_loss_limits():
    ok, why = risk.global_gates(CFG, acct(equity=4840.0), NOW, 2.0, 1.0)        # -3.2% today
    assert not ok and "daily_loss_limit" in why
    ok, why = risk.global_gates(CFG, acct(equity=4690.0, day_start_eq=4690.0), NOW, 2.0, 1.0)   # -6.2% this week
    assert not ok and "weekly_loss_limit" in why


def test_stale_or_incomplete_data_blocks_entries():
    ok, why = risk.global_gates(CFG, acct(), NOW, 31.0, 1.0)
    assert not ok and "data_stale" in why
    ok, why = risk.global_gates(CFG, acct(), NOW, None, 1.0)
    assert not ok and "data_stale" in why
    ok, why = risk.global_gates(CFG, acct(), NOW, 2.0, 0.5)
    assert not ok and "data_incomplete" in why


def test_drawdown_breach():
    assert risk.drawdown_breach(CFG, 4250.0, 5000.0)
    assert not risk.drawdown_breach(CFG, 4300.0, 5000.0)


def test_loss_streak_counter():
    assert risk.losing_streak([0.05, -0.01, -0.02, -0.03, -0.01]) == 4
    assert risk.losing_streak([-0.01, 0.02]) == 0


# ---------------------------------------------------------------- sizing
def test_risk_per_trade_caps_size():
    usd, why = risk.size_entry(CFG, cand(stop=90.0, weight=0.25), ctx())      # 10% stop: 1% risk = $50 -> $500
    assert why == [] and abs(usd - 500.0) < 1.0


def test_max_position_caps_size():
    usd, why = risk.size_entry(CFG, cand(stop=99.0, weight=0.9), ctx())       # 1% stop would allow $5000
    assert why == [] and usd <= 0.25 * 5000.0 + 1e-6


def test_stop_too_far_rejected():
    usd, why = risk.size_entry(CFG, cand(stop=85.0), ctx())
    assert usd == 0 and "stop_too_far" in why


def test_invalid_stop_rejected():
    usd, why = risk.size_entry(CFG, cand(stop=101.0), ctx())
    assert usd == 0 and "invalid_stop" in why


def test_spread_and_slippage_limits():
    assert "spread_too_wide" in risk.size_entry(CFG, cand(spread_bps=50.0), ctx())[1]
    assert "slippage_too_high" in risk.size_entry(CFG, cand(slip_bps=40.0), ctx())[1]


def test_asset_cooldown_after_stop():
    usd, why = risk.size_entry(CFG, cand(), ctx(cooldowns={"ETH": NOW + DAY}))
    assert usd == 0 and "asset_cooldown" in why


def test_position_count_and_budget():
    assert "max_positions" in risk.size_entry(CFG, cand(), ctx(n_positions=6))[1]
    assert "budget_full" in risk.size_entry(CFG, cand(), ctx(budget=0.0))[1]
    usd, _ = risk.size_entry(CFG, cand(stop=99.0, weight=0.9), ctx(budget=0.10))
    assert usd <= 500.0 + 1e-6


def test_correlated_exposure_cap():
    a = [0.01 * ((i % 7) - 3) for i in range(60)]
    rets = {"ETH": a, "SOL": [x * 1.1 for x in a]}
    usd, why = risk.size_entry(CFG, cand(), ctx(held_values={"SOL": 2000.0}, rets=rets))
    assert usd == 0 and "correlated_exposure" in why
    usd, why = risk.size_entry(CFG, cand(stop=99.0, weight=0.9), ctx(held_values={"SOL": 1500.0}, rets=rets))
    assert why == [] and usd <= 0.40 * 5000.0 - 1500.0 + 1e-6


def test_never_more_than_cash():
    usd, why = risk.size_entry(CFG, cand(stop=99.0, weight=0.9), ctx(cash=300.0))
    assert usd <= 300.0 + 1e-6


def test_exchange_minimum_and_lot_rounding():
    usd, why = risk.size_entry(dict(CFG, min_order_usd=10.0), cand(stop=99.0, weight=0.001), ctx())
    assert usd == 0 and why == ["below_exchange_minimum"]
    usd, why = risk.size_entry(CFG, cand(stop=90.0, lot=1.0), ctx())        # $500 at $100 with lot 1 -> exactly 5 units
    assert abs(usd - 500.0) < 1e-6


def test_cannot_sell_more_than_owned():
    assert risk.can_sell(2.0, 5.0) == 2.0
    assert risk.can_sell(2.0, -1.0) == 0.0


def test_edge_gate():
    table = {"bull": dict(n=40, mean=0.02), "early": dict(n=40, mean=-0.01), "turn": dict(n=5, mean=-0.2)}
    assert risk.edge_eligible(CFG, table, "bull")[0]
    assert not risk.edge_eligible(CFG, table, "early")[0]
    assert risk.edge_eligible(CFG, table, "turn")[0]            # too few trades for a verdict
    assert risk.edge_eligible(CFG, {}, "bull")[0]


def test_control_file_cannot_enable_live(tmp_path=None):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "control.json")
    json.dump({"mode": "live", "emergency_stop": False}, open(p, "w"))
    cfg = risk.load_control(p)
    ok, why = risk.global_gates(cfg, acct(), NOW, 2.0, 1.0)
    assert not ok and "live_refused" in why


def test_repository_control_file_is_paper():
    cfg = risk.load_control()
    assert cfg["mode"] == "paper"


def test_no_exchange_order_code_exists():
    """There must be no code path that can send an order to an exchange (no private endpoints, no signing)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    banned = ("/api/v5/trade", "/api/v3/order", "OK-ACCESS-SIGN", "X-MBX-APIKEY", "hmac.new(")
    for f in os.listdir(root):
        if f.endswith(".py"):
            src = open(os.path.join(root, f)).read()
            for b in banned:
                assert b not in src, f"{f} contains {b}"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
    print(f"{n} risk tests passed")
