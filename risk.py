#!/usr/bin/env python3
"""Deterministic risk engine.  The strategy PROPOSES; this module DECIDES whether a proposal is admissible and how big
it may be.  Nothing in the strategy can bypass it.  Pure functions, no I/O except reading control.json, so every rule
is unit-tested (tests/test_risk.py).

Limits are conservative paper-trading defaults, documented in control.json, owned by the owner.  They are not
"optimal" numbers; they are caps.

Global gates (no NEW entries when any fails; exits and stops always keep working):
  mode must be "paper" (live trading is not implemented and is refused), emergency_stop off, account not in a
  drawdown halt, today's and this week's loss inside the limits, not in a cooldown after consecutive losses,
  market data fresh and covering enough of the universe.
Per-entry gates:
  strategy eligible in the current regime, stop distance not above max_stop, spread and slippage inside limits,
  coin not in a cooldown after a stop, position count, correlated exposure, order at or above the exchange minimum.
Sizing: the smallest of the strategy's proposal, risk_per_trade / stop distance, max_position, and the room left in
the regime budget.
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL = os.path.join(HERE, "control.json")
DAY = 86400000

DEFAULTS = dict(
    mode="paper",                    # the only accepted value; "live" is refused (no execution code exists)
    emergency_stop=False,            # owner switch: True blocks every new entry immediately
    risk_per_trade=0.01,             # loss at the stop, as a share of equity
    max_stop=0.12,                   # a stop further than 12% below entry: trade rejected (too volatile)
    max_position=0.25,               # one position at most 25% of equity
    max_positions=6,
    max_cluster_exposure=0.40,       # positions correlated above corr_limit with each other: together at most 40%
    corr_limit=0.80,
    daily_loss_limit=0.03,           # equity down 3% from the day's start: no new entries until the next day
    weekly_loss_limit=0.06,          # down 6% from the week's start: no new entries until the next week
    max_drawdown=0.15,               # down 15% from the peak: close everything, no entries for halt_days
    halt_days=14,
    loss_streak=4,                   # 4 losing closes in a row: no new entries for streak_cooldown_days
    streak_cooldown_days=3,
    asset_cooldown_days=3,           # a coin stopped out: not re-entered for 3 days
    max_spread_bps=30.0,
    max_slip_bps=25.0,
    data_max_age_hours=30,           # the newest daily candle must be at most this old
    min_data_coverage=0.80,          # at least 80% of the universe must have fresh data
    min_order_usd=10.0,
    min_edge=0.005,                  # strategy must have earned at least +0.5% per trade, net, in this regime
    edge_min_trades=20,              # ... measured on at least 20 past trades; fewer trades: no verdict, stays eligible
)


def load_control(path=CONTROL):
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.load(open(path)).get("risk", {}))
        c = json.load(open(path))
        cfg["mode"] = c.get("mode", cfg["mode"])
        cfg["emergency_stop"] = bool(c.get("emergency_stop", cfg["emergency_stop"]))
    except Exception:
        pass
    return cfg


# ---------------------------------------------------------------- global gates
def global_gates(cfg, acct, now_ms, data_age_hours, coverage):
    """acct: dict(equity, peak, day_start_eq, week_start_eq, halt_until, streak_until).  Returns (allow_entries, reasons)."""
    reasons = []
    if cfg.get("mode") != "paper":
        reasons.append("live_refused")
    if cfg.get("emergency_stop"):
        reasons.append("emergency_stop")
    if acct.get("halt_until") and now_ms < acct["halt_until"]:
        reasons.append("drawdown_halt")
    eq = acct["equity"]
    if acct.get("day_start_eq") and eq <= acct["day_start_eq"] * (1 - cfg["daily_loss_limit"]):
        reasons.append("daily_loss_limit")
    if acct.get("week_start_eq") and eq <= acct["week_start_eq"] * (1 - cfg["weekly_loss_limit"]):
        reasons.append("weekly_loss_limit")
    if acct.get("streak_until") and now_ms < acct["streak_until"]:
        reasons.append("loss_streak_cooldown")
    if data_age_hours is None or data_age_hours > cfg["data_max_age_hours"]:
        reasons.append("data_stale")
    if coverage < cfg["min_data_coverage"]:
        reasons.append("data_incomplete")
    return (not reasons), reasons


def drawdown_breach(cfg, equity, peak):
    return peak > 0 and equity <= peak * (1 - cfg["max_drawdown"])


def losing_streak(closed_returns):
    n = 0
    for r in reversed(closed_returns):
        if r <= 0:
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------- correlation
def returns(closes, n=60):
    c = closes[-(n + 1):]
    return [c[i] / c[i - 1] - 1 for i in range(1, len(c))]


def corr(a, b):
    n = min(len(a), len(b))
    if n < 20:
        return 0.0
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def cluster_exposure(cfg, coin, held_values, rets, equity):
    """Value already held in coins correlated with `coin` above corr_limit, as a share of equity."""
    if coin not in rets:
        return 0.0
    tot = 0.0
    for k, v in held_values.items():
        if k in rets and corr(rets[coin], rets[k]) >= cfg["corr_limit"]:
            tot += v
    return tot / equity if equity > 0 else 0.0


# ---------------------------------------------------------------- per-entry decision
def size_entry(cfg, cand, ctx):
    """cand: dict(coin, price, stop, weight, spread_bps, slip_bps, eligible, min_qty, lot).
    ctx: dict(equity, cash, budget, held_values, n_positions, rets, cooldowns{coin: until_ms}, now_ms).
    Returns (usd, reasons): usd > 0 means approved with that size; otherwise reasons say why not."""
    reasons = []
    eq, price, stop = ctx["equity"], cand["price"], cand["stop"]
    if not cand.get("eligible", True):
        reasons.append("strategy_not_eligible_in_regime")
    if price <= 0 or stop <= 0 or stop >= price:
        reasons.append("invalid_stop")
        return 0.0, reasons
    dist = 1 - stop / price
    if dist > cfg["max_stop"]:
        reasons.append("stop_too_far")
    if cand.get("spread_bps") is not None and cand["spread_bps"] > cfg["max_spread_bps"]:
        reasons.append("spread_too_wide")
    if cand.get("slip_bps") is not None and cand["slip_bps"] > cfg["max_slip_bps"]:
        reasons.append("slippage_too_high")
    if ctx["cooldowns"].get(cand["coin"], 0) > ctx["now_ms"]:
        reasons.append("asset_cooldown")
    if ctx["n_positions"] >= cfg["max_positions"]:
        reasons.append("max_positions")
    held = sum(ctx["held_values"].values())
    room = ctx["budget"] * eq - held
    if room <= 0:
        reasons.append("budget_full")
    clus = cluster_exposure(cfg, cand["coin"], ctx["held_values"], ctx.get("rets", {}), eq)
    clus_room = cfg["max_cluster_exposure"] * eq - clus * eq
    if clus_room <= 0:
        reasons.append("correlated_exposure")
    if reasons:
        return 0.0, reasons
    usd = min(cand["weight"] * eq,                      # what the strategy asked for
              cfg["risk_per_trade"] * eq / dist,        # 1% of equity lost at the stop
              cfg["max_position"] * eq,
              room, clus_room, ctx["cash"])
    lot = cand.get("lot") or 0
    if lot:
        qty = math.floor(usd / price / lot + 1e-9) * lot
        usd = qty * price
    else:
        qty = usd / price
    if usd < cfg["min_order_usd"] or qty < (cand.get("min_qty") or 0):
        return 0.0, ["below_exchange_minimum"]
    return usd, []


def edge_eligible(cfg, edge_table, regime):
    """MIN EDGE gate from past (already closed) trades of the same strategy in the same regime.
    Returns (eligible, measured mean or None, n)."""
    e = (edge_table or {}).get(regime)
    if not e or e.get("n", 0) < cfg["edge_min_trades"]:
        return True, (e or {}).get("mean"), (e or {}).get("n", 0)
    return e["mean"] >= cfg["min_edge"], e["mean"], e["n"]


def edge_table_from(trades):
    """Mean net return per position by entry regime, from closed backtest trades [{'regime', 'ret', 'why'}]."""
    by = {}
    for t in trades:
        by.setdefault(t.get("regime"), []).append(t["ret"])
    return {k: dict(n=len(v), mean=sum(v) / len(v)) for k, v in by.items() if k}


def can_sell(owned_qty, qty):
    """Never sell more than is owned (spot: no shorting).  Returns the quantity that may be sold."""
    return max(0.0, min(owned_qty, qty))


REASON_AR = {
    "live_refused": "التداول الحقيقي غير مفعّل ولا يوجد له كود: الوضع ورقي فقط",
    "emergency_stop": "إيقاف الطوارئ مفعّل من المالك",
    "drawdown_halt": "قاطع الهبوط: الحساب هبط 15% من قمته",
    "daily_loss_limit": "حد الخسارة اليومي",
    "weekly_loss_limit": "حد الخسارة الأسبوعي",
    "loss_streak_cooldown": "تبريد بعد خسائر متتالية",
    "data_stale": "بيانات السوق قديمة",
    "data_incomplete": "بيانات ناقصة لجزء كبير من العملات",
    "strategy_not_eligible_in_regime": "الاستراتيجية غير مؤهلة في حالة السوق الحالية",
    "invalid_stop": "وقف غير صالح",
    "stop_too_far": "الوقف أبعد من 12%: العملة متذبذبة أكثر من اللازم",
    "spread_too_wide": "فارق العرض والطلب واسع",
    "slippage_too_high": "الانزلاق المتوقع مرتفع",
    "asset_cooldown": "العملة في فترة تبريد بعد وقف",
    "max_positions": "الحد الأقصى لعدد الصفقات",
    "budget_full": "ميزانية المخاطرة ممتلئة",
    "correlated_exposure": "تعرض مرتفع لعملات مرتبطة ببعضها",
    "below_exchange_minimum": "الحجم أقل من الحد الأدنى في المنصة",
    "insufficient_edge": "الميزة المتوقعة بعد التكاليف غير كافية في حالة السوق هذه",
    "no_entry_signal": "قائد لكن بلا إشارة دخول اليوم",
}
