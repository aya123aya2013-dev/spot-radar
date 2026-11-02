#!/usr/bin/env python3
"""Performance accounting for the agent, from its state and equity log.  Fees and slippage are recorded per fill, once
(the fill price already includes slippage; the fee is charged on top), so nothing is counted twice.
Ratios are only reported with enough history: Sharpe/Sortino/Calmar need 30 daily points."""
import csv, json, math, os
from datetime import datetime, timezone

CAPITAL = 5000.0


def daily_points(csv_path):
    """Last equity reading of each UTC day, with the BTC price beside it: [(day, equity, btc)]."""
    by = {}
    if not os.path.exists(csv_path):
        return []
    for r in list(csv.reader(open(csv_path)))[1:]:
        try:
            d = r[0][:10]
            by[d] = (d, float(r[1]), float(r[2]) if len(r) > 2 and r[2] else None)
        except (ValueError, IndexError):
            pass
    return [by[k] for k in sorted(by)]


def ratios(points):
    if len(points) < 30:
        return dict(sharpe=None, sortino=None, calmar=None, note="needs 30 days")
    eq = [p[1] for p in points]
    r = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
    m = sum(r) / len(r)
    sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
    dn = math.sqrt(sum(min(0.0, x) ** 2 for x in r) / len(r))
    pk, dd = 0.0, 0.0
    for v in eq:
        pk = max(pk, v)
        dd = max(dd, 1 - v / pk)
    yrs = len(r) / 365.0
    cagr = (eq[-1] / eq[0]) ** (1 / yrs) - 1 if eq[0] > 0 else None
    return dict(sharpe=m / sd * math.sqrt(365) if sd > 0 else None, sortino=m / dn * math.sqrt(365) if dn > 0 else None,
                calmar=cagr / dd if cagr is not None and dd > 0 else None)


def summarize(st, csv_path, costs_path=None):
    sells = [t for t in st.get("trades", []) if t.get("side") == "SELL"]
    buys = [t for t in st.get("trades", []) if t.get("side") == "BUY"]
    realized = sum(t.get("pnl_usd", 0.0) for t in sells)
    unreal = sum(h.get("pnl_usd", 0.0) for h in (st.get("holdings") or {}).values())
    fees = sum(t.get("fee_usd", 0.0) for t in st.get("trades", []))
    slip = sum(t.get("slip_usd", 0.0) for t in st.get("trades", []))
    held = sum(h.get("value", 0.0) for h in (st.get("holdings") or {}).values())
    eq = st.get("equity", CAPITAL)
    full = [t for t in sells if t.get("frac", 1.0) >= 0.999 or t.get("closed")]
    pos_pnl = {}
    for t in sells:                                      # whole-position results (partials + final)
        pos_pnl.setdefault(t.get("pos_id") or (t["coin"] + t.get("entry_time", "")), []).append(t.get("pnl_usd", 0.0))
    wins = [t for t in sells if t.get("pnl_usd", 0) > 0]
    losses = [t for t in sells if t.get("pnl_usd", 0) <= 0]
    gw, gl = sum(t["pnl_usd"] for t in wins), -sum(t["pnl_usd"] for t in losses)
    pts = daily_points(csv_path)
    pk, dd, cur_dd = 0.0, 0.0, 0.0
    for p in pts:
        pk = max(pk, p[1])
        dd = max(dd, 1 - p[1] / pk)
    if pts:
        cur_dd = 1 - pts[-1][1] / max(p[1] for p in pts)
    btc0 = next((p[2] for p in pts if p[2]), None)
    btc1 = pts[-1][2] if pts else None
    def period(days):
        if len(pts) < 2:
            return None
        base = pts[max(0, len(pts) - 1 - days)][1]
        return pts[-1][1] / base - 1
    traded = sum(t.get("usd", 0.0) for t in buys) + sum(t.get("usd", 0.0) for t in sells)
    costs = []
    try:
        costs = json.load(open(costs_path)).get("monthly", []) if costs_path else []
    except Exception:
        pass
    monthly_cost = sum(float(c.get("usd", 0)) for c in costs)
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    realized_month = sum(t.get("pnl_usd", 0.0) for t in sells if str(t.get("time", "")).startswith(month))
    support = None
    try:
        bt = json.load(open(os.path.join(os.path.dirname(csv_path), "backtest_G.json")))
        best = bt["challenger"] if bt.get("promotion", {}).get("challenger") else bt["champion"]
        cagr, dd = best["full"].get("cagr"), best["full"].get("max_dd")
        if cagr and monthly_cost > 0:
            support = dict(hist_cagr=cagr, hist_max_dd=dd, monthly_at_cagr=eq * cagr / 12,
                           capital_needed=monthly_cost * 12 / cagr, covers=eq * cagr / 12 >= monthly_cost)
    except Exception:
        pass
    out = dict(
        equity=eq, cash=st.get("cash"), invested=held, exposure=held / eq if eq else 0.0,
        realized_pnl=realized, unrealized_pnl=unreal, total_pnl=eq - CAPITAL,
        fees=fees, slippage=slip, turnover=traded,
        trades_closed=len(sells), win_rate=(len(wins) / len(sells)) if sells else None,
        avg_win=(gw / len(wins)) if wins else None, avg_loss=(-gl / len(losses)) if losses else None,
        expectancy_usd=(realized / len(sells)) if sells else None, profit_factor=(gw / gl) if gl > 0 else None,
        max_drawdown=dd, drawdown_now=cur_dd,
        day=period(1), week=period(7), month=period(30),
        btc_hold=(btc1 / btc0 - 1) if btc0 and btc1 else None, cash_benchmark=0.0,
        days=len(pts), **ratios(pts),
        operating=dict(monthly_cost=monthly_cost, items=costs, realized_this_month=realized_month,
                       coverage=(realized_month / monthly_cost) if monthly_cost > 0 else None,
                       remaining=(monthly_cost - realized_month) if monthly_cost > 0 else None,
                       paper=True, support=support))
    return out
