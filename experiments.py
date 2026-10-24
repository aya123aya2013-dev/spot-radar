#!/usr/bin/env python3
"""Weekly experiment runner: the project's learning loop.

Reads research_backlog.json (ideas with status "pending"), tests every idea that has a test function here on REAL
data with the same discipline as research.py (the exam is the last 30% of each chart, never used to tune anything),
and writes lab_data/experiments.json with, per idea: the average change it makes to the exam return, the share of
charts it improved, and the same numbers under costs x3.  With --synthetic the identical pipeline runs on random
walks and writes experiments_null.json: whatever "improves" there is luck, the yardstick the real result must beat.

It never changes any account or config.  The Friday review reads the result, writes the lesson into the journal
and updates the backlog status (tested / rejected / candidate); a candidate may only enter account C through a
pull request, and A only after live proof.
"""
import json, os, sys
from datetime import datetime, timezone

import spot_lab as sl
import research as rs
import hitrate as hr
import scalp
import policy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sl.OUT
BACKLOG = os.path.join(HERE, "research_backlog.json")
EXAM_FRAC = 0.7
D_COINS = ("BTC", "ETH", "SOL")


# ---------------------------------------------------------------- helpers
def charts():
    """One chart per C coin: its current C strategy if there is one, else a fixed trend baseline.  Plus a 1d
    supertrend baseline for every coin so a filter is judged on more than the strategies we happened to pick."""
    cp = os.path.join(OUT, "config_C.json")
    cfg = json.load(open(cp)) if os.path.exists(cp) else {}
    out = []
    # coins the owner excluded (BNB) or that the screening site has not reviewed (POL) are not used as test charts
    unreviewed = {c.upper() for c in policy.load().get("unreviewed", [])}
    for coin in policy.allowed(rs.C_COINS):
        if coin.upper() in unreviewed:
            continue
        p = cfg.get(coin)
        if p and p.get("strategy") in rs.ALL:
            out.append((coin, p["tf"], p["strategy"], p["params"], p.get("stop", 0)))
        out.append((coin, "1d", "supertrend", dict(n=14, m=3), 0))
    return out


def exam_window(rows):
    n = len(rows)
    return int(n * EXAM_FRAC), n


def run_variant(rows, en, ex, stop, mask=None, cost_mult=1.0):
    """Exam return of a strategy, optionally with entries masked (True = entry allowed)."""
    at = sl.atr(rows)
    en2 = [e and (mask is None or mask[i]) for i, e in enumerate(en)]
    s, e = exam_window(rows)
    old = sl.FEE, sl.SLIP
    sl.FEE, sl.SLIP = old[0] * cost_mult, old[1] * cost_mult
    try:
        return rs.sim(rows, en2, ex, stop, at, s, e)
    finally:
        sl.FEE, sl.SLIP = old


def summarize(deltas, deltas_x3):
    if not deltas:
        return dict(charts=0)
    return dict(charts=len(deltas),
                mean_delta=sum(deltas) / len(deltas),
                share_improved=sum(d > 0 for d in deltas) / len(deltas),
                mean_delta_x3=sum(deltas_x3) / len(deltas_x3) if deltas_x3 else None)


# ---------------------------------------------------------------- entry masks (True = entry allowed)
def mask_weekend(rows):
    return [datetime.fromtimestamp(r[0] / 1000, timezone.utc).weekday() < 5 for r in rows]


def mask_volume(rows, mult=1.5, n=20):
    v = [r[5] for r in rows]
    out = []
    for i in range(len(v)):
        if i < n:
            out.append(True)
            continue
        avg = sum(v[i - n:i]) / n
        out.append(v[i] >= mult * avg if avg > 0 else True)
    return out


_BTC_DAILY = {}


def mask_btc_regime(rows):
    """Alt entries only while BTC's daily close is above its 200-day EMA (yesterday's value, no look-ahead)."""
    if not _BTC_DAILY:
        try:
            d = sl.klines("BTC", "1d")
        except Exception:
            return None
        c = [r[4] for r in d]
        e = sl.ema(c, 200)
        for i, r in enumerate(d):
            _BTC_DAILY[int(r[0] // 86400000)] = c[i] > e[i] if e[i] else True
    out = []
    for r in rows:
        day = int(r[0] // 86400000) - 1
        out.append(_BTC_DAILY.get(day, True))
    return out


# ---------------------------------------------------------------- account-D exit variants
def run_trail(rows, en, trail, sp, hold, start, end, cost_mult=1.0):
    """Like hitrate.run but no fixed target: a trailing stop `trail` below the highest high since entry."""
    fee, slip = sl.FEE * cost_mult, sl.SLIP * cost_mult
    rets, i = [], start
    while i < end - hold - 2:
        if not en[i]:
            i += 1
            continue
        px = rows[i + 1][1] * (1 + slip)
        hi_seen, stp, exit_px, j = px, px * (1 - sp), None, i + 1
        for j in range(i + 1, i + 1 + hold):
            lo, hi = rows[j][3], rows[j][2]
            if lo <= stp:
                exit_px = stp * (1 - slip)
                break
            hi_seen = max(hi_seen, hi)
            stp = max(stp, hi_seen * (1 - trail))
        if exit_px is None:
            j = i + hold
            exit_px = rows[j][4] * (1 - slip)
        rets.append(exit_px * (1 - fee) / (px * (1 + fee)) - 1)
        i = j + 1
    return rets


def d_entries(rows):
    a, b = scalp.signal_series(rows)
    return [bool(x or y) for x, y in zip(a, b)]


def mean(x):
    return sum(x) / len(x) if x else 0.0


# ---------------------------------------------------------------- the tests
def test_filter(mask_fn):
    deltas, d3, per = [], [], []
    for coin, tf, school, params, stop in charts():
        try:
            rows = sl.klines(coin, tf)
            if len(rows) < 900:
                continue
            fn = rs.ALL[school][0]
            en, ex = fn(rows, **params)
            m = mask_fn(rows)
            if m is None:
                continue
            base, var = run_variant(rows, en, ex, stop), run_variant(rows, en, ex, stop, m)
            b3, v3 = run_variant(rows, en, ex, stop, cost_mult=rs.COST_STRESS), run_variant(rows, en, ex, stop, m, rs.COST_STRESS)
            deltas.append(var["ret"] - base["ret"])
            d3.append(v3["ret"] - b3["ret"])
            per.append(dict(coin=coin, tf=tf, school=school, base=base["ret"], variant=var["ret"], n_base=base["n"], n_variant=var["n"]))
        except Exception as e:
            print(f"! {coin} {tf} {school}: {e}")
    s = summarize(deltas, d3)
    s["per_chart"] = per
    return s


def test_d_exits():
    """Account D: fixed target 2% / stop 3% / 24h (current rules) against a 2% trailing stop and against a wider
    target 3%.  Same entries, same exam window, mean net return per trade."""
    out = {"fixed_2_3": [], "trail_2": [], "fixed_3_3": [], "fixed_2_3_x3": [], "trail_2_x3": []}
    n_tr = 0
    for coin in D_COINS:
        try:
            rows = sl.klines(coin, "1h")
            if len(rows) < 900:
                continue
            en = d_entries(rows)
            s, e = exam_window(rows)
            out["fixed_2_3"] += hr.run(rows, en, 0.02, 0.03, 24, s, e)
            out["fixed_3_3"] += hr.run(rows, en, 0.03, 0.03, 24, s, e)
            out["trail_2"] += run_trail(rows, en, 0.02, 0.03, 24, s, e)
            out["fixed_2_3_x3"] += hr.run(rows, en, 0.02, 0.03, 24, s, e, cost_mult=rs.COST_STRESS)
            out["trail_2_x3"] += run_trail(rows, en, 0.02, 0.03, 24, s, e, cost_mult=rs.COST_STRESS)
            n_tr += 1
        except Exception as ex:
            print(f"! D {coin}: {ex}")
    res = {k: dict(n=len(v), mean=mean(v), win=(sum(x > 0 for x in v) / len(v)) if v else 0.0) for k, v in out.items()}
    res["charts"] = n_tr
    res["mean_delta"] = res["trail_2"]["mean"] - res["fixed_2_3"]["mean"]
    res["mean_delta_x3"] = res["trail_2_x3"]["mean"] - res["fixed_2_3_x3"]["mean"]
    res["share_improved"] = None
    return res


def test_dca_ruler():
    """Not a strategy: a second measuring stick.  Buying BTC in equal weekly instalments over the exam window
    versus putting everything in on day one."""
    rows = sl.klines("BTC", "1d")
    s, e = exam_window(rows)
    closes = [r[4] for r in rows[s:e]]
    weeks = list(range(0, len(closes), 7))
    per = 1.0 / len(weeks)
    qty = sum(per / closes[i] for i in weeks)
    dca = qty * closes[-1] - 1
    lump = closes[-1] / closes[0] - 1
    return dict(charts=1, dca_ret=dca, lump_ret=lump, mean_delta=dca - lump, share_improved=None, mean_delta_x3=None,
                days=len(closes))


# ---------------------------------------------------------------- account-B variant: weekly instead of daily rebalance
def mom_backtest_weekly(days, opn, close, e50, e100b, p, start, end, every=7, cost_mult=1.0):
    old = sl.FEE, sl.SLIP
    sl.FEE, sl.SLIP = old[0] * cost_mult, old[1] * cost_mult
    try:
        cash, hold, eq, trades = sl.CAPITAL, {}, [], 0
        for t in range(start, end - 1):
            price = {c: close[c][t] for c in close if close[c][t]}
            eq.append(cash + sum(q * price.get(c, 0) for c, q in hold.items()))
            if (t - start) % every:
                continue
            tgt = sl.mom_targets(close, e50, e100b, t, p)
            if set(tgt) != set(hold):
                nxt = {c: opn[c][t + 1] for c in close if opn[c][t + 1]}
                cash, hold, fills = sl.rebalance(cash, hold, tgt, nxt, expo=p.get("expo", 1.0))
                trades += len(fills)
        last_px = {c: close[c][end - 1] for c in close if close[c][end - 1]}
        final = cash + sum(q * last_px.get(c, 0) * (1 - sl.FEE) for c, q in hold.items())
        pk, dd = 0.0, 0.0
        for x in eq:
            pk = max(pk, x)
            dd = max(dd, 1 - x / pk) if pk else 0
        return dict(ret=final / sl.CAPITAL - 1, dd=dd, n=trades)
    finally:
        sl.FEE, sl.SLIP = old


def test_momentum_weekly():
    """Account B: the chosen rotation parameters, rebalanced daily (current) against weekly, on the exam window."""
    cp = os.path.join(OUT, "config_B.json")
    p = (json.load(open(cp)).get("params") if os.path.exists(cp) else None) or dict(lookback=60, top=3, regime=1, expo=0.4)
    uni = sl.load_universe(1000)
    if "BTC" not in uni or len(uni) < 5:
        return dict(charts=0, error="not enough data")
    days, opn, close = sl.align(uni)
    e50, e100b = sl.mom_prepare(close)
    s, e = int(len(days) * EXAM_FRAC), len(days)
    daily = mom_backtest_weekly(days, opn, close, e50, e100b, p, s, e, every=1)
    weekly = mom_backtest_weekly(days, opn, close, e50, e100b, p, s, e, every=7)
    d3 = mom_backtest_weekly(days, opn, close, e50, e100b, p, s, e, every=1, cost_mult=rs.COST_STRESS)
    w3 = mom_backtest_weekly(days, opn, close, e50, e100b, p, s, e, every=7, cost_mult=rs.COST_STRESS)
    b0, b1 = close["BTC"][s], close["BTC"][e - 1]
    return dict(charts=1, params=p, daily=daily, weekly=weekly, btc_hold=(b1 / b0 - 1) if b0 and b1 else None,
                mean_delta=weekly["ret"] - daily["ret"], mean_delta_x3=w3["ret"] - d3["ret"], share_improved=None,
                dd_delta=weekly["dd"] - daily["dd"], trades_daily=daily["n"], trades_weekly=weekly["n"])


# ---------------------------------------------------------------- sizing variant: inverse-volatility targeting
VOL_TARGET = 0.015     # size so that one ATR move costs about 1.5% of the slice


def frac_vol_target(atr_val, price):
    if price <= 0 or atr_val <= 0:
        return 0.25
    return max(0.1, min(1.0, VOL_TARGET / (atr_val / price)))


def test_vol_target_sizing():
    """Same entries and exits, different position size: current rule (3% risk per 2.5 ATR, floor 25%) against a
    plain inverse-volatility target with a lower floor.  Judged on exam return and drawdown."""
    deltas, d3, dds, per = [], [], [], []
    for coin, tf, school, params, stop in charts():
        try:
            rows = sl.klines(coin, tf)
            if len(rows) < 900:
                continue
            fn = rs.ALL[school][0]
            en, ex = fn(rows, **params)
            base, b3 = run_variant(rows, en, ex, stop), run_variant(rows, en, ex, stop, cost_mult=rs.COST_STRESS)
            orig = sl.position_fraction
            sl.position_fraction = frac_vol_target
            try:
                var, v3 = run_variant(rows, en, ex, stop), run_variant(rows, en, ex, stop, cost_mult=rs.COST_STRESS)
            finally:
                sl.position_fraction = orig
            deltas.append(var["ret"] - base["ret"])
            d3.append(v3["ret"] - b3["ret"])
            dds.append(var["dd"] - base["dd"])
            per.append(dict(coin=coin, tf=tf, school=school, base=base["ret"], variant=var["ret"], dd_base=base["dd"], dd_variant=var["dd"]))
        except Exception as ex_:
            print(f"! {coin} {tf} {school}: {ex_}")
    s = summarize(deltas, d3)
    s["mean_dd_delta"] = mean(dds)
    s["per_chart"] = per
    return s


def mask_low_vol(rows, n=14, look=100):
    """Entries only while ATR(14)/close is at or below its median of the last 100 bars (calm market)."""
    at = sl.atr(rows, n)
    rel = [a / r[4] if r[4] else 0.0 for a, r in zip(at, rows)]
    out = []
    for i in range(len(rel)):
        if i < look:
            out.append(True)
            continue
        w = sorted(rel[i - look:i])
        out.append(rel[i - 1] <= w[len(w) // 2])   # previous bar's value: no look-ahead
    return out


def test_d_hold12():
    """Account D: same entries and 2% / 3% levels, but exit after 12 hours instead of 24 (shorter exposure)."""
    out = {"hold_24": [], "hold_12": [], "hold_24_x3": [], "hold_12_x3": []}
    n_tr = 0
    for coin in D_COINS:
        try:
            rows = sl.klines(coin, "1h")
            if len(rows) < 900:
                continue
            en = d_entries(rows)
            s, e = exam_window(rows)
            out["hold_24"] += hr.run(rows, en, 0.02, 0.03, 24, s, e)
            out["hold_12"] += hr.run(rows, en, 0.02, 0.03, 12, s, e)
            out["hold_24_x3"] += hr.run(rows, en, 0.02, 0.03, 24, s, e, cost_mult=rs.COST_STRESS)
            out["hold_12_x3"] += hr.run(rows, en, 0.02, 0.03, 12, s, e, cost_mult=rs.COST_STRESS)
            n_tr += 1
        except Exception as ex:
            print(f"! D {coin}: {ex}")
    res = {k: dict(n=len(v), mean=mean(v), win=(sum(x > 0 for x in v) / len(v)) if v else 0.0) for k, v in out.items()}
    res["charts"] = n_tr
    res["mean_delta"] = res["hold_12"]["mean"] - res["hold_24"]["mean"]
    res["mean_delta_x3"] = res["hold_12_x3"]["mean"] - res["hold_24_x3"]["mean"]
    res["share_improved"] = None
    return res


TESTS = {
    "weekend_filter": lambda: test_filter(mask_weekend),
    "volume_confirm": lambda: test_filter(mask_volume),
    "btc_regime_for_alts": lambda: test_filter(mask_btc_regime),
    "d_trailing_exit": test_d_exits,
    "dca_ruler": test_dca_ruler,
    "low_vol_entry": lambda: test_filter(mask_low_vol),
    "d_hold_12": test_d_hold12,
    "momentum_weekly": test_momentum_weekly,
    "vol_target_sizing": test_vol_target_sizing,
}


def main():
    try:
        backlog = json.load(open(BACKLOG))
    except Exception:
        backlog = {"items": []}
    todo = [it for it in backlog.get("items", []) if it.get("status") == "pending" and it.get("id") in TESTS]
    results, started = {}, datetime.now(timezone.utc)
    for it in todo:
        print(f"== {it['id']}")
        try:
            results[it["id"]] = TESTS[it["id"]]()
            r = results[it["id"]]
            md = r.get("mean_delta")
            print(f"   charts {r.get('charts')}  mean delta {md*100:+.2f}%  improved {r.get('share_improved')}  x3 {r.get('mean_delta_x3')}" if md is not None else "   no result")
        except Exception as e:
            results[it["id"]] = dict(error=str(e)[:200])
            print("   failed:", e)
    out = dict(generated=started.isoformat(), synthetic=sl.SYN, exam="last 30% of each chart, never used for tuning",
               costs=dict(fee=sl.FEE, slip=sl.SLIP, stress=rs.COST_STRESS), results=results)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "experiments_null.json" if sl.SYN else "experiments.json"), "w"), indent=1)
    print("wrote", "experiments_null.json" if sl.SYN else "experiments.json", "with", len(results), "results")


if __name__ == "__main__":
    main()
