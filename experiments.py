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
    for coin in rs.C_COINS:
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


TESTS = {
    "weekend_filter": lambda: test_filter(mask_weekend),
    "volume_confirm": lambda: test_filter(mask_volume),
    "btc_regime_for_alts": lambda: test_filter(mask_btc_regime),
    "d_trailing_exit": test_d_exits,
    "dca_ruler": test_dca_ruler,
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
