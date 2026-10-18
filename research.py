#!/usr/bin/env python3
"""Research lab (account C): many more analysis schools, judged much more strictly.

Account A (live paper account) is never touched by this file. C has its own state, config and logs.

Schools added on top of A's six: Ichimoku, rolling VWAP, market structure (BOS), Wyckoff spring,
Fibonacci pullback, volume profile (POC / value area), trend pullback, volatility squeeze,
Supertrend, OBV trend.

Validation (stricter than A):
  * first 70% of history is split into 3 sequential folds, a candidate must win >= 2 of 3 folds
    and trade at least 8 times in total;
  * the last 30% is never used for selection. It is the exam;
  * a pick is active only if the exam return is positive, it traded >= 5 times, a bootstrap of its
    own trades says P(profit) >= 0.7, and it beats the median of every combination tried on that
    chart (otherwise it is no better than luck from trying many combinations);
  * the file also writes a leaderboard of which schools actually held up on unseen data.
"""
import json, math, os, random, sys
from datetime import datetime, timezone

import spot_lab as sl

OUT = sl.OUT
FOLDS, SEL_FRAC = 3, 0.7
MIN_TRADES_SEL, MIN_TRADES_EXAM, BOOT_P = 8, 5, 0.7


# ---------------------------------------------------------------- helpers
def win(a, n, fn):
    """Rolling window ending at the current bar (inclusive)."""
    return [None if i < n - 1 else fn(a[i - n + 1:i + 1]) for i in range(len(a))]


def mid_hl(rows, n):
    hh, ll = win([r[2] for r in rows], n, max), win([r[3] for r in rows], n, min)
    return [None if a is None else (a + b) / 2 for a, b in zip(hh, ll)]


# ---------------------------------------------------------------- new schools: each returns (entry[], exit[])
def s_ichimoku(rows, t, k, b):
    c = [r[4] for r in rows]
    T, K, B = mid_hl(rows, t), mid_hl(rows, k), mid_hl(rows, b)
    A = [None if T[i] is None or K[i] is None else (T[i] + K[i]) / 2 for i in range(len(c))]
    n = len(c)
    en, ex = [False] * n, [False] * n

    def top(i):                                   # the cloud at bar i was drawn k bars earlier
        j = i - k
        return None if j < 0 or A[j] is None or B[j] is None else max(A[j], B[j])
    for i in range(1, n):
        tp, tp0 = top(i), top(i - 1)
        if tp is None or K[i] is None:
            continue
        up = c[i] > tp and T[i] > K[i]
        up0 = tp0 is not None and K[i - 1] is not None and c[i - 1] > tp0 and T[i - 1] > K[i - 1]
        en[i] = up and not up0
        ex[i] = c[i] < K[i]
    return en, ex


def s_vwap(rows, n, k):
    c = [r[4] for r in rows]
    tp = [(r[2] + r[3] + r[4]) / 3 for r in rows]
    v = [r[5] for r in rows]
    en, ex = [False] * len(c), [False] * len(c)
    for i in range(n - 1, len(c)):
        vs = v[i - n + 1:i + 1]
        sv = sum(vs) or 1.0
        vw = sum(a * b for a, b in zip(tp[i - n + 1:i + 1], vs)) / sv
        sd = math.sqrt(sum(b * (a - vw) ** 2 for a, b in zip(tp[i - n + 1:i + 1], vs)) / sv)
        en[i] = sd > 0 and c[i] < vw - k * sd
        ex[i] = c[i] > vw
    return en, ex


def s_structure(rows, k):
    """Break of structure: price closes above the last swing high while swing lows are rising."""
    h, l, c = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows]
    n = len(c)
    en, ex = [False] * n, [False] * n
    shi, slo = [], []
    for i in range(2 * k, n):
        j = i - k                                  # a pivot at j is only known k bars later
        if h[j] == max(h[j - k:i + 1]):
            shi.append(h[j])
        if l[j] == min(l[j - k:i + 1]):
            slo.append(l[j])
        if len(shi) >= 2 and len(slo) >= 2:
            en[i] = slo[-1] > slo[-2] and c[i] > shi[-1] >= c[i - 1]
            ex[i] = c[i] < slo[-1]
    return en, ex


def s_wyckoff(rows, n, vm):
    """Spring: price dips under a tight range on a volume spike, then closes back inside."""
    h, l, c, v = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows]
    at, vs = sl.atr(rows), sl.sma(v, 20)
    N = len(c)
    en, ex = [False] * N, [False] * N
    for i in range(max(n, 20), N):
        lo, hi = min(l[i - n:i]), max(h[i - n:i])
        en[i] = (l[i] < lo < c[i] and vs[i] and v[i] > vm * vs[i] and (hi - lo) <= 10 * at[i])
        ex[i] = c[i] > (hi + lo) / 2
    return en, ex


def s_fib(rows, n, band):
    """Buy the 0.382-0.618 (or 0.5-0.786) retracement of the last up-leg while the trend is up."""
    o, h, l, c = ([r[j] for r in rows] for j in (1, 2, 3, 4))
    e100, at = sl.ema(c, 100), sl.atr(rows)
    N = len(c)
    en, ex = [False] * N, [False] * N
    for i in range(max(n, 100), N):
        wh, wl = h[i - n:i], l[i - n:i]
        hi, lo = max(wh), min(wl)
        leg = hi - lo
        if leg < 2 * at[i]:
            continue
        up_leg = wl.index(lo) < wh.index(hi)
        if up_leg and c[i] > e100[i]:
            en[i] = hi - band[1] * leg <= c[i] <= hi - band[0] * leg and c[i] > o[i] and c[i] > c[i - 1]
            ex[i] = c[i] >= hi or c[i] < hi - 0.886 * leg
    return en, ex


def s_volprofile(rows, n, mode, bins=20):
    h, l, c, v = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows]
    tp = [(r[2] + r[3] + r[4]) / 3 for r in rows]
    N = len(c)
    en, ex = [False] * N, [False] * N
    poc = [None] * N
    val = [None] * N
    vah = [None] * N
    for i in range(n, N):
        lo, hi = min(l[i - n:i]), max(h[i - n:i])
        if hi <= lo:
            continue
        w = (hi - lo) / bins
        vol = [0.0] * bins
        for j in range(i - n, i):
            vol[min(bins - 1, int((tp[j] - lo) / w))] += v[j]
        p = max(range(bins), key=vol.__getitem__)
        a = b = p
        acc, tot = vol[p], sum(vol)
        while acc < 0.7 * tot and (a > 0 or b < bins - 1):
            up = vol[b + 1] if b < bins - 1 else -1
            dn = vol[a - 1] if a > 0 else -1
            if up >= dn:
                b += 1
                acc += vol[b]
            else:
                a -= 1
                acc += vol[a]
        poc[i], val[i], vah[i] = lo + (p + 0.5) * w, lo + a * w, lo + (b + 1) * w
    for i in range(n + 1, N):
        if poc[i] is None or poc[i - 1] is None:
            continue
        if mode == "revert":            # price re-enters the value area from below, target the POC
            en[i] = c[i - 1] < val[i - 1] <= c[i] < poc[i]
            ex[i] = c[i] >= poc[i]
        else:                           # accepted above the value area: continuation
            en[i] = c[i - 1] <= vah[i - 1] < c[i]
            ex[i] = c[i] < poc[i]
    return en, ex


def s_pullback(rows, thr, out):
    c = [r[4] for r in rows]
    R, e50, e200 = sl.rsi(c), sl.ema(c, 50), sl.ema(c, 200)
    N = len(c)
    en = [i > 0 and R[i] is not None and R[i - 1] is not None and c[i] > e200[i] and e50[i] > e200[i]
          and R[i - 1] < thr <= R[i] for i in range(N)]
    ex = [R[i] is not None and (R[i] > out or c[i] < e50[i]) for i in range(N)]
    return en, ex


def s_squeeze(rows, pct):
    c = [r[4] for r in rows]
    mid, up, lo = sl.boll(c, 20, 2)
    N = len(c)
    bw = [None if up[i] is None else (up[i] - lo[i]) / mid[i] for i in range(N)]
    en, ex = [False] * N, [False] * N
    for i in range(140, N):
        hist = sorted(x for x in bw[i - 126:i - 6] if x is not None)
        if not hist:
            continue
        thr = hist[int(pct * (len(hist) - 1))]
        squeezed = min(x for x in bw[i - 6:i] if x is not None) <= thr
        en[i] = squeezed and c[i] > up[i] and c[i - 1] <= up[i - 1]
        ex[i] = c[i] < mid[i]
    return en, ex


def s_supertrend(rows, n, m):
    h, l, c = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows]
    at = sl.atr(rows, n)
    N = len(c)
    fu, fl, d = [0.0] * N, [0.0] * N, [1] * N
    for i in range(N):
        ub, lb = (h[i] + l[i]) / 2 + m * at[i], (h[i] + l[i]) / 2 - m * at[i]
        if i == 0:
            fu[i], fl[i] = ub, lb
            continue
        fu[i] = ub if ub < fu[i - 1] or c[i - 1] > fu[i - 1] else fu[i - 1]
        fl[i] = lb if lb > fl[i - 1] or c[i - 1] < fl[i - 1] else fl[i - 1]
        d[i] = 1 if c[i] > fu[i - 1] else -1 if c[i] < fl[i - 1] else d[i - 1]
    return [i > 0 and d[i] == 1 and d[i - 1] == -1 for i in range(N)], [d[i] == -1 for i in range(N)]


def s_obv(rows, n, m):
    c, v = [r[4] for r in rows], [r[5] for r in rows]
    obv, s = [], 0.0
    for i in range(len(c)):
        if i:
            s += v[i] if c[i] > c[i - 1] else -v[i] if c[i] < c[i - 1] else 0
        obv.append(s)
    eo, ec = sl.ema(obv, n), sl.ema(c, m)
    ok = [obv[i] > eo[i] and c[i] > ec[i] for i in range(len(c))]
    return [i > 0 and ok[i] and not ok[i - 1] for i in range(len(c))], [c[i] < ec[i] for i in range(len(c))]


NEW = {
    "ichimoku": (s_ichimoku, [dict(t=9, k=26, b=52), dict(t=20, k=60, b=120)]),
    "vwap_revert": (s_vwap, [dict(n=n, k=k) for n in (48, 120) for k in (1.5, 2.0)]),
    "structure_bos": (s_structure, [dict(k=k) for k in (3, 5, 8)]),
    "wyckoff_spring": (s_wyckoff, [dict(n=n, vm=vm) for n in (20, 40) for vm in (1.2, 1.8)]),
    "fib_pullback": (s_fib, [dict(n=n, band=b) for n in (30, 60) for b in ((0.382, 0.618), (0.5, 0.786))]),
    "vol_profile": (s_volprofile, [dict(n=n, mode=m) for n in (100, 200) for m in ("revert", "breakout")]),
    "trend_pullback": (s_pullback, [dict(thr=t, out=o) for t in (40, 45) for o in (65, 75)]),
    "squeeze_break": (s_squeeze, [dict(pct=p) for p in (0.1, 0.2, 0.3)]),
    "supertrend": (s_supertrend, [dict(n=n, m=m) for n, m in ((10, 2), (10, 3), (14, 3), (14, 4), (20, 3), (20, 4))]),
    "obv_trend": (s_obv, [dict(n=n, m=m) for n in (20, 50) for m in (50, 100)]),
}
ALL = {**sl.STRATS, **NEW}
STOPS = sl.STOPS


# ---------------------------------------------------------------- simulator that also returns trade returns
def sim(rows, en, ex, stop, at, start, end):
    cash, qty, basis, peak, frac = sl.SLEEVE, 0.0, 0.0, 0.0, 1.0
    cash0, rets, eq = cash, [], []
    for i in range(start, end - 1):
        c = rows[i][4]
        if qty > 0:
            peak = max(peak, c)
            if ex[i] or (stop and c < peak - stop * at[i]):
                pr = qty * rows[i + 1][1] * (1 - sl.SLIP) * (1 - sl.FEE)
                cash += pr
                rets.append((pr / basis - 1) * frac)
                qty = 0.0
        elif en[i] and cash > 0:
            frac = sl.position_fraction(at[i], c)
            invest = cash * frac
            basis, peak = invest, c
            qty = invest * (1 - sl.FEE) / (rows[i + 1][1] * (1 + sl.SLIP))
            cash -= invest
        eq.append(cash + qty * c)
    final = cash + qty * rows[end - 1][4] * (1 - sl.FEE)
    pk, dd = 0.0, 0.0
    for e in eq:
        pk = max(pk, e)
        dd = max(dd, 1 - e / pk) if pk else 0
    return dict(ret=final / cash0 - 1, dd=dd, n=len(rets), rets=rets, bh=rows[end - 1][4] / rows[start][4] - 1)


def boot_profit(rets, trials=1000):
    if len(rets) < 3:
        return 0.0
    rnd, ok = random.Random(7), 0
    for _ in range(trials):
        g = 1.0
        for _ in rets:
            g *= 1 + rnd.choice(rets)
        ok += g > 1
    return ok / trials


# ---------------------------------------------------------------- research backtest -> config_C.json
def cmd_research():
    series, family_rows = [], {}
    for coin in sl.COINS:
        for tf in sl.TFS:
            try:
                rows = sl.klines(coin, tf)
            except Exception as e:
                print(f"! {coin} {tf}: {e}")
                continue
            if len(rows) < 900:
                continue
            n = len(rows)
            sel = int(n * SEL_FRAC)
            edges = [sl.WARMUP + (sel - sl.WARMUP) * k // FOLDS for k in range(FOLDS + 1)]
            at = sl.atr(rows)
            cands, allhold, fam_best, tests = [], [], {}, 0
            for name, (fn, grid) in ALL.items():
                for params in grid:
                    try:
                        en, ex = fn(rows, **params)
                    except Exception as e:
                        print(f"! {name} {params}: {e}")
                        continue
                    for stop in STOPS:
                        tests += 1
                        folds = [sim(rows, en, ex, stop, at, edges[k], edges[k + 1]) for k in range(FOLDS)]
                        tot = sum(f["n"] for f in folds)
                        wins = sum(f["ret"] > 0 for f in folds)
                        mret = sum(f["ret"] for f in folds) / FOLDS
                        mdd = sum(f["dd"] for f in folds) / FOLDS
                        hold = sim(rows, en, ex, stop, at, sel, n)
                        if hold["n"]:
                            allhold.append(hold["ret"])
                        if tot < MIN_TRADES_SEL or wins < 2:
                            continue
                        c = dict(strategy=name, params=params, stop=stop, score=mret - 0.5 * mdd,
                                 folds=[f["ret"] for f in folds], hold=hold)
                        cands.append(c)
                        if name not in fam_best or c["score"] > fam_best[name]["score"]:
                            fam_best[name] = c
            cands.sort(key=lambda c: c["score"], reverse=True)
            med = sorted(allhold)[len(allhold) // 2] if allhold else 0.0
            for name, c in fam_best.items():
                family_rows.setdefault(name, []).append(c["hold"]["ret"])
            series.append(dict(coin=coin, tf=tf, cands=cands[:8], tests=tests, median_hold=med,
                               base_rate=(sum(x > 0 for x in allhold) / len(allhold)) if allhold else 0,
                               bh=cands[0]["hold"]["bh"] if cands else 0))
            top = cands[0] if cands else None
            print(f"{coin:4} {tf:3} tests {tests:4}  picked {top['strategy'] if top else '-':15} "
                  f"exam {top['hold']['ret']*100:+6.1f}%  (median of all {med*100:+5.1f}%)" if top else
                  f"{coin:4} {tf:3} tests {tests:4}  nothing passed selection")

    picks = {}
    for coin in sl.COINS:
        good = []
        for s in (x for x in series if x["coin"] == coin):
            for c in s["cands"]:
                h = c["hold"]
                bp = boot_profit(h["rets"])
                if h["ret"] > 0 and h["n"] >= MIN_TRADES_EXAM and bp >= BOOT_P and h["ret"] > s["median_hold"]:
                    good.append((c["score"], s, c, bp))
        good.sort(key=lambda x: x[0], reverse=True)

        def pack(s, c, bp, active=True):
            return dict(active=active, tf=s["tf"], strategy=c["strategy"], params=c["params"], stop=c["stop"],
                        test_ret=c["hold"]["ret"], test_bh=c["hold"]["bh"], test_dd=c["hold"]["dd"],
                        test_n=c["hold"]["n"], boot_p=bp, tests=s["tests"], base_rate=s["base_rate"])
        if good:
            _, s, c, bp = good[0]
            picks[coin] = pack(s, c, bp)
            picks[coin]["alts"] = [pack(s2, c2, bp2) for _, s2, c2, bp2 in good[1:3]]
        else:
            ranked = sorted(((s, c) for s in series if s["coin"] == coin for c in s["cands"][:1]),
                            key=lambda x: x[1]["score"], reverse=True)
            if ranked:
                s, c = ranked[0]
                picks[coin] = pack(s, c, boot_profit(c["hold"]["rets"]), False)
            else:
                picks[coin] = dict(active=False, tf="1d", strategy="ema_cross", params=dict(fast=12, slow=26),
                                   stop=0, test_ret=0, test_bh=0)
            picks[coin]["alts"] = []
    picks["generated"] = datetime.now(timezone.utc).isoformat()
    json.dump(picks, open(os.path.join(OUT, "config_C.json"), "w"), indent=1)

    board = sorted(({"school": k, "charts": len(v), "mean_exam": sum(v) / len(v),
                     "share_positive": sum(x > 0 for x in v) / len(v)} for k, v in family_rows.items()),
                   key=lambda r: r["mean_exam"], reverse=True)
    json.dump(dict(generated=picks["generated"], leaderboard=board,
                   note="best-by-selection pick of each school on each chart, judged on the unseen last 30%"),
              open(os.path.join(OUT, "research_C.json"), "w"), indent=1)
    print("\nSchool leaderboard (unseen exam, mean over charts where the school passed selection):")
    for r in board:
        print(f"  {r['school']:15} {r['mean_exam']*100:+6.1f}%  positive in {r['share_positive']*100:3.0f}% of {r['charts']} charts")
    print("\nAccount C picks:")
    for c in sl.COINS:
        p = picks[c]
        tag = "ACTIVE" if p["active"] else "cash"
        print(f"  {c:4} {tag:7} {p['strategy']} {p['tf']}  exam {p['test_ret']*100:+.1f}%  vs hold {p['test_bh']*100:+.1f}%")


# ---------------------------------------------------------------- paper account C (reuses A's engine on its own files)
def cycle_c():
    cfg_path = os.path.join(OUT, "config_C.json")
    if not os.path.exists(cfg_path):
        return
    cfg = json.load(open(cfg_path))
    sl.STRATS = ALL
    sl.STATE = os.path.join(OUT, "paper_C.json")
    sl.HEADERS = {"paper_trades.csv": ["time", "coin", "side", "price", "result", "reason"],
                  "paper_equity.csv": ["time", "equity", "buy_and_hold"]}
    orig = sl.log_csv
    sl.log_csv = lambda name, row: orig(name.replace(".csv", "_C.csv"), row)
    sl.HEADERS.update({"paper_trades_C.csv": sl.HEADERS["paper_trades.csv"],
                       "paper_equity_C.csv": sl.HEADERS["paper_equity.csv"]})
    print("--- account C (research lab)")
    init = {}
    for c in sl.COINS:
        rows = sl.klines(c, "1h", 400)
        init[c] = sl.last_price(c, rows)
    st = sl.load_state(init)
    st = sl.cycle(cfg, st)
    json.dump(st, open(sl.STATE, "w"), indent=1)


if __name__ == "__main__":
    cmd_research()
