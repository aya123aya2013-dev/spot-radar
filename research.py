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
C_COINS = ["BTC", "ETH", "SOL", "OP", "POL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC"]   # state keys; excluded coins stay cash (policy.py)
MAX_TOP_SHARE = 0.7        # the best single trade may not be more than 70% of the exam profit
COST_STRESS = 3.0          # profit must survive fees and slippage multiplied by this
TREND_SCHOOLS = ("ema_cross", "donchian", "macd_trend", "supertrend", "ichimoku", "squeeze_break", "obv_trend",
                 "structure_bos", "composite", "vote")
C_MAX_EXPOSURE = 0.6       # account C never holds more than 60% of its equity in coins at once
WF_K = 5                   # walk-forward: 5 consecutive test blocks, training always expands from the start


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


_VOTE_CACHE = {}
VOTE_PANEL = [("ema_cross", dict(fast=12, slow=50)), ("donchian", dict(n=40)), ("macd_trend", dict(trend=100)),
              ("supertrend", dict(n=10, m=3)), ("ichimoku", dict(t=9, k=26, b=52)), ("obv_trend", dict(n=20, m=50)),
              ("structure_bos", dict(k=5))]


def _in_pos(en, ex):
    s, out = False, []
    for a, b in zip(en, ex):
        s = (s and not b) or (not s and a)
        out.append(s)
    return out


def s_vote(rows, k):
    """Buy when at least k of 7 schools are holding a long at the same time; sell when fewer than k are."""
    key = (len(rows), rows[0][0], rows[0][1], rows[-1][4])
    if key not in _VOTE_CACHE:
        _VOTE_CACHE.clear()
        _VOTE_CACHE[key] = [_in_pos(*ALL[name][0](rows, **p)) for name, p in VOTE_PANEL]
    states = _VOTE_CACHE[key]
    votes = [sum(st[i] for st in states) for i in range(len(rows))]
    return [i > 0 and votes[i] >= k > votes[i - 1] for i in range(len(rows))], [v < k for v in votes]


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
    "vote": (s_vote, [dict(k=k) for k in (2, 3, 4)]),
}
ALL = {**sl.STRATS, **NEW}
STOPS = sl.STOPS


# ---------------------------------------------------------------- simulator that also returns trade returns
def sim(rows, en, ex, stop, at, start, end):
    cash, qty, basis, peak, frac = sl.SLEEVE, 0.0, 0.0, 0.0, 1.0
    cash0, rets, eq, closed, ent = cash, [], [], [], 0
    for i in range(start, end - 1):
        c = rows[i][4]
        if qty > 0:
            peak = max(peak, c)
            if ex[i] or (stop and c < peak - stop * at[i]):
                pr = qty * rows[i + 1][1] * (1 - sl.SLIP) * (1 - sl.FEE)
                cash += pr
                rets.append((pr / basis - 1) * frac)
                closed.append((ent, rets[-1]))
                qty = 0.0
        elif en[i] and cash > 0:
            frac = sl.position_fraction(at[i], c)
            invest = cash * frac
            basis, peak, ent = invest, c, i
            qty = invest * (1 - sl.FEE) / (rows[i + 1][1] * (1 + sl.SLIP))
            cash -= invest
        eq.append(cash + qty * c)
    final = cash + qty * rows[end - 1][4] * (1 - sl.FEE)
    pk, dd = 0.0, 0.0
    for e in eq:
        pk = max(pk, e)
        dd = max(dd, 1 - e / pk) if pk else 0
    return dict(ret=final / cash0 - 1, dd=dd, n=len(rets), rets=rets, trades=closed, bh=rows[end - 1][4] / rows[start][4] - 1)


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


# ---------------------------------------------------------------- external data used as filters (reported, never selected on)
def load_fng():
    """{ 'YYYY-MM-DD': value 0..100 } from alternative.me, or {} when unreachable."""
    try:
        if sl.SYN:
            rnd = random.Random(3)
            return {datetime.fromtimestamp(t * 86400, timezone.utc).strftime("%Y-%m-%d"): rnd.randint(10, 90)
                    for t in range(int(__import__("time").time() // 86400) - 3000, int(__import__("time").time() // 86400) + 1)}
        d = sl.http_json("https://api.alternative.me/fng/?limit=0&format=json")["data"]
        return {datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).strftime("%Y-%m-%d"): int(x["value"]) for x in d}
    except Exception as e:
        print("  fear&greed unavailable:", e)
        return {}


def load_funding(coin):
    """[(time_ms, rate)] from Binance futures funding history, or [] when unreachable."""
    try:
        if sl.SYN:
            rnd = random.Random(sum(map(ord, coin)))
            now = int(__import__("time").time() * 1000)
            return [(now - (2000 - i) * 8 * 3600 * 1000, rnd.gauss(0.0001, 0.0002)) for i in range(2000)]
        try:
            d = sl.http_json(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={coin}USDT&limit=1000")
            return [(int(x["fundingTime"]), float(x["fundingRate"])) for x in d]
        except Exception as e:
            print(f"  funding {coin}: Binance unreachable ({str(e)[:40]}), trying OKX public (about 3 months)")
        out, after = [], ""
        for _ in range(6):                                    # OKX serves 100 rows a page, newest first
            d = sl.http_json(f"https://www.okx.com/api/v5/public/funding-rate-history?instId={coin}-USDT-SWAP&limit=100{after}")
            rows = d.get("data") or []
            if not rows:
                break
            out += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in rows]
            after = f"&after={rows[-1]['fundingTime']}"
        return sorted(out)
    except Exception as e:
        print(f"  funding {coin} unavailable:", str(e)[:60])
        return []


def load_stable():
    """{YYYY-MM-DD: total stablecoin supply in USD} from DefiLlama (public), or {} when unreachable."""
    try:
        if sl.SYN:
            rnd, v, out = random.Random(3), 1.5e11, {}
            for i in range(1500):
                v *= 1 + rnd.gauss(0.0003, 0.002)
                out[datetime.fromtimestamp(int(__import__("time").time()) - (1500 - i) * 86400, timezone.utc).strftime("%Y-%m-%d")] = v
            return out
        d = sl.http_json("https://stablecoins.llama.fi/stablecoincharts/all")
        out = {}
        for x in d:
            tot = (x.get("totalCirculatingUSD") or x.get("totalCirculating") or {}).get("peggedUSD")
            if tot:
                out[datetime.fromtimestamp(int(x["date"]), timezone.utc).strftime("%Y-%m-%d")] = float(tot)
        return out
    except Exception as e:
        print("  stablecoin supply unavailable:", str(e)[:60])
        return {}


def mask_stable(rows, stable):
    """True where entries stay allowed: stablecoin supply (dry powder) grew over the 7 days up to yesterday."""
    if not stable:
        return None
    ok = []
    for r in rows:
        d1 = datetime.fromtimestamp(r[0] / 1000 - 86400, timezone.utc).strftime("%Y-%m-%d")
        d8 = datetime.fromtimestamp(r[0] / 1000 - 8 * 86400, timezone.utc).strftime("%Y-%m-%d")
        a_, b_ = stable.get(d1), stable.get(d8)
        ok.append(True if a_ is None or b_ is None else a_ > b_)
    return ok


def mask_fng(rows, fng, limit):
    """True where entries stay allowed: yesterday's fear&greed reading is below `limit` (not too greedy)."""
    ok, last = [], 50
    for r in rows:
        day = datetime.fromtimestamp(r[0] / 1000 - 86400, timezone.utc).strftime("%Y-%m-%d")
        last = fng.get(day, last)
        ok.append(last < limit)
    return ok


def mask_funding(rows, fund, limit):
    import bisect
    if not fund:
        return None
    ts = [t for t, _ in fund]
    ok = []
    for r in rows:
        j = bisect.bisect_right(ts, r[0]) - 1
        ok.append(True if j < 0 else fund[j][1] < limit)
    return ok


def regimes(rows):
    """bull / bear / side for every bar, from that bar's own EMA50 and EMA200 (no look-ahead)."""
    c = [r[4] for r in rows]
    e50, e200 = sl.ema(c, 50), sl.ema(c, 200)
    out = []
    for i in range(len(c)):
        if i < 200:
            out.append(None)
        elif c[i] > e200[i] and e50[i] > e200[i]:
            out.append("bull")
        elif c[i] < e200[i] and e50[i] < e200[i]:
            out.append("bear")
        else:
            out.append("side")
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else 0.0


def portfolio_stats(daily, window=180):
    """How much the coins move together. N_eff is how many independent bets the coins really are."""
    coins = [c for c in C_COINS if c in daily]
    if len(coins) < 3:
        return None
    common = sorted(set.intersection(*[set(t for t, _ in daily[c]) for c in coins]))[-(window + 1):]
    if len(common) < 60:
        return None
    ret = {}
    for c in coins:
        m = dict(daily[c])
        px = [m[t] for t in common]
        ret[c] = [px[i] / px[i - 1] - 1 for i in range(1, len(px))]
    pairs = [(a, b, pearson(ret[a], ret[b])) for i, a in enumerate(coins) for b in coins[i + 1:]]
    avg = sum(p[2] for p in pairs) / len(pairs)
    n = len(coins)
    n_eff = n / (1 + (n - 1) * max(avg, 0))
    hi = max(pairs, key=lambda p: p[2])
    return dict(coins=n, window_days=len(common) - 1, avg_corr=avg, n_eff=n_eff, max_pair=[hi[0], hi[1], hi[2]])


def exec_report():
    """Real order-book snapshot: what a market buy of our order size would cost versus the 0.05% slippage assumed."""
    if sl.SYN:
        return {}
    out = {}
    for coin in C_COINS:
        try:
            d = sl.api(f"/api/v3/depth?symbol={coin}USDT&limit=100")
            asks = [(float(p), float(q)) for p, q in d["asks"]]
            bids = [(float(p), float(q)) for p, q in d["bids"]]
            mid = (asks[0][0] + bids[0][0]) / 2
            res = dict(spread_bps=(asks[0][0] - bids[0][0]) / mid * 1e4)
            for label, notional in (("sleeve", sl.CAPITAL / len(C_COINS)), ("1000", 1000.0)):
                left, cost, qty = notional, 0.0, 0.0
                for p, q in asks:
                    take = min(left, p * q)
                    cost += take
                    qty += take / p
                    left -= take
                    if left <= 1e-9:
                        break
                res[f"slip_bps_{label}"] = ((cost / qty) / mid - 1) * 1e4 if qty and left <= 1e-9 else None
            out[coin] = res
        except Exception as e:
            print(f"  depth {coin} unavailable:", str(e)[:60])
    return out


def stress_sim(rows, en, ex, stop, at, start, end):
    f, s = sl.FEE, sl.SLIP
    sl.FEE, sl.SLIP = f * COST_STRESS, s * COST_STRESS
    try:
        return sim(rows, en, ex, stop, at, start, end)
    finally:
        sl.FEE, sl.SLIP = f, s


def top_share(rets):
    pos = sum(r for r in rets if r > 0)
    return (max(rets) / pos) if pos > 0 else 1.0


# ---------------------------------------------------------------- research backtest -> config_C.json
def coin_slip(coin, base, okx):
    """Slippage per side for this coin: the model default, or 1.5x the live OKX measurement for 1000$ when that is higher
    (only coins that okx_check.py measures, and only with at least 50 samples). Costs can only go up, never down."""
    try:
        a = ((okx or {}).get("avg") or {}).get(coin) or {}
        if a.get("samples", 0) >= 50 and a.get("okx_slip_bps") is not None:
            return max(base, 1.5 * a["okx_slip_bps"] / 1e4)
    except Exception:
        pass
    return base


def cmd_research():
    base_slip = sl.SLIP
    okx = None
    try:
        okx = json.load(open(os.path.join(OUT, "okx_check.json")))
    except Exception:
        pass
    series, family_rows, family_sel = [], {}, {}
    fng = load_fng()
    filt_rows = {"fng<75": [], "fng<60": [], "funding<0.03%": [], "stable_growth_7d": []}
    stable = load_stable()
    matrix, gate_rows, wf_rows, regime_now, daily = {}, [], [], {}, {}
    for coin in C_COINS:
        fund = load_funding(coin)
        sl.SLIP = coin_slip(coin, base_slip, okx)
        if sl.SLIP != base_slip:
            print(f"  {coin}: slippage assumption raised to {sl.SLIP*1e4:.1f} bps per side (live OKX measurement)")
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
            wb = [sl.WARMUP + (n - sl.WARMUP) * k // (WF_K + 1) for k in range(WF_K + 2)]
            wf_all = []
            rg = regimes(rows)
            if tf == "1d":
                regime_now[coin] = rg[-1]
                daily[coin] = [(r[0], r[4]) for r in rows]
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
                        wf = []
                        for k in range(1, WF_K + 1):
                            a_ = sim(rows, en, ex, stop, at, sl.WARMUP, wb[k])
                            b_ = sim(rows, en, ex, stop, at, wb[k], wb[k + 1])
                            wf.append((a_["ret"] - 0.5 * a_["dd"] if a_["n"] >= 5 else None, b_["ret"], b_["n"], b_["bh"]))
                        wf_all.append(wf)
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
                family_sel.setdefault(name, []).append(c["score"])
            series.append(dict(coin=coin, tf=tf, cands=cands[:8], tests=tests, median_hold=med,
                               base_rate=(sum(x > 0 for x in allhold) / len(allhold)) if allhold else 0,
                               bh=cands[0]["hold"]["bh"] if cands else 0))
            top = cands[0] if cands else None
            if top:
                en, ex = ALL[top["strategy"]][0](rows, **top["params"])
                base = sim(rows, en, ex, top["stop"], at, sel, n)["ret"]
                for label, mk in (("fng<75", mask_fng(rows, fng, 75) if fng else None),
                                  ("fng<60", mask_fng(rows, fng, 60) if fng else None),
                                  ("funding<0.03%", mask_funding(rows, fund, 0.0003)),
                                  ("stable_growth_7d", mask_stable(rows, stable))):
                    if mk is None:
                        continue
                    en2 = [a and b for a, b in zip(en, mk)]
                    filt_rows[label].append(sim(rows, en2, ex, top["stop"], at, sel, n)["ret"] - base)
                for c in cands[:8]:
                    e1, x1 = ALL[c["strategy"]][0](rows, **c["params"])
                    st = stress_sim(rows, e1, x1, c["stop"], at, sel, n)
                    c["stress_ret"], c["top_share"] = st["ret"], top_share(c["hold"]["rets"])
            # ---- school x market-regime matrix, and a regime gate trained ONLY on the selection folds
            for name, c in fam_best.items():
                en_f, ex_f = ALL[name][0](rows, **c["params"])
                by = {}
                for ei, r in sim(rows, en_f, ex_f, c["stop"], at, edges[0], sel)["trades"]:
                    if rg[ei]:
                        by.setdefault(rg[ei], []).append(r)
                allowed = {k for k, v in by.items() if len(v) >= 3 and sum(v) / len(v) > 0}
                for ei, r in c["hold"]["trades"]:
                    if rg[ei]:
                        matrix.setdefault(name, {}).setdefault(rg[ei], []).append(r)
                en_g = [a and rg[i] in allowed for i, a in enumerate(en_f)]
                gate_rows.append(sim(rows, en_g, ex_f, c["stop"], at, sel, n)["ret"] - c["hold"]["ret"])
            # ---- walk-forward of the selection process itself
            chain, chain_hold, nb, npos = 1.0, 1.0, 0, 0
            for k in range(WF_K):
                pool = [(w[k][0], w[k][1]) for w in wf_all if w[k][0] is not None]
                if not pool:
                    continue
                best_ = max(pool, key=lambda x: x[0])
                chain *= 1 + best_[1]
                chain_hold *= 1 + wf_all[0][k][3]
                nb += 1
                npos += best_[1] > 0
            if nb:
                wf_rows.append(dict(coin=coin, tf=tf, chain=chain - 1, hold=chain_hold - 1, blocks=nb, pos=npos))
            print(f"{coin:4} {tf:3} tests {tests:4}  picked {top['strategy'] if top else '-':15} "
                  f"exam {top['hold']['ret']*100:+6.1f}%  (median of all {med*100:+5.1f}%)" if top else
                  f"{coin:4} {tf:3} tests {tests:4}  nothing passed selection")

    sl.SLIP = base_slip
    # School-level prior fixed IN ADVANCE from the literature, not from our data (our own selection-fold scores
    # are positive for almost every school, because the best of many versions always looks good in-sample, and
    # the exam period is shared by all coins, so using it would leak). Spot is long-only: momentum / trend
    # schools are eligible for account C; mean-reversion ones are still tested and reported but not picked.
    school_sel = {k: sum(v) / len(v) for k, v in family_sel.items() if v}
    allowed_schools = set(TREND_SCHOOLS) & set(school_sel)
    print("schools eligible for account C (trend prior):", sorted(allowed_schools))
    print("schools tested but not eligible:", sorted(set(school_sel) - allowed_schools))
    picks = {}
    for coin in C_COINS:
        good = []
        for s in (x for x in series if x["coin"] == coin):
            for c in s["cands"]:
                h = c["hold"]
                bp = boot_profit(h["rets"])
                if (c["strategy"] in allowed_schools and h["ret"] > 0 and h["n"] >= MIN_TRADES_EXAM and bp >= BOOT_P and h["ret"] > s["median_hold"]
                        and c.get("stress_ret", -1) > 0 and c.get("top_share", 1) <= MAX_TOP_SHARE):
                    good.append((c["score"], s, c, bp))
        good.sort(key=lambda x: x[0], reverse=True)

        def pack(s, c, bp, active=True):
            return dict(active=active, tf=s["tf"], strategy=c["strategy"], params=c["params"], stop=c["stop"],
                        test_ret=c["hold"]["ret"], test_bh=c["hold"]["bh"], test_dd=c["hold"]["dd"],
                        test_n=c["hold"]["n"], boot_p=bp, tests=s["tests"], base_rate=s["base_rate"],
                        stress_ret=c.get("stress_ret"), top_share=c.get("top_share"),
                        test_rets=[round(x, 5) for x in c["hold"]["rets"]])
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
    act = [p for k, p in picks.items() if k != "generated" and p["active"] and p.get("test_rets")]
    mc = None
    if act:
        rnd, outs = random.Random(11), []
        for _ in range(2000):
            tot = 0.0
            for p in act:
                g = 1.0
                for _ in p["test_rets"]:
                    g *= 1 + rnd.choice(p["test_rets"])
                tot += g - 1
            outs.append(tot / len(C_COINS))
        outs.sort()
        mc = dict(p05=outs[100], p50=outs[1000], p95=outs[1900], p_loss=sum(x < 0 for x in outs) / len(outs), coins=len(act))
    filters = [dict(filter=k, charts=len(v), mean_delta=sum(v) / len(v), share_improved=sum(x > 0 for x in v) / len(v))
               for k, v in filt_rows.items() if v]
    reg_out = {name: {r: dict(n=len(v), mean=sum(v) / len(v), win=sum(x > 0 for x in v) / len(v)) for r, v in d.items()}
               for name, d in matrix.items()}
    gate = dict(charts=len(gate_rows), mean_delta=sum(gate_rows) / len(gate_rows),
                share_improved=sum(x > 0 for x in gate_rows) / len(gate_rows)) if gate_rows else None
    wf_sum = None
    if wf_rows:
        wf_sum = dict(charts=len(wf_rows), mean_chain=sum(r["chain"] for r in wf_rows) / len(wf_rows),
                      mean_hold=sum(r["hold"] for r in wf_rows) / len(wf_rows),
                      share_positive=sum(r["chain"] > 0 for r in wf_rows) / len(wf_rows),
                      share_beat_hold=sum(r["chain"] > r["hold"] for r in wf_rows) / len(wf_rows),
                      blocks_positive=sum(r["pos"] for r in wf_rows) / max(1, sum(r["blocks"] for r in wf_rows)))
    port = portfolio_stats(daily)
    execr = exec_report()
    actp = [p for k, p in picks.items() if k != "generated" and p.get("active")]
    bench = dict(active=len(actp),
                 mean_pick=(sum(p["test_ret"] for p in actp) / len(actp)) if actp else None,
                 mean_hold=(sum(p["test_bh"] for p in actp) / len(actp)) if actp else None,
                 share_beat_hold=(sum(p["test_ret"] > p["test_bh"] for p in actp) / len(actp)) if actp else None,
                 btc_hold=picks["BTC"].get("test_bh"), btc_pick=picks["BTC"].get("test_ret"), btc_active=bool(picks["BTC"].get("active")))
    json.dump(dict(generated=picks["generated"], benchmark=bench, leaderboard=board, filters=filters, account_mc=mc,
                   regime_now=regime_now, regime_matrix=reg_out, regime_gate=gate, walk_forward=wf_sum,
                   portfolio=port, execution=execr, max_exposure=C_MAX_EXPOSURE,
                   school_prior={k: round(v, 4) for k, v in sorted(school_sel.items(), key=lambda x: -x[1])}, allowed_schools=sorted(allowed_schools),
                   n_coins=len(C_COINS),
                   note="best-by-selection pick of each school on each chart, judged on the unseen last 30%"),
              open(os.path.join(OUT, "research_C.json"), "w"), indent=1)
    print("\nSchool leaderboard (unseen exam, mean over charts where the school passed selection):")
    for r in board:
        print(f"  {r['school']:15} {r['mean_exam']*100:+6.1f}%  positive in {r['share_positive']*100:3.0f}% of {r['charts']} charts")
    for f in filters:
        print(f"  filter {f['filter']:14} changes the exam return by {f['mean_delta']*100:+.2f} points on average, "
              f"better on {f['share_improved']*100:.0f}% of {f['charts']} charts")
    if gate:
        print(f"  regime gate (trained on selection folds only): exam change {gate['mean_delta']*100:+.2f} points on average, "
              f"better on {gate['share_improved']*100:.0f}% of {gate['charts']} school-chart pairs")
    if wf_sum:
        print(f"  walk-forward ({WF_K} blocks): chained {wf_sum['mean_chain']*100:+.1f}% vs hold {wf_sum['mean_hold']*100:+.1f}%, "
              f"positive on {wf_sum['share_positive']*100:.0f}% of {wf_sum['charts']} charts, blocks positive {wf_sum['blocks_positive']*100:.0f}%")
    if port:
        print(f"  coins move together: avg correlation {port['avg_corr']:.2f}, {port['coins']} coins behave like {port['n_eff']:.1f} independent bets")
    for c_, r_ in execr.items():
        print(f"  exec {c_:5} spread {r_['spread_bps']:.1f} bps, slip for the sleeve: {r_.get('slip_bps_sleeve')}")
    if mc:
        print(f"  account Monte Carlo (resampled exam trades): 5% {mc['p05']*100:+.1f}%  median {mc['p50']*100:+.1f}%  "
              f"95% {mc['p95']*100:+.1f}%  chance of loss {mc['p_loss']*100:.0f}%")
    print("\nAccount C picks:")
    for c in C_COINS:
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
    sl.COINS = C_COINS
    sl.MAX_EXPOSURE = C_MAX_EXPOSURE
    sl.SLEEVE = sl.CAPITAL / len(C_COINS)
    sl.STATE = os.path.join(OUT, "paper_C.json")
    if os.path.exists(sl.STATE):
        old = json.load(open(sl.STATE))
        if sorted(old.get("cash", {})) != sorted(C_COINS):      # coin set changed: restart C (it has only paper money)
            print("  account C: coin set changed, restarting its virtual $5000")
            os.remove(sl.STATE)
    sl.HEADERS = {"paper_trades.csv": ["time", "coin", "side", "price", "result", "reason"],
                  "paper_equity.csv": ["time", "equity", "buy_and_hold"]}
    orig = sl.log_csv
    sl.log_csv = lambda name, row: orig(name.replace(".csv", "_C.csv"), row)
    sl.HEADERS.update({"paper_trades_C.csv": sl.HEADERS["paper_trades.csv"],
                       "paper_equity_C.csv": sl.HEADERS["paper_equity.csv"]})
    print("--- account C (research lab)")
    init = {}
    if not os.path.exists(sl.STATE):
        for c in sl.COINS:
            init[c] = sl.last_price(c, sl.klines(c, "1h", 50))
    st = sl.load_state(init)
    st = sl.cycle(cfg, st)
    json.dump(st, open(sl.STATE, "w"), indent=1)


if __name__ == "__main__":
    cmd_research()
