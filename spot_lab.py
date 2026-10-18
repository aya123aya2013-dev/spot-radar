#!/usr/bin/env python3
"""Spot Lab: strategy backtests plus a $5000 paper-trading account on real prices.

  python3 spot_lab.py backtest            download history, test strategies, write lab_data/config.json
  python3 spot_lab.py paper               paper-trade in a loop, one check every 5 minutes
  python3 spot_lab.py report              show the paper account vs buy-and-hold

Flags: --synthetic (fake data, for a code test)  --once (single paper cycle)
Needs only Python 3.8+ and an internet connection. No API key, no real orders.
"""
import csv, itertools, json, math, os, random, sys, time, urllib.request
from datetime import datetime, timezone

COINS = ["BTC", "ETH", "SOL", "OP", "POL"]
CAPITAL = 5000.0
SLEEVE = CAPITAL / len(COINS)      # each coin gets its own slice of the account
FEE, SLIP = 0.001, 0.0005          # per side: 0.1% exchange fee, 0.05% slippage
TFS = {"1h": 3600, "4h": 14400, "1d": 86400}
WARMUP = 210
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SPOT_LAB_DATA") or os.path.join(HERE, "lab_data")
os.makedirs(OUT, exist_ok=True)
SYN = "--synthetic" in sys.argv
HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]


# ---------------------------------------------------------------- data
def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "spot-lab"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def api(path):
    err = None
    for h in HOSTS:
        try:
            return http_json(h + path)
        except Exception as e:
            err = e
    raise RuntimeError(f"all data hosts failed: {err}")


def synthetic(sym, tf, n):
    rnd = random.Random(sum(map(ord, sym + tf)))
    step = TFS[tf] * 1000
    t = int(time.time() * 1000) // step * step - n * step
    p, drift, rows = 100.0, 0.0, []
    for i in range(n):
        if i % 250 == 0:
            drift = rnd.choice([-0.0015, 0.0, 0.0012, 0.0025])
        o = p
        p *= 1 + drift + rnd.gauss(0, 0.012)
        hi, lo = max(o, p) * (1 + abs(rnd.gauss(0, 0.004))), min(o, p) * (1 - abs(rnd.gauss(0, 0.004)))
        rows.append([t + i * step, o, hi, lo, p, 1000 + rnd.random() * 500])
    return rows


def klines(sym, tf, total=3000, closed_only=True):
    """[time_ms, open, high, low, close, volume]; the still-forming candle is dropped unless closed_only=False."""
    if SYN:
        rows = synthetic(sym, tf, total)
        return rows[:-1] if closed_only else rows
    rows, end = [], None
    while len(rows) < total:
        q = f"/api/v3/klines?symbol={sym}USDT&interval={tf}&limit=1000" + (f"&endTime={end}" if end else "")
        b = api(q)
        if not b:
            break
        rows = [[int(k[0]), *map(float, k[1:6])] for k in b] + rows
        end = int(b[0][0]) - 1
        if len(b) < 1000:
            break
    rows = rows[-total:]
    if closed_only and rows and rows[-1][0] + TFS[tf] * 1000 > time.time() * 1000:
        rows = rows[:-1]
    return rows


def last_price(sym, rows):
    if SYN:
        return rows[-1][4] * (1 + random.gauss(0, 0.002))
    return float(api(f"/api/v3/ticker/price?symbol={sym}USDT")["price"])


# ---------------------------------------------------------------- indicators
def ema(a, n):
    k, out, e = 2 / (n + 1), [], None
    for x in a:
        e = x if e is None else x * k + e * (1 - k)
        out.append(e)
    return out


def sma(a, n):
    out, s = [None] * len(a), 0.0
    for i, x in enumerate(a):
        s += x
        if i >= n:
            s -= a[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def rsi(c, n=14):
    out, g, l = [None] * len(c), 0.0, 0.0
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        u, dn = max(d, 0), max(-d, 0)
        if i <= n:
            g += u / n
            l += dn / n
            if i == n:
                out[i] = 100 - 100 / (1 + g / l) if l else 100
        else:
            g, l = (g * (n - 1) + u) / n, (l * (n - 1) + dn) / n
            out[i] = 100 - 100 / (1 + g / l) if l else 100
    return out


def atr(rows, n=14):
    out, a = [], 0.0
    for i, r in enumerate(rows):
        tr = r[2] - r[3] if i == 0 else max(r[2] - r[3], abs(r[2] - rows[i - 1][4]), abs(r[3] - rows[i - 1][4]))
        a = (a * i + tr) / (i + 1) if i < n else (a * (n - 1) + tr) / n
        out.append(a)
    return out


def macd_hist(c):
    f, s = ema(c, 12), ema(c, 26)
    m = [x - y for x, y in zip(f, s)]
    sg = ema(m, 9)
    return [x - y for x, y in zip(m, sg)]


def boll(c, n, k):
    mid = sma(c, n)
    up, lo = [None] * len(c), [None] * len(c)
    for i in range(n - 1, len(c)):
        w = c[i - n + 1:i + 1]
        sd = math.sqrt(sum((x - mid[i]) ** 2 for x in w) / n)
        up[i], lo[i] = mid[i] + k * sd, mid[i] - k * sd
    return mid, up, lo


def rolling(a, n, fn):
    return [None if i < n else fn(a[i - n:i]) for i in range(len(a))]  # excludes current bar


def score(rows):
    """Same composite score as the dashboard, -100..+100."""
    c, v = [r[4] for r in rows], [r[5] for r in rows]
    e50, e200, R, H = ema(c, 50), ema(c, 200), rsi(c), macd_hist(c)
    _, bu, bl = boll(c, 20, 2)
    vs = sma(v, 20)
    out = [0] * len(c)
    for i in range(200, len(c)):
        s = (25 if c[i] > e200[i] else -25) + (15 if c[i] > e50[i] else -15) + (10 if e50[i] > e200[i] else -10)
        s += (15 if H[i] > H[i - 1] else 7) if H[i] > 0 else (-15 if H[i] < H[i - 1] else -7)
        r = R[i]
        s += 15 if r < 30 else 8 if r < 40 else -15 if r > 70 else -8 if r > 60 else 0
        pb = (c[i] - bl[i]) / ((bu[i] - bl[i]) or 1)
        s += 10 if pb < 0.2 else -10 if pb > 0.8 else 0
        if vs[i] and v[i] > 1.5 * vs[i]:
            s += 10 if c[i] > rows[i][1] else -10
        out[i] = max(-100, min(100, s))
    return out


# ---------------------------------------------------------------- strategies
# each returns (entry[], exit[]) boolean lists aligned with rows
def s_ema_cross(rows, fast, slow):
    c = [r[4] for r in rows]
    f, s = ema(c, fast), ema(c, slow)
    en = [i > 0 and f[i] > s[i] and f[i - 1] <= s[i - 1] for i in range(len(c))]
    ex = [f[i] < s[i] for i in range(len(c))]
    return en, ex


def s_rsi(rows, lo, hi, trend):
    c = [r[4] for r in rows]
    R, e200 = rsi(c), ema(c, 200)
    en = [R[i] is not None and R[i] < lo and (not trend or c[i] > e200[i]) for i in range(len(c))]
    ex = [R[i] is not None and R[i] > hi for i in range(len(c))]
    return en, ex


def s_boll(rows, n, k):
    c = [r[4] for r in rows]
    mid, _, lo = boll(c, n, k)
    en = [lo[i] is not None and c[i] < lo[i] for i in range(len(c))]
    ex = [mid[i] is not None and c[i] > mid[i] for i in range(len(c))]
    return en, ex


def s_donchian(rows, n):
    h, l, c = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows]
    hh, ll = rolling(h, n, max), rolling(l, max(n // 2, 5), min)
    en = [hh[i] is not None and c[i] > hh[i] for i in range(len(c))]
    ex = [ll[i] is not None and c[i] < ll[i] for i in range(len(c))]
    return en, ex


def s_macd(rows, trend):
    c = [r[4] for r in rows]
    H, t = macd_hist(c), ema(c, trend)
    en = [i > 0 and H[i] > 0 >= H[i - 1] and c[i] > t[i] for i in range(len(c))]
    ex = [H[i] < 0 for i in range(len(c))]
    return en, ex


def s_composite(rows, buy, sell):
    sc = score(rows)
    return [x >= buy for x in sc], [x <= sell for x in sc]


STRATS = {
    "ema_cross": (s_ema_cross, [dict(fast=f, slow=s) for f in (9, 12, 20) for s in (26, 50, 100)]),
    "rsi_revert": (s_rsi, [dict(lo=a, hi=b, trend=t) for a in (25, 30, 35) for b in (50, 60, 70) for t in (0, 1)]),
    "boll_revert": (s_boll, [dict(n=20, k=k) for k in (2, 2.5)]),
    "donchian": (s_donchian, [dict(n=n) for n in (20, 40, 55)]),
    "macd_trend": (s_macd, [dict(trend=t) for t in (50, 100, 200)]),
    "composite": (s_composite, [dict(buy=b, sell=s) for b in (20, 30, 40) for s in (-20, 0)]),
}
STOPS = (0, 3)  # ATR multiple for a trailing stop; 0 = no stop


# ---------------------------------------------------------------- simulator
RISK_PER_TRADE = 0.03   # a 2.5-ATR adverse move may cost at most 3% of the coin's slice
ATR_RISK_MULT = 2.5
MIN_FRAC, MAX_FRAC = 0.25, 1.0
MAX_ACCOUNT_DD = 0.25   # whole account: sell everything and halt at -25% from its peak


def position_fraction(atr_val, price):
    """Share of the coin's slice to invest, smaller when the coin is more volatile."""
    if price <= 0 or atr_val <= 0:
        return MIN_FRAC
    return max(MIN_FRAC, min(MAX_FRAC, RISK_PER_TRADE / (ATR_RISK_MULT * atr_val / price)))


def simulate(rows, en, ex, stop, at, start, end, cash0=SLEEVE):
    """Long-only spot. Signal on bar i close, fill at bar i+1 open."""
    cash, qty, basis, peak = cash0, 0.0, 0.0, 0.0
    eq, rets = [], []
    for i in range(start, end - 1):
        c = rows[i][4]
        if qty > 0:
            peak = max(peak, c)
            if ex[i] or (stop and c < peak - stop * at[i]):
                proceeds = qty * rows[i + 1][1] * (1 - SLIP) * (1 - FEE)
                cash += proceeds
                rets.append(proceeds / basis - 1)
                qty = 0.0
        elif en[i] and cash > 0:
            invest = cash * position_fraction(at[i], c)
            basis, peak = invest, c
            qty = invest * (1 - FEE) / (rows[i + 1][1] * (1 + SLIP))
            cash -= invest
        eq.append(cash + qty * c)
    final = cash + qty * rows[end - 1][4] * (1 - FEE)
    pk, dd = 0.0, 0.0
    for e in eq:
        pk = max(pk, e)
        dd = max(dd, 1 - e / pk) if pk else 0
    bh = rows[end - 1][4] / rows[start][4] - 1
    wins = sum(1 for r in rets if r > 0)
    return dict(ret=final / cash0 - 1, dd=dd, n=len(rets), win=wins / len(rets) if rets else 0, bh=bh)


def run_combo(rows, name, params, stop, lo, hi):
    fn = STRATS[name][0]
    en, ex = fn(rows, **params)
    return simulate(rows, en, ex, stop, atr(rows), lo, hi)


# ---------------------------------------------------------------- backtest
def cmd_backtest():
    results, picks = [], {}
    for coin in COINS:
        for tf in TFS:
            try:
                rows = klines(coin, tf)
            except Exception as e:
                print(f"! {coin} {tf}: {e}")
                continue
            if len(rows) < 700:
                print(f"! {coin} {tf}: only {len(rows)} candles, skipped")
                continue
            split = int(len(rows) * 0.6)
            best = None
            for name, (_, grid) in STRATS.items():
                for params in grid:
                    for stop in STOPS:
                        tr = run_combo(rows, name, params, stop, WARMUP, split)
                        sc = tr["ret"] - 0.5 * tr["dd"] if tr["n"] >= 5 else -9
                        if best is None or sc > best[0]:
                            best = (sc, name, params, stop, tr)
            _, name, params, stop, tr = best
            te = run_combo(rows, name, params, stop, split, len(rows))
            days = (len(rows) - split) * TFS[tf] / 86400
            r = dict(coin=coin, tf=tf, strategy=name, params=params, stop=stop, train=tr, test=te, test_days=round(days))
            results.append(r)
            print(f"{coin:4} {tf:3} {name:12} train {tr['ret']*100:+7.1f}%  test {te['ret']*100:+7.1f}% "
                  f"(buy&hold {te['bh']*100:+7.1f}%, dd {te['dd']*100:4.1f}%, {te['n']} trades)")
    if not results:
        sys.exit("No data. Check your internet connection.")
    for coin in COINS:
        cand = sorted((r for r in results if r["coin"] == coin),
                      key=lambda r: r["train"]["ret"] - 0.5 * r["train"]["dd"], reverse=True)
        if not cand:
            continue
        good = [r for r in cand if r["test"]["ret"] > 0 and r["test"]["n"] >= 3]

        def pack(r, active):
            return dict(active=active, tf=r["tf"], strategy=r["strategy"], params=r["params"], stop=r["stop"],
                        test_ret=r["test"]["ret"], test_bh=r["test"]["bh"])
        if good:
            picks[coin] = pack(good[0], True)
            picks[coin]["alts"] = [pack(r, True) for r in good[1:3]]
        else:
            picks[coin] = pack(cand[0], False)
            picks[coin]["alts"] = []
    picks["generated"] = datetime.now(timezone.utc).isoformat()
    json.dump(results, open(os.path.join(OUT, "backtest.json"), "w"), indent=1)
    json.dump(picks, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    print("\nChosen for paper trading (train on first 60%, judged on the unseen last 40%):")
    for c, p in picks.items():
        if c == "generated":
            continue
        tag = "ACTIVE" if p["active"] else "cash (no edge found)"
        print(f"  {c:4} {tag:22} {p['strategy']} {p['params']} stop={p['stop']} tf={p['tf']}  "
              f"test {p['test_ret']*100:+.1f}% vs buy&hold {p['test_bh']*100:+.1f}%")
    coins_p = [p for k, p in picks.items() if k != "generated"]
    avg_t = sum(p["test_ret"] if p["active"] else 0 for p in coins_p) / len(coins_p)
    avg_b = sum(p["test_bh"] for p in coins_p) / len(coins_p)
    print(f"\nWhole account on the unseen period: strategy {avg_t*100:+.1f}% vs buy&hold {avg_b*100:+.1f}% "
          f"(${CAPITAL*(1+avg_t):,.0f} vs ${CAPITAL*(1+avg_b):,.0f})")
    print("Saved lab_data/backtest.json and lab_data/config.json. Next: python3 spot_lab.py paper")
    try:
        cmd_momentum_backtest()
    except Exception as e:
        print("  momentum backtest failed:", e)
    try:
        import research
        research.cmd_research()
    except Exception as e:
        print("  research backtest failed:", e)


# ---------------------------------------------------------------- challenger B: momentum rotation
UNIVERSE = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC", "TRX", "ATOM", "NEAR", "OP", "POL"]
MOM_GRID = [dict(lookback=l, top=n, regime=r, expo=x) for l in (30, 60, 90) for n in (2, 3, 5) for r in (0, 1) for x in (0.4, 0.7, 1.0)]
MOM_MAX_DD = 0.35     # a candidate whose training drawdown exceeds this is rejected outright
DAY_MS = 86400 * 1000
MOM_WARMUP = 110
STATE_B = os.path.join(OUT, "paper_B.json")


def ema_gaps(a, n):
    k, e, out = 2 / (n + 1), None, []
    for x in a:
        if x is None:
            out.append(None)
            continue
        e = x if e is None else x * k + e * (1 - k)
        out.append(e)
    return out


def load_universe(total, closed_only=True):
    out = {}
    for c in UNIVERSE:
        try:
            rows = klines(c, "1d", total, closed_only=closed_only)
        except Exception as e:
            print(f"  ! universe {c}: {e}")
            continue
        if len(rows) >= 130:
            out[c] = rows
    return out


def align(uni):
    days = sorted({r[0] for rows in uni.values() for r in rows})
    ix = {d: i for i, d in enumerate(days)}
    opn = {c: [None] * len(days) for c in uni}
    close = {c: [None] * len(days) for c in uni}
    for c, rows in uni.items():
        for r in rows:
            opn[c][ix[r[0]]], close[c][ix[r[0]]] = r[1], r[4]
    return days, opn, close


def mom_prepare(close):
    return {c: ema_gaps(cl, 50) for c, cl in close.items()}, ema_gaps(close["BTC"], 100)


def mom_targets(close, e50, e100b, t, p):
    """Coins to hold after the close of day t: strongest trailing returns that are above their own EMA50."""
    if p["regime"] and (close["BTC"][t] is None or e100b[t] is None or close["BTC"][t] <= e100b[t]):
        return []
    cands = []
    for c, cl in close.items():
        if t - p["lookback"] < 0 or cl[t] is None or cl[t - p["lookback"]] is None or e50[c][t] is None:
            continue
        if cl[t] <= e50[c][t]:
            continue
        sc = cl[t] / cl[t - p["lookback"]] - 1
        if sc > 0:
            cands.append((sc, c))
    cands.sort(reverse=True)
    return [c for _, c in cands[:p["top"]]]


def rebalance(cash, hold, tgt, price, buy_px=None, expo=1.0):
    """Sell what left the target list, buy new entries with equal budgets. Returns cash, hold, list of fills."""
    fills = []
    buy_px = buy_px or price
    for c in [c for c in hold if c not in tgt]:
        px = price.get(c)
        if px:
            cash += hold[c] * px * (1 - SLIP) * (1 - FEE)
            fills.append(("SELL", c, px, hold.pop(c)))
    total = cash + sum(q * price[c] for c, q in hold.items() if price.get(c))
    new = [c for c in tgt if c not in hold and buy_px.get(c)]
    if new:
        budget = min(cash / len(new), total * expo / max(len(tgt), 1))
        for c in new:
            px = buy_px[c]
            qty = budget * (1 - FEE) / (px * (1 + SLIP))
            cash -= budget
            hold[c] = qty
            fills.append(("BUY", c, px, qty))
    return cash, hold, fills


def mom_backtest(days, opn, close, e50, e100b, p, start, end, cash0=CAPITAL):
    cash, hold, eq, trades, last = cash0, {}, [], 0, None
    for t in range(start, end - 1):
        price = {c: close[c][t] for c in close if close[c][t]}
        eq.append(cash + sum(q * price.get(c, 0) for c, q in hold.items()))
        tgt = mom_targets(close, e50, e100b, t, p)
        if set(tgt) != set(hold):
            nxt = {c: opn[c][t + 1] for c in close if opn[c][t + 1]}
            cash, hold, fills = rebalance(cash, hold, tgt, nxt, expo=p.get("expo", 1.0))
            trades += len(fills)
    last_px = {c: close[c][end - 1] for c in close if close[c][end - 1]}
    final = cash + sum(q * last_px.get(c, 0) * (1 - FEE) for c, q in hold.items())
    pk, dd = 0.0, 0.0
    for e in eq:
        pk = max(pk, e)
        dd = max(dd, 1 - e / pk) if pk else 0
    b0, b1 = close["BTC"][start], close["BTC"][end - 1]
    return dict(ret=final / cash0 - 1, dd=dd, n=trades, bh=(b1 / b0 - 1) if b0 and b1 else 0)


def cmd_momentum_backtest():
    print("\nChallenger B: momentum rotation over", len(UNIVERSE), "coins (daily)")
    uni = load_universe(1000)
    if "BTC" not in uni or len(uni) < 5:
        print("  not enough data, skipped")
        return
    days, opn, close = align(uni)
    e50, e100b = mom_prepare(close)
    split = int(len(days) * 0.6)
    best = None
    for p in MOM_GRID:
        tr = mom_backtest(days, opn, close, e50, e100b, p, MOM_WARMUP, split)
        if tr["n"] < 6 or tr["dd"] > MOM_MAX_DD:
            continue
        sc = tr["ret"] - 0.5 * tr["dd"]
        if best is None or sc > best[0]:
            best = (sc, p, tr)
    if best is None:
        print(f"  no candidate kept training drawdown under {MOM_MAX_DD*100:.0f}%: challenger B stays in cash")
        json.dump(dict(active=False, params=dict(lookback=60, top=3, regime=1, expo=0.4), coins=sorted(uni),
                       reason="no candidate under the drawdown limit", generated=datetime.now(timezone.utc).isoformat()),
                  open(os.path.join(OUT, "config_B.json"), "w"), indent=1)
        return
    _, p, tr = best
    te = mom_backtest(days, opn, close, e50, e100b, p, split, len(days))
    active = te["ret"] > 0 and te["dd"] <= MOM_MAX_DD
    print(f"  best (training drawdown <= {MOM_MAX_DD*100:.0f}%): {p}")
    print(f"  train {tr['ret']*100:+.1f}% dd {tr['dd']*100:.0f}%   test {te['ret']*100:+.1f}% dd {te['dd']*100:.0f}% "
          f"(BTC hold {te['bh']*100:+.1f}%, {te['n']} fills)  ->  {'ACTIVE' if active else 'cash: failed out-of-sample'}")
    json.dump(dict(active=active, params=p, train=tr, test=te, coins=sorted(uni), generated=datetime.now(timezone.utc).isoformat()),
              open(os.path.join(OUT, "config_B.json"), "w"), indent=1)


def cycle_b():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cp = os.path.join(OUT, "config_B.json")
    conf = json.load(open(cp)) if os.path.exists(cp) else dict(active=False, params=dict(lookback=60, top=3, regime=1, expo=0.4))
    p, active = conf["params"], conf.get("active", False)
    rows_all = load_universe(260, closed_only=False)
    if "BTC" not in rows_all:
        print("  B: no data this cycle")
        return
    now_ms = time.time() * 1000
    closed = {c: (r[:-1] if r[-1][0] + DAY_MS > now_ms else r) for c, r in rows_all.items()}
    price = {c: r[-1][4] for c, r in rows_all.items()}
    days, opn, close = align(closed)
    e50, e100b = mom_prepare(close)
    t = len(days) - 1
    st = json.load(open(STATE_B)) if os.path.exists(STATE_B) else dict(
        start=datetime.now(timezone.utc).isoformat(), cash=CAPITAL, hold={}, last_bar=0, trades=[], events=[],
        btc_qty=CAPITAL / price["BTC"], eq_peak=CAPITAL, halted_until=0)
    value = lambda: st["cash"] + sum(q * price.get(c, 0) for c, q in st["hold"].items())
    st["eq_peak"] = max(st["eq_peak"], value())
    if st["halted_until"] and now_ms >= st["halted_until"]:
        st["halted_until"], st["eq_peak"] = 0, value()
        st["events"].append(dict(time=now, text="B: استُؤنف التداول بعد أسبوع التوقف"))
    if not st["halted_until"] and 1 - value() / st["eq_peak"] >= MAX_ACCOUNT_DD:
        for c in list(st["hold"]):
            px = price.get(c)
            if px:
                st["cash"] += st["hold"].pop(c) * px * (1 - SLIP) * (1 - FEE)
                st["trades"].append(dict(time=now, coin=c, side="SELL", price=px, why="halt"))
        st["halted_until"] = now_ms + 7 * DAY_MS
        st["events"].append(dict(time=now, text=f"B: قاطع الحساب أغلق كل المراكز بعد هبوط {MAX_ACCOUNT_DD*100:.0f}% وتوقف أسبوعاً"))
    bar = days[t]
    if not st["halted_until"] and bar != st["last_bar"]:
        tgt = mom_targets(close, e50, e100b, t, p) if active else []
        if set(tgt) != set(st["hold"]):
            st["cash"], st["hold"], fills = rebalance(st["cash"], st["hold"], tgt, price, expo=p.get("expo", 1.0))
            for side, c, px, _ in fills:
                st["trades"].append(dict(time=now, coin=c, side=side, price=px, why="rebalance"))
                print(f"  B {side} {c} @ {px:g}")
            st["events"].append(dict(time=now, text="B: التركيبة الجديدة " + (", ".join(tgt) if tgt else ("نقد (سوق BTC ضعيف أو لا زخم)" if active else "نقد (لم تنجح الاستراتيجية في الاختبار)"))))
        st["last_bar"] = bar
    st["trades"], st["events"] = st["trades"][-60:], st["events"][-20:]
    v = value()
    st["params"], st["active"], st["prices"] = p, active, {c: price[c] for c in st["hold"]}
    st["holdings"] = {c: dict(qty=q, value=q * price.get(c, 0)) for c, q in st["hold"].items()}
    st["equity"], st["btc_hold"] = v, st["btc_qty"] * price["BTC"]
    log_csv_b([now, round(v, 2), round(st["btc_hold"], 2)])
    json.dump(st, open(STATE_B, "w"), indent=1)
    print(f"[{now}] B account ${v:,.2f} ({(v/CAPITAL-1)*100:+.2f}%)  BTC hold ${st['btc_hold']:,.2f}  holding {list(st['hold']) or 'cash'}")


def log_csv_b(row):
    path = os.path.join(OUT, "paper_equity_B.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "equity", "btc_hold"])
        w.writerow(row)


# ---------------------------------------------------------------- paper trading
STATE = os.path.join(OUT, "paper.json")


PAUSE_LOSS = -0.08     # live compounded return of closed trades for one strategy
PAUSE_DD = 0.12        # drawdown of the coin's sleeve from its own peak
PAUSE_MIN_TRADES = 3


def load_state(prices):
    if os.path.exists(STATE):
        st = json.load(open(STATE))
    else:
        st = dict(start=datetime.now(timezone.utc).isoformat(), cash={c: SLEEVE for c in COINS},
                  qty={c: 0.0 for c in COINS}, basis={c: 0.0 for c in COINS}, peak={c: 0.0 for c in COINS},
                  last_bar={c: 0 for c in COINS}, bh_qty={c: SLEEVE / prices[c] for c in COINS}, trades=[])
    st.setdefault("override", {})
    st.setdefault("paused", {})
    st.setdefault("tried", {})
    st.setdefault("sleeve_peak", {c: SLEEVE for c in COINS})
    st.setdefault("events", [])
    st.setdefault("cfg_gen", "")
    st.setdefault("perf", {})
    st.setdefault("frac", {})
    st.setdefault("halted", "")
    st.setdefault("eq_peak", 0.0)
    return st


def equity(st, prices):
    return sum(st["cash"][c] + st["qty"][c] * prices[c] for c in COINS)


HEADERS = {"paper_trades.csv": ["time", "coin", "side", "price", "result", "reason"],
           "paper_equity.csv": ["time", "equity", "buy_and_hold"]}


def log_csv(name, row):
    path = os.path.join(OUT, name)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADERS[name])
        w.writerow(row)


def skey(p):
    return f"{p['strategy']}|{p['tf']}|{json.dumps(p['params'], sort_keys=True)}|{p['stop']}"


def effective(cfg, st, c):
    """The strategy that is allowed to trade this coin right now, or None (stay in cash)."""
    if c in st["paused"]:
        return None
    p = st["override"].get(c) or cfg.get(c)
    return p if p and p.get("active") else None


def event(st, now, text):
    st["events"] = (st["events"] + [dict(time=now, text=text)])[-30:]
    print("  EVENT:", text)


def live_stats(st, c, key):
    rets = [t.get("sret", t["ret"]) for t in st["trades"]
            if t["coin"] == c and t["side"] == "SELL" and t.get("key") == key and t.get("gen", "") == st["cfg_gen"]]
    comp = 1.0
    for r in rets:
        comp *= 1 + r
    return dict(n=len(rets), win=(sum(1 for r in rets if r > 0) / len(rets)) if rets else 0, ret=comp - 1)


def sell(st, c, px, now, why):
    proceeds = st["qty"][c] * px * (1 - SLIP) * (1 - FEE)
    ret = proceeds / st["basis"][c] - 1
    frac = st["frac"].get(c, 1.0)
    st["trades"].append(dict(time=now, coin=c, side="SELL", price=px, ret=ret, sret=ret * frac, why=why,
                             key=st.get("holding_key", {}).get(c), gen=st["cfg_gen"]))
    log_csv("paper_trades.csv", [now, c, "SELL", px, f"{ret*100:.2f}%", why])
    print(f"  SELL {c} @ {px:g}  {ret*100:+.2f}%  ({why})")
    st["cash"][c] += proceeds
    st["qty"][c] = 0.0


def cycle(cfg, st):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if cfg.get("generated", "") != st["cfg_gen"]:       # a fresh backtest replaced the strategy set
        if st["cfg_gen"]:
            event(st, now, "تم تحديث الاستراتيجيات بعد اختبار جديد، وأُلغيت التحويلات والإيقافات السابقة")
        st["override"], st["paused"], st["tried"] = {}, {}, {}
        st["cfg_gen"] = cfg.get("generated", "")
        if st["halted"]:
            event(st, now, "استُؤنف التداول بعد اختبار جديد للاستراتيجيات، وبدأ حد الهبوط من الرصيد الحالي")
            st["halted"], st["eq_peak"] = "", 0.0
    prices, data = {}, {}
    for c in COINS:
        p = effective(cfg, st, c) or cfg.get(c)
        tf = p["tf"] if p else "1h"
        try:
            rows = klines(c, tf, 400)
            data[c] = rows
            prices[c] = last_price(c, rows)
        except Exception as e:
            print(f"  ! {c}: {e}")
    if len(prices) < len(COINS):
        print("  skipping this cycle, missing prices")
        return st
    st.setdefault("holding_key", {})
    eq_now = equity(st, prices)
    st["eq_peak"] = max(st["eq_peak"] or 0.0, eq_now)
    if not st["halted"] and 1 - eq_now / st["eq_peak"] >= MAX_ACCOUNT_DD:
        for c in COINS:
            if st["qty"][c] > 0:
                sell(st, c, prices[c], now, "halt")
        st["halted"] = now
        event(st, now, f"قاطع الحساب: هبط الرصيد {MAX_ACCOUNT_DD*100:.0f}% من قمته، أُغلقت كل الصفقات وتوقف التداول حتى الاختبار الأسبوعي القادم")
    for c in COINS:
        p = None if st["halted"] else effective(cfg, st, c)
        if not p:
            continue
        key = skey(p)
        rows, px = data[c], prices[c]
        en, ex = STRATS[p["strategy"]][0](rows, **p["params"])
        at = atr(rows)
        bar = rows[-1][0]
        if st["qty"][c] > 0:
            st["peak"][c] = max(st["peak"][c], px)
            stopped = p["stop"] and px < st["peak"][c] - p["stop"] * at[-1]
            if ex[-1] or stopped:
                sell(st, c, px, now, "stop" if stopped else "signal")
        elif en[-1] and bar != st["last_bar"][c]:
            frac = position_fraction(at[-1], px)
            invest = st["cash"][c] * frac
            st["basis"][c], st["peak"][c], st["frac"][c] = invest, px, frac
            st["qty"][c] = invest * (1 - FEE) / (px * (1 + SLIP))
            st["cash"][c] -= invest
            st["holding_key"][c] = key
            st["trades"].append(dict(time=now, coin=c, side="BUY", price=px, key=key, gen=st["cfg_gen"], size=frac))
            log_csv("paper_trades.csv", [now, c, "BUY", px, "", ""])
            print(f"  BUY  {c} @ {px:g}  size {frac*100:.0f}% of slice")
        if en[-1]:
            st["last_bar"][c] = bar
        # ---- live performance check: pause or switch a strategy that is losing
        v = st["cash"][c] + st["qty"][c] * px
        st["sleeve_peak"][c] = max(st["sleeve_peak"].get(c, SLEEVE), v)
        dd = 1 - v / st["sleeve_peak"][c]
        ls = live_stats(st, c, key)
        why = None
        if ls["n"] >= PAUSE_MIN_TRADES and ls["ret"] <= PAUSE_LOSS:
            why = f"خسارة حية {ls['ret']*100:.1f}% في {ls['n']} صفقات"
        elif dd >= PAUSE_DD:
            why = f"هبوط {dd*100:.1f}% من قمة رصيد العملة"
        if why:
            if st["qty"][c] > 0:
                sell(st, c, px, now, "pause")
            st["tried"].setdefault(c, []).append(key)
            nxt = next((a for a in cfg.get(c, {}).get("alts", []) if skey(a) not in st["tried"][c]), None)
            st["sleeve_peak"][c] = st["cash"][c]
            if nxt:
                st["override"][c] = nxt
                event(st, now, f"{c}: توقفت {p['strategy']} ({why}) وتم التحويل إلى {nxt['strategy']} {nxt['tf']}")
            else:
                st["paused"][c] = why
                event(st, now, f"{c}: توقفت {p['strategy']} ({why}) ولا بديل، العملة نقد حتى الاختبار القادم")
    # ---- performance log, shown on the dashboard
    perf = {}
    for c in COINS:
        p = effective(cfg, st, c)
        v = st["cash"][c] + st["qty"][c] * prices[c]
        base = cfg.get(c, {})
        if p:
            ls = live_stats(st, c, skey(p))
            status = "بديلة" if c in st["override"] else "نشطة"
        else:
            ls = dict(n=0, win=0, ret=0)
            status = "موقوفة" if c in st["paused"] else "نقد"
        perf[c] = dict(status=status, strategy=(p or base).get("strategy"), tf=(p or base).get("tf"),
                       live_n=ls["n"], live_win=ls["win"], live_ret=ls["ret"],
                       sleeve_ret=v / SLEEVE - 1, sleeve_dd=1 - v / st["sleeve_peak"].get(c, SLEEVE),
                       backtest_ret=(p or base).get("test_ret"), reason=st["paused"].get(c, ""),
                       size=st["frac"].get(c) if st["qty"][c] > 0 else None)
    st["perf"] = perf
    eq = equity(st, prices)
    bh = sum(st["bh_qty"][c] * prices[c] for c in COINS)
    log_csv("paper_equity.csv", [now, round(eq, 2), round(bh, 2)])
    print(f"[{now}] account ${eq:,.2f} ({(eq/CAPITAL-1)*100:+.2f}%)   buy&hold ${bh:,.2f} ({(bh/CAPITAL-1)*100:+.2f}%)")
    st["prices"] = prices
    return st


def cmd_paper():
    cfg_path = os.path.join(OUT, "config.json")
    if not os.path.exists(cfg_path):
        sys.exit("Run `python3 spot_lab.py backtest` first.")
    cfg = json.load(open(cfg_path))
    print("Paper account: $%.0f virtual. Checking every 5 minutes. Ctrl+C to stop (state is kept)." % CAPITAL)
    st = None
    while True:
        try:
            if st is None:
                init = {}
                for c in COINS:
                    rows = klines(c, "1h", 400)
                    init[c] = last_price(c, rows)
                st = load_state(init)
            st = cycle(cfg, st)
            json.dump(st, open(STATE, "w"), indent=1)
            try:
                cycle_b()
            except Exception as e:
                print("  challenger B error:", e)
            try:
                import research
                research.cycle_c()
            except Exception as e:
                print("  research account C error:", e)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("  cycle error:", e)
        if "--once" in sys.argv:
            return
        time.sleep(300)


def cmd_report():
    if not os.path.exists(STATE):
        sys.exit("No paper account yet. Run `python3 spot_lab.py paper`.")
    st = json.load(open(STATE))
    px = st.get("prices")
    if not px:
        sys.exit("No prices recorded yet.")
    eq = equity(st, px)
    bh = sum(st["bh_qty"][c] * px[c] for c in COINS)
    sells = [t for t in st["trades"] if t["side"] == "SELL"]
    wins = sum(1 for t in sells if t["ret"] > 0)
    print(f"Started {st['start'][:16]} UTC")
    print(f"Account      ${eq:,.2f}  ({(eq/CAPITAL-1)*100:+.2f}%)")
    print(f"Buy & hold   ${bh:,.2f}  ({(bh/CAPITAL-1)*100:+.2f}%)")
    print(f"Closed trades {len(sells)}, winners {wins}")
    for c in COINS:
        pos = "in position" if st["qty"][c] > 0 else "cash"
        print(f"  {c:4} ${st['cash'][c] + st['qty'][c]*px[c]:,.2f}  {pos}")


if __name__ == "__main__":
    cmds = {"backtest": cmd_backtest, "paper": cmd_paper, "report": cmd_report}
    cmd = next((a for a in sys.argv[1:] if a in cmds), None)
    if not cmd:
        sys.exit(__doc__)
    cmds[cmd]()
