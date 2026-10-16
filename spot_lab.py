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


def klines(sym, tf, total=3000):
    """Closed candles only: [time_ms, open, high, low, close, volume]."""
    if SYN:
        return synthetic(sym, tf, total)[:-1]
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
    if rows and rows[-1][0] + TFS[tf] * 1000 > time.time() * 1000:
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
def simulate(rows, en, ex, stop, at, start, end, cash0=SLEEVE):
    """Long-only spot. Signal on bar i close, fill at bar i+1 open."""
    cash, qty, basis, peak = cash0, 0.0, 0.0, 0.0
    eq, rets = [], []
    for i in range(start, end - 1):
        c = rows[i][4]
        if qty > 0:
            peak = max(peak, c)
            if ex[i] or (stop and c < peak - stop * at[i]):
                cash = qty * rows[i + 1][1] * (1 - SLIP) * (1 - FEE)
                rets.append(cash / basis - 1)
                qty = 0.0
        elif en[i] and cash > 0:
            basis, peak = cash, c
            qty = cash * (1 - FEE) / (rows[i + 1][1] * (1 + SLIP))
            cash = 0.0
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
        cand = [r for r in results if r["coin"] == coin]
        if not cand:
            continue
        r = max(cand, key=lambda r: r["train"]["ret"] - 0.5 * r["train"]["dd"])
        ok = r["test"]["ret"] > 0 and r["test"]["n"] >= 3
        picks[coin] = dict(active=ok, tf=r["tf"], strategy=r["strategy"], params=r["params"], stop=r["stop"],
                           test_ret=r["test"]["ret"], test_bh=r["test"]["bh"])
    json.dump(results, open(os.path.join(OUT, "backtest.json"), "w"), indent=1)
    json.dump(picks, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    print("\nChosen for paper trading (train on first 60%, judged on the unseen last 40%):")
    for c, p in picks.items():
        tag = "ACTIVE" if p["active"] else "cash (no edge found)"
        print(f"  {c:4} {tag:22} {p['strategy']} {p['params']} stop={p['stop']} tf={p['tf']}  "
              f"test {p['test_ret']*100:+.1f}% vs buy&hold {p['test_bh']*100:+.1f}%")
    avg_t = sum(p["test_ret"] if p["active"] else 0 for p in picks.values()) / len(picks)
    avg_b = sum(p["test_bh"] for p in picks.values()) / len(picks)
    print(f"\nWhole account on the unseen period: strategy {avg_t*100:+.1f}% vs buy&hold {avg_b*100:+.1f}% "
          f"(${CAPITAL*(1+avg_t):,.0f} vs ${CAPITAL*(1+avg_b):,.0f})")
    print("Saved lab_data/backtest.json and lab_data/config.json. Next: python3 spot_lab.py paper")


# ---------------------------------------------------------------- paper trading
STATE = os.path.join(OUT, "paper.json")


def load_state(prices):
    if os.path.exists(STATE):
        return json.load(open(STATE))
    st = dict(start=datetime.now(timezone.utc).isoformat(), cash={c: SLEEVE for c in COINS},
              qty={c: 0.0 for c in COINS}, basis={c: 0.0 for c in COINS}, peak={c: 0.0 for c in COINS},
              last_bar={c: 0 for c in COINS}, bh_qty={c: SLEEVE / prices[c] for c in COINS}, trades=[])
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


def cycle(cfg, st):
    prices, data = {}, {}
    for c in COINS:
        p = cfg.get(c)
        tf = p["tf"] if p and p["active"] else "1h"
        try:
            rows = klines(c, tf, 400)
            data[c] = rows
            prices[c] = last_price(c, rows)
        except Exception as e:
            print(f"  ! {c}: {e}")
    if len(prices) < len(COINS):
        print("  skipping this cycle, missing prices")
        return st
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for c in COINS:
        p = cfg.get(c)
        if not p or not p["active"]:
            continue
        rows, px = data[c], prices[c]
        fn = STRATS[p["strategy"]][0]
        en, ex = fn(rows, **p["params"])
        at = atr(rows)
        bar = rows[-1][0]
        if st["qty"][c] > 0:
            st["peak"][c] = max(st["peak"][c], px)
            stopped = p["stop"] and px < st["peak"][c] - p["stop"] * at[-1]
            if ex[-1] or stopped:
                cash = st["qty"][c] * px * (1 - SLIP) * (1 - FEE)
                ret = cash / st["basis"][c] - 1
                st["trades"].append(dict(time=now, coin=c, side="SELL", price=px, ret=ret, why="stop" if stopped else "signal"))
                log_csv("paper_trades.csv", [now, c, "SELL", px, f"{ret*100:.2f}%", "stop" if stopped else "signal"])
                print(f"  SELL {c} @ {px:g}  {ret*100:+.2f}%  ({'stop' if stopped else 'signal'})")
                st["cash"][c], st["qty"][c] = cash, 0.0
        elif en[-1] and bar != st["last_bar"][c]:
            cash = st["cash"][c]
            st["basis"][c], st["peak"][c] = cash, px
            st["qty"][c] = cash * (1 - FEE) / (px * (1 + SLIP))
            st["cash"][c] = 0.0
            st["trades"].append(dict(time=now, coin=c, side="BUY", price=px))
            log_csv("paper_trades.csv", [now, c, "BUY", px, "", ""])
            print(f"  BUY  {c} @ {px:g}")
        if en[-1]:
            st["last_bar"][c] = bar
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
