#!/usr/bin/env python3
"""Bear-market hunting research (spot only, no shorting, no leverage).  Read-only: writes lab_data/bear_hunt.json.

Question: when the market is falling, is there anything a spot trader can hunt?  Three playbooks, every parameter
fixed BEFORE looking at results (no grid search, so there is nothing to over-fit):

  P1 capitulation bounce   BTC in a bear regime; the coin has crashed (RSI14 < 25 within the last 3 days and at least
                           20% under its 20-day high), selling climaxed (a day with volume >= 1.8x its 20-day average
                           in the last 3 days), and today closes green, above yesterday's close, in the top 40% of its
                           range.  Enter next open.  Stop 1% under the lowest low of the last 3 days (at most 10% away).
                           Exit when the price touches the falling EMA20 or +10%, whichever first; time exit 10 days.
  P2 strength in weakness  BTC in a bear regime; the coin is above its rising EMA50, beat BTC by 20+ points over 30 days,
                           and closes at a new 20-day high.  Enter next open.  Stop 2 ATR (at most 12%), then a trailing
                           stop 3 ATR under the highest close; time exit 40 days.
  P3 deep-value DCA        BTC and ETH only: close more than 20% under the 350-day EMA and RSI14 < 35.  Buy, at most once
                           every 14 days, and hold 180 days (no stop).  This is accumulation, not trading.

Bear regime = BTC daily close under its EMA200 and EMA50 under EMA200 (yesterday's values, no look-ahead).
Every playbook is compared with RANDOM entries taken on bear days with the SAME exits, with costs x3, by bootstrap,
by the two halves of history, and per bear episode.  With --synthetic the same pipeline runs on random walks and writes
bear_hunt_null.json: whatever "works" there is luck, the yardstick.
"""
import json, os, random, sys
from datetime import datetime, timezone

import spot_lab as sl

OUT = sl.OUT
COINS = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC", "OP", "DOT", "TRX", "ATOM", "NEAR",
         "XLM", "HBAR", "SUI", "APT", "ARB", "ALGO", "FIL", "ICP", "ETC", "BCH", "ZRO", "ZK", "ZEC", "TAO",
         "RENDER", "QNT", "WLD"]
DAYS = 3000
FEE, SLIP = sl.FEE, sl.SLIP


# ---------------------------------------------------------------- helpers
def stats(r):
    if not r:
        return dict(n=0, win=0.0, mean=0.0, median=0.0, worst=0.0, best=0.0)
    s = sorted(r)
    return dict(n=len(r), win=sum(x > 0 for x in r) / len(r), mean=sum(r) / len(r), median=s[len(s) // 2],
                worst=s[0], best=s[-1])


def boot(r, trials=2000):
    if len(r) < 5:
        return 0.0
    rnd, ok = random.Random(11), 0
    for _ in range(trials):
        ok += sum(rnd.choice(r) for _ in r) > 0
    return ok / trials


def net(entry, exit_, cost_mult=1.0):
    f, s = FEE * cost_mult, SLIP * cost_mult
    return exit_ * (1 - s) * (1 - f) / (entry * (1 + s) * (1 + f)) - 1


def btc_regime(btc):
    c = [r[4] for r in btc]
    e50, e200 = sl.ema(c, 50), sl.ema(c, 200)
    by_day = {}
    for i, r in enumerate(btc):
        by_day[r[0] // 86400000] = i >= 200 and c[i] < e200[i] and e50[i] < e200[i]
    return by_day, c


def episodes(btc, bear_by_day):
    """Contiguous bear stretches of BTC (>= 30 days), as (start_day, end_day, label)."""
    out, start = [], None
    days = [r[0] // 86400000 for r in btc]
    for d in days:
        b = bear_by_day.get(d - 1, False)
        if b and start is None:
            start = d
        if not b and start is not None:
            if d - start >= 30:
                out.append((start, d))
            start = None
    if start is not None and days[-1] - start >= 30:
        out.append((start, days[-1]))
    lab = lambda d: datetime.fromtimestamp(d * 86400, timezone.utc).strftime("%Y-%m")
    return [(a, b, f"{lab(a)} → {lab(b)}") for a, b in out]


# ---------------------------------------------------------------- the three playbooks (fixed rules)
def p1_signals(rows, bear):
    c = [r[4] for r in rows]
    R, e20 = sl.rsi(c), sl.ema(c, 20)
    out = []
    for i in range(len(rows)):
        ok = False
        if i >= 25 and bear(rows[i][0]):
            hi20 = max(r[2] for r in rows[i - 20:i + 1])
            vavg = sum(r[5] for r in rows[i - 23:i - 3]) / 20 or 1
            crashed = any(R[j] is not None and R[j] < 25 for j in range(i - 2, i + 1)) and c[i] <= hi20 * 0.80
            climax = any(rows[j][5] >= 1.8 * vavg for j in range(i - 2, i + 1))
            rng = rows[i][2] - rows[i][3]
            green = rows[i][4] > rows[i][1] and c[i] > c[i - 1] and rng > 0 and (rows[i][4] - rows[i][3]) / rng >= 0.6
            ok = crashed and climax and green
        out.append(ok)
    return out, e20


def p1_exit(rows, i, e20, cost_mult):
    """Entry at open of i+1."""
    if i + 1 >= len(rows):
        return None
    px = rows[i + 1][1]
    stop = max(min(r[3] for r in rows[i - 2:i + 1]) * 0.99, px * 0.90)
    tgt = px * 1.10
    for j in range(i + 1, min(len(rows), i + 11)):
        lo, hi = rows[j][3], rows[j][2]
        if lo <= stop:
            return net(px, min(stop, rows[j][1]), cost_mult), j, "stop"
        level = min(tgt, e20[j - 1]) if e20[j - 1] > px else tgt
        if hi >= level:
            return net(px, max(level, rows[j][1]) if rows[j][1] > level else level, cost_mult), j, "ema20" if level < tgt else "target"
    j = min(len(rows) - 1, i + 10)
    return net(px, rows[j][4], cost_mult), j, "time"


def p2_signals(rows, bear, btc_close_by_day):
    c = [r[4] for r in rows]
    e50 = sl.ema(c, 50)
    out = []
    for i in range(len(rows)):
        ok = False
        if i >= 60 and bear(rows[i][0]):
            d = rows[i][0] // 86400000
            b_now, b_then = btc_close_by_day.get(d), btc_close_by_day.get(d - 30)
            if b_now and b_then:
                rel = (c[i] / c[i - 30] - 1) - (b_now / b_then - 1)
                ok = (c[i] > e50[i] and e50[i] > e50[i - 5] and rel >= 0.20 and c[i] > max(r[2] for r in rows[i - 20:i]))
        out.append(ok)
    return out


def p2_exit(rows, i, at, cost_mult):
    if i + 1 >= len(rows):
        return None
    px = rows[i + 1][1]
    stop = px - min(2 * at[i], 0.12 * px)
    best = px
    for j in range(i + 1, min(len(rows), i + 41)):
        if rows[j][3] <= stop:
            return net(px, min(stop, rows[j][1]), cost_mult), j, "stop"
        best = max(best, rows[j][4])
        stop = max(stop, best - 3 * at[j])
    j = min(len(rows) - 1, i + 40)
    return net(px, rows[j][4], cost_mult), j, "time"


def p3_signals(rows):
    c = [r[4] for r in rows]
    e350, R = sl.ema(c, 350), sl.rsi(c)
    return [i >= 350 and c[i] < 0.80 * e350[i] and R[i] is not None and R[i] < 35 for i in range(len(rows))]


def p3_exit(rows, i, cost_mult):
    if i + 181 >= len(rows):
        return None
    return net(rows[i + 1][1], rows[i + 181][4], cost_mult), i + 181, "hold180"


# ---------------------------------------------------------------- run
def run_playbook(name, data, bear, btc_close_by_day, cost_mult=1.0, random_mode=False, rnd=None):
    trades = []
    for coin, rows in data.items():
        if name == "P3" and coin not in ("BTC", "ETH"):
            continue
        at = sl.atr(rows)
        if name == "P1":
            sig, e20 = p1_signals(rows, bear)
        elif name == "P2":
            sig = p2_signals(rows, bear, btc_close_by_day)
        else:
            sig = p3_signals(rows)
        if random_mode:                                   # same number of entries, random bear days (or any day for P3)
            pool = [i for i in range(60, len(rows) - 2) if (name == "P3" and i >= 350) or (name != "P3" and bear(rows[i][0]))]
            k = sum(sig)
            sig = [False] * len(rows)
            for i in rnd.sample(pool, min(k, len(pool))):
                sig[i] = True
            if name == "P1":
                e20 = sl.ema([r[4] for r in rows], 20)
        i, last = 25, -999
        while i < len(rows) - 2:
            if not sig[i] or (name == "P3" and i - last < 14):
                i += 1
                continue
            if name == "P1":
                res = p1_exit(rows, i, e20, cost_mult)
            elif name == "P2":
                res = p2_exit(rows, i, at, cost_mult)
            else:
                res = p3_exit(rows, i, cost_mult)
            if res is None:
                break
            ret, j, why = res
            trades.append(dict(coin=coin, day=rows[i][0] // 86400000, ret=ret, why=why, days=j - i))
            last = i
            i = (j + 1) if name != "P3" else i + 1
    return trades


def summarize(trades, eps):
    r = [t["ret"] for t in trades]
    out = stats(r)
    out["boot_p"] = boot(r)
    if trades:
        mid = sorted(t["day"] for t in trades)[len(trades) // 2]
        out["first_half"] = stats([t["ret"] for t in trades if t["day"] < mid])
        out["second_half"] = stats([t["ret"] for t in trades if t["day"] >= mid])
        out["by_exit"] = {k: stats([t["ret"] for t in trades if t["why"] == k]) for k in sorted({t["why"] for t in trades})}
        out["by_episode"] = {lab: stats([t["ret"] for t in trades if a <= t["day"] <= b]) for a, b, lab in eps}
        out["avg_days"] = sum(t["days"] for t in trades) / len(trades)
        out["coins"] = len({t["coin"] for t in trades})
    return out


def main():
    data = {}
    for c in COINS:
        try:
            rows = sl.klines(c, "1d", DAYS)
            if len(rows) >= 300:
                data[c] = rows
        except Exception as e:
            print(f"  ! {c}: {e}")
    if "BTC" not in data:
        print("no BTC data")
        return
    bear_by_day, _ = btc_regime(data["BTC"])
    bear = lambda t: bear_by_day.get(t // 86400000 - 1, False)       # yesterday's regime
    btc_close_by_day = {r[0] // 86400000: r[4] for r in data["BTC"]}
    eps = episodes(data["BTC"], bear_by_day)
    bear_days = sum(1 for v in bear_by_day.values() if v)
    res = {}
    for name in ("P1", "P2", "P3"):
        real = run_playbook(name, data, bear, btc_close_by_day)
        x3 = run_playbook(name, data, bear, btc_close_by_day, cost_mult=3.0)
        rnd_all = []
        for seed in range(5):
            rnd_all += run_playbook(name, data, bear, btc_close_by_day, random_mode=True, rnd=random.Random(seed))
        res[name] = dict(real=summarize(real, eps), costs_x3=stats([t["ret"] for t in x3]), random=stats([t["ret"] for t in rnd_all]),
                         recent=[dict(coin=t["coin"], date=datetime.fromtimestamp(t["day"] * 86400, timezone.utc).strftime("%Y-%m-%d"),
                                      ret=round(t["ret"], 4), why=t["why"], days=t["days"]) for t in sorted(real, key=lambda t: t["day"])[-12:]])
        s, r_ = res[name]["real"], res[name]["random"]
        print(f"{name}: n={s['n']} win {s['win']*100:.0f}% mean {s['mean']*100:+.2f}% median {s['median']*100:+.2f}% "
              f"| random mean {r_['mean']*100:+.2f}% | x3 {res[name]['costs_x3']['mean']*100:+.2f}% | boot {s['boot_p']:.2f}")
        if s["n"]:
            print(f"     halves {s['first_half']['mean']*100:+.2f}% / {s['second_half']['mean']*100:+.2f}%  exits "
                  + ", ".join(f"{k} {v['n']} ({v['mean']*100:+.1f}%)" for k, v in s["by_exit"].items()))
            print("     episodes " + " | ".join(f"{k}: {v['n']} {v['mean']*100:+.1f}%" for k, v in s["by_episode"].items() if v["n"]))
    out = dict(generated=datetime.now(timezone.utc).isoformat(), synthetic=sl.SYN, coins=len(data), bear_days=bear_days,
               episodes=[lab for _, _, lab in eps], playbooks=res,
               note="rules fixed in advance; random = same number of entries on random bear days with the same exits (5 seeds)")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "bear_hunt_null.json" if sl.SYN else "bear_hunt.json"), "w"), indent=1)
    print("bear days", bear_days, "episodes", [lab for _, _, lab in eps])


if __name__ == "__main__":
    main()
