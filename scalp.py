#!/usr/bin/env python3
"""Account D: daily quick-strike session (paper only). Rules are fixed in advance and never tuned on live results.

  coins      BTC, ETH, SOL (liquid)
  signals    closed 1h bar: RSI(14) < 30 while price > EMA200   OR   pullback to EMA20 inside an EMA50 > EMA200 uptrend
  exits      +2% target (take profit), -3% stop, or 24 hours (whichever comes first); on a bar that touches both, the stop is assumed first
  size       20% of the account per trade, at most 3 open positions and 3 new trades per UTC day
  guards     no new trades for the rest of the day after -1% of the day's starting equity; account halts at -10% from the start
  costs      OKX taker fee 0.10% + slippage 0.05% per side (same as the other accounts)
  gate       no real money considered before 30 closed trades AND a positive average trade AND bootstrap P(average > 0) >= 0.9

`python scalp.py`          one cycle (called every 5 minutes by paper.yml)
`python scalp.py backtest` the same rules on history, with a random-entry comparison and costs x3
It never trades real money and needs no keys. State: lab_data/paper_D.json, paper_equity_D.csv, backtest_D.json
"""
import csv, json, math, os, random, sys, time
from datetime import datetime, timezone

import spot_lab as sl
import hitrate as hr

OUT = sl.OUT
STATE = os.path.join(OUT, "paper_D.json")
CAPITAL = 5000.0
COINS = ["BTC", "ETH", "SOL"]
TP, SL_, HOLD_H = 0.02, 0.03, 24
SIZE_FRAC, MAX_OPEN, MAX_DAY = 0.20, 3, 3
DAY_LOSS_CAP, HALT_DD = 0.01, 0.10
GATE_TRADES, GATE_BOOT = 30, 0.90
FEE, SLIP = sl.FEE, sl.SLIP


# ---------------------------------------------------------------- signals (reuse the research definitions, fixed parameters)
def signal_series(rows):
    a = hr.e_rsi_dip(rows, th=30, trend=True)
    b = hr.e_ema_pull(rows, band=1.0, rmax=50)
    return a, b


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def boot_mean_pos(r, trials=1000):
    if len(r) < 5:
        return 0.0
    rnd, ok = random.Random(9), 0
    for _ in range(trials):
        ok += sum(rnd.choice(r) for _ in r) > 0
    return ok / trials


def streak(r):
    m = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        m = max(m, cur)
    return m


def make_stats(trades):
    sells = [t for t in trades if t["side"] == "SELL" and t.get("ret") is not None]
    r = [t["ret"] for t in sells]
    n = len(r)
    out = dict(trades=n, gate_trades=GATE_TRADES)
    if not n:
        out["gate"] = f"لا صفقات مغلقة بعد (0 من {GATE_TRADES})"
        return out
    wins = [x for x in r if x > 0]
    losses = [x for x in r if x <= 0]
    lo, hi = wilson(len(wins), n)
    aw = sum(wins) / len(wins) if wins else 0.0
    al = sum(losses) / len(losses) if losses else 0.0
    be = (-al / (aw - al)) if (aw - al) > 0 and al < 0 else None
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None
    by = {}
    for key in ("why", "signal", "coin"):
        d = {}
        for t in sells:
            v = t.get(key) or "-"
            d.setdefault(v, []).append(t["ret"])
        by[key] = {k: dict(n=len(v), win=sum(x > 0 for x in v) / len(v), mean=sum(v) / len(v)) for k, v in d.items()}
    mean = sum(r) / n
    bp = boot_mean_pos(r)
    gate = (f"مغلقة {n} من {GATE_TRADES} المطلوبة" if n < GATE_TRADES else
            ("اجتاز شرط العدد، لكن المتوسط سالب" if mean <= 0 else
             ("اجتاز شرط العدد والمتوسط، وثقة الربح دون 90%" if bp < GATE_BOOT else "اجتاز كل الشروط (يبقى قراراً بشرياً ولا ضمان)")))
    out.update(win_rate=len(wins) / n, win_ci=[lo, hi], avg_win=aw, avg_loss=al, expectancy=mean, breakeven_win=be,
               profit_factor=pf, boot_p=bp, worst=min(r), loss_streak=streak(r),
               avg_hours=sum(t.get("hours", 0) for t in sells) / n, by=by, gate=gate)
    return out


# ---------------------------------------------------------------- market data
def bars5(coin, rows1h):
    """Recent 5-minute candles [t, o, h, l, c, v] (synthetic mode: the latest 1h candle as one bar)."""
    if sl.SYN:
        return [rows1h[-1]]
    b = sl.api(f"/api/v3/klines?symbol={coin}USDT&interval=5m&limit=300")
    return [[int(k[0]), *map(float, k[1:6])] for k in b]


# ---------------------------------------------------------------- one cycle
def new_state(price):
    return dict(start=datetime.now(timezone.utc).isoformat(), cash=CAPITAL, pos={}, trades=[], events=[], last_bar={},
                day=dict(date="", start_eq=CAPITAL, n=0), halted=False, btc_qty=CAPITAL / price["BTC"])


def cycle():
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M")
    now_ms = int(time.time() * 1000)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = {}
    for c in COINS:
        rows = sl.klines(c, "1h", 400, closed_only=False)
        closed = rows[:-1] if rows[-1][0] + 3600 * 1000 > now_ms else rows
        b5 = bars5(c, rows)
        data[c] = dict(rows=closed, b5=b5, price=b5[-1][4] if b5 else rows[-1][4])
    price = {c: d["price"] for c, d in data.items()}
    st = json.load(open(STATE)) if os.path.exists(STATE) else new_state(price)
    value = lambda: st["cash"] + sum(p["qty"] * price[c] for c, p in st["pos"].items())

    def close(c, exit_px, why):
        p = st["pos"].pop(c)
        proceeds = p["qty"] * exit_px * (1 - SLIP) * (1 - FEE)
        st["cash"] += proceeds
        ret = proceeds / p["basis"] - 1
        hours = (now_ms - p["entry_ms"]) / 3600000
        st["trades"].append(dict(time=stamp, coin=c, side="SELL", price=exit_px, ret=ret, why=why, signal=p["signal"], hours=round(hours, 1)))
        print(f"  D SELL {c} @ {exit_px:g} {ret*100:+.2f}% ({why})")

    # new UTC day: reset the daily counters
    if st["day"]["date"] != today:
        st["day"] = dict(date=today, start_eq=value(), n=0)

    # ---- exits: walk the 5-minute candles since entry, first trigger wins, stop before target on the same candle
    for c in list(st["pos"]):
        p = st["pos"][c]
        done = False
        for t, o, h, lo, cl, _ in data[c]["b5"]:
            if t < p["entry_ms"] - 300000:
                continue
            if lo <= p["sl"]:
                close(c, min(p["sl"], o), "stop")
                done = True
                break
            if h >= p["tp"]:
                close(c, p["tp"], "target")
                done = True
                break
        if not done and now_ms - p["entry_ms"] >= HOLD_H * 3600000:
            close(c, price[c], "time")

    # ---- account breaker
    if not st["halted"] and value() <= CAPITAL * (1 - HALT_DD):
        for c in list(st["pos"]):
            close(c, price[c], "halt")
        st["halted"] = True
        st["events"].append(dict(time=stamp, text=f"D: توقف الحساب بعد هبوط {HALT_DD*100:.0f}% من البداية"))

    # ---- entries: once per new closed 1h bar per coin
    day_ok = (value() / st["day"]["start_eq"] - 1) > -DAY_LOSS_CAP
    for c in COINS:
        rows = data[c]["rows"]
        bar = rows[-1][0]
        if bar == st["last_bar"].get(c):
            continue
        st["last_bar"][c] = bar
        if st["halted"] or c in st["pos"] or len(st["pos"]) >= MAX_OPEN or st["day"]["n"] >= MAX_DAY or not day_ok:
            continue
        a, b = signal_series(rows)
        sig = "rsi_dip" if a[-1] else ("ema_pull" if b[-1] else None)
        if not sig:
            continue
        invest = min(st["cash"], value() * SIZE_FRAC)
        if invest < 50:
            continue
        px = price[c]
        fill = px * (1 + SLIP)
        qty = invest * (1 - FEE) / fill
        st["cash"] -= invest
        st["pos"][c] = dict(qty=qty, basis=invest, entry=px, tp=px * (1 + TP), sl=px * (1 - SL_), entry_ms=now_ms, signal=sig)
        st["day"]["n"] += 1
        st["trades"].append(dict(time=stamp, coin=c, side="BUY", price=px, signal=sig))
        print(f"  D BUY {c} @ {px:g} ({sig}), target {px*(1+TP):g}, stop {px*(1-SL_):g}")

    st["trades"] = st["trades"][-400:]
    st["events"] = st["events"][-20:]
    v = value()
    st["prices"] = dict(price)
    st["holdings"] = {c: dict(qty=p["qty"], value=p["qty"] * price[c], entry=p["entry"], tp=p["tp"], sl=p["sl"],
                              pnl=p["qty"] * price[c] * (1 - FEE) / p["basis"] - 1, signal=p["signal"]) for c, p in st["pos"].items()}
    st["equity"], st["btc_hold"] = v, st["btc_qty"] * price["BTC"]
    st["rules"] = dict(coins=COINS, tp=TP, stop=SL_, hold_hours=HOLD_H, size=SIZE_FRAC, max_open=MAX_OPEN, max_day=MAX_DAY,
                       day_loss_cap=DAY_LOSS_CAP, halt_dd=HALT_DD, fee=FEE, slip=SLIP)
    st["stats"] = make_stats(st["trades"])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "paper_equity_D.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "equity", "btc_hold"])
        w.writerow([stamp, round(v, 2), round(st["btc_hold"], 2)])
    json.dump(st, open(STATE, "w"), indent=1)
    print(f"[{stamp}] D account ${v:,.2f} ({(v/CAPITAL-1)*100:+.2f}%) open {list(st['pos']) or 'none'} today {st['day']['n']}/{MAX_DAY}")


# ---------------------------------------------------------------- history test of the same rules
def backtest():
    res, allr, allrnd, allst = {}, [], [], []
    for c in COINS:
        rows = sl.klines(c, "1h", 3000)
        n = len(rows)
        a, b = signal_series(rows)
        en = [x or y for x, y in zip(a, b)]
        sel = int(n * 0.7)
        r_all = hr.run(rows, en, TP, SL_, HOLD_H, sl.WARMUP, n)
        r_ex = hr.run(rows, en, TP, SL_, HOLD_H, sel, n)
        r_st = hr.run(rows, en, TP, SL_, HOLD_H, sel, n, cost_mult=3.0)
        r_rn = hr.run(rows, [True] * n, TP, SL_, HOLD_H, sel, n, step=7)
        days = max(1.0, (n - sl.WARMUP) / 24)
        res[c] = dict(candles=n, all=hr.stats(r_all), exam=hr.stats(r_ex), exam_x3_costs=hr.stats(r_st),
                      random_entries=hr.stats(r_rn), trades_per_day=len(r_all) / days)
        allr += r_ex
        allrnd += r_rn
        allst += r_st
        print(f"  {c}: exam {hr.stats(r_ex)}  random {hr.stats(r_rn)}  x3 costs {hr.stats(r_st)}")
    ex, rn, s3 = hr.stats(allr), hr.stats(allrnd), hr.stats(allst)
    out = dict(generated=datetime.now(timezone.utc).isoformat(), rules=dict(tp=TP, stop=SL_, hold_hours=HOLD_H),
               per_coin=res, exam_all=ex, random_all=rn, exam_x3_all=s3, boot_p=boot_mean_pos(allr),
               note="signals and parameters were fixed in advance, not tuned; exam = last 30% of each chart")
    json.dump(out, open(os.path.join(OUT, "backtest_D.json"), "w"), indent=1)
    print(f"ALL exam: n={ex['n']} win {ex['win']*100:.0f}% avg {ex['mean']*100:+.2f}%  | random entries: win {rn['win']*100:.0f}% avg {rn['mean']*100:+.2f}% "
          f"| costs x3: avg {s3['mean']*100:+.2f}% | P(avg>0) {out['boot_p']:.2f}")


if __name__ == "__main__":
    backtest() if "backtest" in sys.argv else cycle()
