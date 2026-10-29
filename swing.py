#!/usr/bin/env python3
"""Account F: patient swing trading on the big frames (paper only).  Rules fixed in advance:

  coins      EVERY coin the owner's screening site marks permissible that trades on Binance spot (religious exclusions
             only; the execution-cost exclusion of the quick accounts does not apply here: a few basis points do not
             matter against a 5% target)
  frames     weekly and daily decide the trend, 4h confirms and times the entry.  Everything is derived from the
             4h candles (daily/weekly resampled from completed UTC days/weeks), so the live rule and the backtest see
             exactly the same thing and never a candle that has not closed yet.
  trend      weekly: close > EMA10 and EMA10 rising.   daily: EMA20 > EMA50 and close > EMA50.
  confirm    4h: EMA20 > EMA50, close > EMA20, MACD histogram > 0, and a FRESH trigger on the last closed 4h bar:
             close breaks the 20-bar high, or closes back above EMA20 after a dip below it (pullback resumed).
  ranking    when several coins qualify, the strongest 30-day return goes first (at most 3 new entries per UTC day)
  stop       2 x daily ATR(14) below entry, capped at 8%; exit on a daily close below the daily EMA50 (trend gone)
  target     +5%.  When the price has been +5%: at every 4h close the momentum is re-examined (4h close > EMA20,
             MACD histogram > 0, daily EMA20 > EMA50).  Still positive: hold, with the stop raised to max(+3%,
             high - 2 x daily ATR).  Faded: sell at that close.  Time exit after 30 days.
  size       10% of the account per trade, at most 8 open positions; account halts at -20% from the start
  costs      OKX taker 0.10% + slippage 0.05% per side
  gate       no real money considered before 20 closed trades AND a positive average AND bootstrap P(average > 0) >= 0.9

`python swing.py`          one cycle (every 5 minutes from paper.yml; entries only on a new closed 4h bar)
`python swing.py backtest` the same rules on the last ~500 days of 4h candles; exam = last 30%; random-entry and costs x3 comparisons
State: lab_data/paper_F.json, paper_equity_F.csv, backtest_F.json.  Never trades real money, needs no keys.
"""
import csv, json, os, sys, time
from datetime import datetime, timezone

import spot_lab as sl
import policy
from scalp import wilson, boot_mean_pos, streak

OUT = sl.OUT
STATE = os.path.join(OUT, "paper_F.json")
CAPITAL = 5000.0
_REL = frozenset(c.upper() for c in policy.load().get("excluded", []))       # religious exclusions only
ALL_PERMISSIBLE = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC", "OP", "DOT", "TRX", "ATOM", "NEAR",
                   "XLM", "HBAR", "SUI", "APT", "ARB", "ALGO", "FIL", "ICP", "ETC", "BCH", "ZRO", "ZK", "ZEC", "TAO",
                   "RENDER", "QNT", "WLD"]
COINS = [c for c in ALL_PERMISSIBLE if c not in _REL]
TARGET, LOCK_AFTER_TARGET = 0.05, 0.03
ATR_MULT, STOP_CAP, HOLD_DAYS = 2.0, 0.08, 30
SIZE_FRAC, MAX_OPEN, MAX_DAY = 0.10, 8, 3
HALT_DD = 0.20
GATE_TRADES, GATE_BOOT = 20, 0.90
FEE, SLIP = sl.FEE, sl.SLIP
H4 = 4 * 3600 * 1000
DAY = 86400 * 1000


# ---------------------------------------------------------------- frames derived from 4h candles, no look-ahead
def resample(rows, span):
    """Completed buckets of `span` ms: [t_start, o, h, l, c, v]; the last bucket is dropped if still open at rows[-1]."""
    out, cur = [], None
    for r in rows:
        b = r[0] // span * span
        if cur is None or b != cur[0]:
            if cur is not None:
                out.append(cur)
            cur = [b, r[1], r[2], r[3], r[4], r[5]]
        else:
            cur[2], cur[3], cur[4], cur[5] = max(cur[2], r[2]), min(cur[3], r[3]), r[4], cur[5] + r[5]
    if cur is not None and rows and rows[-1][0] + H4 >= cur[0] + span:
        out.append(cur)                                   # bucket complete at the last 4h close
    return out


def frames(rows4):
    """Per 4h bar i: the trend state from the frames completed BEFORE that bar closed, plus the 4h confirmation."""
    n = len(rows4)
    c4 = [r[4] for r in rows4]
    e20, e50 = sl.ema(c4, 20), sl.ema(c4, 50)
    mh = sl.macd_hist(c4)
    hi20 = [max(r[2] for r in rows4[max(0, i - 20):i]) if i >= 20 else None for i in range(n)]
    daily = resample(rows4, DAY)
    weekly = resample(rows4, 7 * DAY)
    dc = [d[4] for d in daily]
    de20, de50, datr = sl.ema(dc, 20), sl.ema(dc, 50), sl.atr(daily)
    wc = [w[4] for w in weekly]
    we10 = sl.ema(wc, 10)
    di = wi = -1
    out = []
    for i, r in enumerate(rows4):
        end = r[0] + H4                                   # this 4h bar closes here
        while di + 1 < len(daily) and daily[di + 1][0] + DAY <= end:
            di += 1
        while wi + 1 < len(weekly) and weekly[wi + 1][0] + 7 * DAY <= end:
            wi += 1
        d_ok = di >= 60 and de20[di] > de50[di] and dc[di] > de50[di]
        w_ok = wi >= 12 and wc[wi] > we10[wi] and we10[wi] > we10[wi - 1]
        c_ok = i > 50 and e20[i] > e50[i] and c4[i] > e20[i] and mh[i] > 0
        trigger = i > 20 and ((hi20[i] is not None and c4[i] > hi20[i]) or (c4[i - 1] < e20[i - 1] and c4[i] > e20[i]))
        mom_ok = i > 50 and c4[i] > e20[i] and mh[i] > 0 and di >= 60 and de20[di] > de50[di]
        trend_gone = di >= 60 and dc[di] < de50[di]
        ret30 = c4[i] / c4[i - 180] - 1 if i >= 180 else 0.0
        out.append(dict(entry=d_ok and w_ok and c_ok and trigger, mom_ok=mom_ok, trend_gone=trend_gone,
                        atr=datr[di] if di >= 14 else None, ret30=ret30))
    return out


def init_stop(px, atr_d):
    return px - min(ATR_MULT * atr_d, STOP_CAP * px) if atr_d else px * (1 - STOP_CAP)


# ---------------------------------------------------------------- statistics (same shape as D/E)
def make_stats(trades):
    sells = [t for t in trades if t["side"] == "SELL" and t.get("ret") is not None]
    r = [t["ret"] for t in sells]
    n = len(r)
    out = dict(trades=n, gate_trades=GATE_TRADES)
    if not n:
        out["gate"] = f"لا صفقات مغلقة بعد (0 من {GATE_TRADES})"
        return out
    wins, losses = [x for x in r if x > 0], [x for x in r if x <= 0]
    lo, hi = wilson(len(wins), n)
    aw = sum(wins) / len(wins) if wins else 0.0
    al = sum(losses) / len(losses) if losses else 0.0
    be = (-al / (aw - al)) if (aw - al) > 0 and al < 0 else None
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None
    by = {}
    for key in ("why", "coin"):
        d = {}
        for t in sells:
            d.setdefault(t.get(key) or "-", []).append(t["ret"])
        by[key] = {k: dict(n=len(v), win=sum(x > 0 for x in v) / len(v), mean=sum(v) / len(v)) for k, v in d.items()}
    mean, bp = sum(r) / n, boot_mean_pos(r)
    gate = (f"مغلقة {n} من {GATE_TRADES} المطلوبة" if n < GATE_TRADES else
            ("اجتاز شرط العدد، لكن المتوسط سالب" if mean <= 0 else
             ("اجتاز شرط العدد والمتوسط، وثقة الربح دون 90%" if bp < GATE_BOOT else "اجتاز كل الشروط (يبقى قراراً بشرياً ولا ضمان)")))
    out.update(win_rate=len(wins) / n, win_ci=[lo, hi], avg_win=aw, avg_loss=al, expectancy=mean, breakeven_win=be,
               profit_factor=pf, boot_p=bp, worst=min(r), loss_streak=streak(r),
               avg_hours=sum(t.get("hours", 0) for t in sells) / n, by=by, gate=gate)
    return out


# ---------------------------------------------------------------- live cycle
def new_state(price):
    return dict(start=datetime.now(timezone.utc).isoformat(), cash=CAPITAL, pos={}, trades=[], events=[], last_bar={},
                day=dict(date="", n=0), halted=False, btc_qty=CAPITAL / price["BTC"])


def cycle():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    now_ms = int(time.time() * 1000)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    held = list(json.load(open(STATE)).get("pos", {})) if os.path.exists(STATE) else []
    data = {}
    for c in sorted(set(COINS) | set(held)):
        try:
            rows = sl.klines(c, "4h", 1500, closed_only=False)
        except Exception as e:
            print(f"  ! {c}: {e}")
            continue
        live_px = rows[-1][4]
        closed = rows[:-1] if rows[-1][0] + H4 > now_ms else rows
        if len(closed) < 300:
            continue
        data[c] = dict(rows=closed, price=live_px, fr=frames(closed)[-1])
    if "BTC" not in data:
        print("  F: no data this cycle")
        return
    price = {c: d["price"] for c, d in data.items()}
    st = json.load(open(STATE)) if os.path.exists(STATE) else new_state(price)
    value = lambda: st["cash"] + sum(p["qty"] * price.get(c, p["entry"]) for c, p in st["pos"].items())

    def close(c, exit_px, why):
        p = st["pos"].pop(c)
        proceeds = p["qty"] * exit_px * (1 - SLIP) * (1 - FEE)
        st["cash"] += proceeds
        ret = proceeds / p["basis"] - 1
        st["trades"].append(dict(time=stamp, coin=c, side="SELL", price=exit_px, ret=ret, why=why,
                                 hours=round((now_ms - p["entry_ms"]) / 3600000, 1), high=p.get("high"), reached_target=p.get("reached", False)))
        print(f"  F SELL {c} @ {exit_px:g} {ret*100:+.2f}% ({why})")

    if st["day"]["date"] != today:
        st["day"] = dict(date=today, n=0)

    # ---- exits: stop on the live price; momentum review and trend exit only on a NEW closed 4h bar
    for c in list(st["pos"]):
        p = st["pos"][c]
        if c not in data:
            continue
        px = price[c]
        p["high"] = max(p.get("high", p["entry"]), px)
        if px <= p["sl"]:
            close(c, px, "lock" if p["sl"] >= p["entry"] * (1 + LOCK_AFTER_TARGET) - 1e-9 else "stop")
            continue
        bar = data[c]["rows"][-1][0]
        new_bar = bar != p.get("last_bar")
        p["last_bar"] = bar
        fr = data[c]["fr"]
        if p["high"] >= p["entry"] * (1 + TARGET):
            p["reached"] = True
            p["sl"] = max(p["sl"], p["entry"] * (1 + LOCK_AFTER_TARGET), p["high"] - ATR_MULT * (fr["atr"] or 0))
            if new_bar and not fr["mom_ok"]:
                close(c, px, "momentum")
                continue
        if new_bar and fr["trend_gone"]:
            close(c, px, "trend")
            continue
        if now_ms - p["entry_ms"] >= HOLD_DAYS * DAY:
            close(c, px, "time")

    for c in list(st["pos"]):
        if c in _REL:
            close(c, price.get(c, st["pos"][c]["entry"]), "policy")
            st["events"].append(dict(time=stamp, text=f"F: أُغلقت صفقة {c} لأن العملة مستبعدة بسياسة العملات"))

    if not st["halted"] and value() <= CAPITAL * (1 - HALT_DD):
        for c in list(st["pos"]):
            close(c, price[c], "halt")
        st["halted"] = True
        st["events"].append(dict(time=stamp, text=f"F: توقف الحساب بعد هبوط {HALT_DD*100:.0f}% من البداية"))

    # ---- entries on a new closed 4h bar, strongest 30-day return first
    cands = []
    for c in COINS:
        if c not in data:
            continue
        bar = data[c]["rows"][-1][0]
        if bar == st["last_bar"].get(c):
            continue
        st["last_bar"][c] = bar
        if st["halted"] or c in st["pos"]:
            continue
        fr = data[c]["fr"]
        if fr["entry"]:
            cands.append((fr["ret30"], c))
    for ret30, c in sorted(cands, reverse=True):
        if len(st["pos"]) >= MAX_OPEN or st["day"]["n"] >= MAX_DAY:
            break
        invest = min(st["cash"], value() * SIZE_FRAC)
        if invest < 50:
            continue
        px = price[c]
        qty = invest * (1 - FEE) / (px * (1 + SLIP))
        st["cash"] -= invest
        st["pos"][c] = dict(qty=qty, basis=invest, entry=px, sl=init_stop(px, data[c]["fr"]["atr"]), entry_ms=now_ms,
                            high=px, reached=False, last_bar=data[c]["rows"][-1][0])
        st["day"]["n"] += 1
        st["trades"].append(dict(time=stamp, coin=c, side="BUY", price=px, ret30=ret30))
        print(f"  F BUY {c} @ {px:g} (30d {ret30*100:+.1f}%), stop {st['pos'][c]['sl']:g}, target {px*(1+TARGET):g}")

    st["trades"], st["events"] = st["trades"][-400:], st["events"][-20:]
    v = value()
    st["prices"] = dict(price)
    st["holdings"] = {c: dict(qty=p["qty"], value=p["qty"] * price.get(c, p["entry"]), entry=p["entry"], tp=p["entry"] * (1 + TARGET), sl=p["sl"],
                              pnl=p["qty"] * price.get(c, p["entry"]) * (1 - FEE) / p["basis"] - 1, signal="swing",
                              protected=p["sl"] > p["entry"], reached=p.get("reached", False)) for c, p in st["pos"].items()}
    st["equity"], st["btc_hold"] = v, st["btc_qty"] * price["BTC"]
    st["rules"] = dict(coins=COINS, tp=TARGET, stop=STOP_CAP, hold_hours=HOLD_DAYS * 24, size=SIZE_FRAC, max_open=MAX_OPEN, max_day=MAX_DAY,
                       day_loss_cap=0, halt_dd=HALT_DD, fee=FEE, slip=SLIP, frames="1w+1d trend, 4h confirm",
                       swing=dict(target=TARGET, lock=LOCK_AFTER_TARGET, atr_mult=ATR_MULT, stop_cap=STOP_CAP, hold_days=HOLD_DAYS))
    st["stats"] = make_stats(st["trades"])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "paper_equity_F.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "equity", "btc_hold"])
        w.writerow([stamp, round(v, 2), round(st["btc_hold"], 2)])
    json.dump(st, open(STATE, "w"), indent=1)
    print(f"[{stamp}] F account ${v:,.2f} ({(v/CAPITAL-1)*100:+.2f}%) open {list(st['pos']) or 'none'} candidates {len(cands)}")


# ---------------------------------------------------------------- history test on 4h candles
def run(rows, fr, start, end, cost_mult=1.0, random_step=0):
    fee, slip = FEE * cost_mult, SLIP * cost_mult
    rets, i, whys = [], start, []
    while i < end - 2:
        go = (i % random_step == 0) if random_step else fr[i]["entry"]
        if not go:
            i += 1
            continue
        px = rows[i + 1][1] * (1 + slip)
        stp, high, reached, exit_px, why, j = init_stop(px, fr[i]["atr"]), px, False, None, None, i + 1
        for j in range(i + 1, min(end, i + 1 + HOLD_DAYS * 6)):
            lo, hi, cl = rows[j][3], rows[j][2], rows[j][4]
            if lo <= stp:
                exit_px, why = stp * (1 - slip), ("lock" if stp >= px * (1 + LOCK_AFTER_TARGET) - 1e-9 else "stop")
                break
            high = max(high, hi)
            if high >= px * (1 + TARGET):
                reached = True
                stp = max(stp, px * (1 + LOCK_AFTER_TARGET), high - ATR_MULT * (fr[j]["atr"] or 0))
                if not fr[j]["mom_ok"]:
                    exit_px, why = cl * (1 - slip), "momentum"
                    break
            if fr[j]["trend_gone"]:
                exit_px, why = cl * (1 - slip), "trend"
                break
        if exit_px is None:
            exit_px, why = rows[j][4] * (1 - slip), "time"
        rets.append(exit_px * (1 - fee) / (px * (1 + fee)) - 1)
        whys.append(why)
        i = j + 1
    return rets, whys


def stats(r):
    if not r:
        return dict(n=0, win=0.0, mean=0.0, worst=0.0)
    return dict(n=len(r), win=sum(x > 0 for x in r) / len(r), mean=sum(r) / len(r), worst=min(r))


def backtest():
    res, ex_all, rn_all, st_all, why_all = {}, [], [], [], {}
    for c in COINS:
        try:
            rows = sl.klines(c, "4h", 3000)
        except Exception as e:
            print(f"  ! {c}: {e}")
            continue
        n = len(rows)
        if n < 900:
            continue
        fr = frames(rows)
        sel = int(n * 0.7)
        r_all, _ = run(rows, fr, 300, n)
        r_ex, w_ex = run(rows, fr, sel, n)
        r_st, _ = run(rows, fr, sel, n, cost_mult=3.0)
        r_rn, _ = run(rows, fr, sel, n, random_step=30)
        for w, x in zip(w_ex, r_ex):
            why_all.setdefault(w, []).append(x)
        res[c] = dict(candles=n, all=stats(r_all), exam=stats(r_ex), exam_x3_costs=stats(r_st), random_entries=stats(r_rn),
                      trades_per_month=len(r_all) / max(1.0, (n - 300) / 180))
        ex_all += r_ex
        rn_all += r_rn
        st_all += r_st
        print(f"  {c}: exam {stats(r_ex)}  random {stats(r_rn)}  x3 {stats(r_st)}")
    ex, rn, s3 = stats(ex_all), stats(rn_all), stats(st_all)
    out = dict(generated=datetime.now(timezone.utc).isoformat(),
               rules=dict(target=TARGET, lock=LOCK_AFTER_TARGET, atr_mult=ATR_MULT, stop_cap=STOP_CAP, hold_days=HOLD_DAYS),
               per_coin=res, exam_all=ex, random_all=rn, exam_x3_all=s3, boot_p=boot_mean_pos(ex_all),
               by_why={k: stats(v) for k, v in why_all.items()},
               note="rules fixed in advance; 4h candles with daily/weekly resampled from completed buckets; exam = last 30%")
    json.dump(out, open(os.path.join(OUT, "backtest_F.json"), "w"), indent=1)
    print(f"ALL exam: n={ex['n']} win {ex['win']*100:.0f}% avg {ex['mean']*100:+.2f}% | random: win {rn['win']*100:.0f}% avg {rn['mean']*100:+.2f}% "
          f"| costs x3: avg {s3['mean']*100:+.2f}% | P(avg>0) {out['boot_p']:.2f} | exits {{ {', '.join(f'{k}: {len(v)}' for k, v in why_all.items())} }}")


if __name__ == "__main__":
    backtest() if "backtest" in sys.argv else cycle()
