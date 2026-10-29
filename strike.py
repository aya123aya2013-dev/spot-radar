#!/usr/bin/env python3
"""Account E: daily "take 2% and leave" session with profit protection (paper only).  Rules fixed in advance:

  coins      liquid coins the owner's screening site marks permissible (coin policy applied on top)
  entry      closed 1h bar, at least ONE of: RSI(14) < 30 in an uptrend (price > EMA200); pullback to EMA20 inside an
             EMA50 > EMA200 uptrend with RSI < 50; long lower wick (>= 60% of the range) rejected in an uptrend.
             A coin that already traded today is only re-entered on a STRONG signal: two of the three at once.
  exits      the owner's rule: 2% is the box.  Initial stop -3%.  At +1% the stop moves to entry +0.2% (fees covered).
             When the price reaches +2% the stop is LOCKED at +2% and the trade is left to run; from +3% the stop trails
             1% below the highest price seen (the "sell at a suitable time").  48h time exit.  No fixed take-profit.
  size       15% of the account per trade, at most 5 open positions and 6 new trades per UTC day; when several coins
             signal on the same bar, the ones with the stronger signal and the better 7-day momentum go first
  guards     no new trades after -1.5% of the day's starting equity; account halts at -10% from the start
  costs      OKX taker fee 0.10% + slippage 0.05% per side
  gate       no real money considered before 30 closed trades AND a positive average AND bootstrap P(average > 0) >= 0.9

`python strike.py`          one cycle (every 5 minutes from paper.yml)
`python strike.py backtest` the same rules on history: exam = last 30% of each chart, random entries, costs x3,
                            and the same signals with the plain fixed exits (so the protection rule is judged alone)
State: lab_data/paper_E.json, paper_equity_E.csv, backtest_E.json.  Never trades real money, needs no keys.
"""
import csv, json, os, sys, time
from datetime import datetime, timezone

import spot_lab as sl
import hitrate as hr
import policy
from scalp import wilson, boot_mean_pos, streak, bars5

OUT = sl.OUT
STATE = os.path.join(OUT, "paper_E.json")
CAPITAL = 5000.0
# every coin the screening site marks permissible AND that trades on Binance spot (data source); policy applied on top
COINS = policy.allowed(["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC", "OP", "DOT", "TRX", "ATOM", "NEAR",
                        "XLM", "HBAR", "SUI", "APT", "ARB", "ALGO", "FIL", "ICP", "ETC", "BCH", "ZRO", "TAO", "RENDER", "QNT", "WLD", "ZEC"])
TP, SL_, HOLD_H = 0.02, 0.03, 48         # TP is the LOCK level, not a take-profit
BE_TRIGGER, BE_LOCK = 0.010, 0.002       # after +1.0%: stop to entry +0.2%
LOCK_TRIGGER, LOCK = 0.020, 0.020        # after +2.0%: stop locked at +2.0%
TRAIL_TRIGGER, TRAIL = 0.030, 0.010      # after +3.0%: stop trails 1% under the high
SIZE_FRAC, MAX_OPEN, MAX_DAY = 0.15, 5, 6
DAY_LOSS_CAP, HALT_DD = 0.015, 0.10
GATE_TRADES, GATE_BOOT = 30, 0.90
# learning rule, fixed in advance: a coin or a signal type with >= PAUSE_MIN closed trades, a negative average and
# bootstrap P(average > 0) < PAUSE_P is paused for PAUSE_DAYS, then gets another chance.  Nothing else adapts.
PAUSE_MIN, PAUSE_P, PAUSE_DAYS = 12, 0.15, 7
FEE, SLIP = sl.FEE, sl.SLIP


# ---------------------------------------------------------------- signals (research definitions, fixed parameters)
def signal_score(rows):
    """Per bar: how many of the three entry signals fire (0..3)."""
    a = hr.e_rsi_dip(rows, th=30, trend=True)
    b = hr.e_ema_pull(rows, band=1.0, rmax=50)
    c = hr.e_wick(rows, wick=0.6, trend=True)
    return [int(x) + int(y) + int(z) for x, y, z in zip(a, b, c)]


# ---------------------------------------------------------------- exit rule shared by live and backtest
def protected_stop(entry, high_seen, stop):
    """Stop level after profit protection, given the highest price seen since entry."""
    gain = high_seen / entry - 1
    if gain >= TRAIL_TRIGGER:
        stop = max(stop, high_seen * (1 - TRAIL))
    if gain >= LOCK_TRIGGER:
        stop = max(stop, entry * (1 + LOCK))
    if gain >= BE_TRIGGER:
        stop = max(stop, entry * (1 + BE_LOCK))
    return stop


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
    for key in ("why", "signal", "coin"):
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


# ---------------------------------------------------------------- one live cycle
def new_state(price):
    return dict(start=datetime.now(timezone.utc).isoformat(), cash=CAPITAL, pos={}, trades=[], events=[], last_bar={},
                day=dict(date="", start_eq=CAPITAL, n=0, traded=[]), halted=False, btc_qty=CAPITAL / price["BTC"])


def cycle():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    now_ms = int(time.time() * 1000)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    held = list(json.load(open(STATE)).get("pos", {})) if os.path.exists(STATE) else []
    data = {}
    for c in sorted(set(COINS) | set(held)):
        try:
            rows = sl.klines(c, "1h", 400, closed_only=False)
        except Exception as e:
            print(f"  ! {c}: {e}")
            continue
        closed = rows[:-1] if rows[-1][0] + 3600 * 1000 > now_ms else rows
        b5 = bars5(c, rows)
        data[c] = dict(rows=closed, b5=b5, price=b5[-1][4] if b5 else rows[-1][4])
    if "BTC" not in data:
        print("  E: no data this cycle")
        return
    price = {c: d["price"] for c, d in data.items()}
    st = json.load(open(STATE)) if os.path.exists(STATE) else new_state(price)
    st["day"].setdefault("traded", [])
    value = lambda: st["cash"] + sum(p["qty"] * price.get(c, p["entry"]) for c, p in st["pos"].items())

    def close(c, exit_px, why):
        p = st["pos"].pop(c)
        proceeds = p["qty"] * exit_px * (1 - SLIP) * (1 - FEE)
        st["cash"] += proceeds
        ret = proceeds / p["basis"] - 1
        hours = (now_ms - p["entry_ms"]) / 3600000
        st["trades"].append(dict(time=stamp, coin=c, side="SELL", price=exit_px, ret=ret, why=why, signal=p["signal"],
                                 hours=round(hours, 1), high=p.get("high"), protected=p.get("sl", 0) > p["entry"]))
        print(f"  E SELL {c} @ {exit_px:g} {ret*100:+.2f}% ({why})")

    # one-off repair of a bookkeeping bug (2026-10-29 00:20 UTC): with signal "single" paused, two candidates had their
    # cash deducted and then skipped, so 27.75% of the account vanished without any trade and the -10% breaker fired.
    # The money is restored exactly (cash / (1 - 0.15)^2) and the breaker reset; no trade result is touched.
    if not st.get("repair_20261029") and st.get("halted") and not st["pos"] and abs(st["cash"] - 3570.5616) < 1:
        before = st["cash"]
        st["cash"] = st["cash"] / (1 - SIZE_FRAC) ** 2
        st["halted"] = False
        st["repair_20261029"] = dict(cash_before=before, cash_after=st["cash"])
        st["events"].append(dict(time=stamp, text=f"E: تصحيح خطأ برمجي: خُصم ${st['cash']-before:,.2f} بلا صفقة عند إيقاف الإشارة، أُعيد المبلغ وأُلغي التوقف الناتج عنه"))

    if st["day"]["date"] != today:
        st["day"] = dict(date=today, start_eq=value(), n=0, traded=[])

    # ---- exits on the 5-minute candles since entry, stop first on a candle that touches both
    for c in list(st["pos"]):
        p = st["pos"][c]
        if c not in data:
            continue
        done = False
        for t, o, h, lo, cl, _ in data[c]["b5"]:
            if t < p["entry_ms"] - 300000:
                continue
            if lo <= p["sl"]:
                why = "locked" if p["sl"] >= p["entry"] * (1 + LOCK) - 1e-9 else ("protect" if p["sl"] > p["entry"] else "stop")
                close(c, min(p["sl"], o) if why == "stop" else p["sl"], why)
                done = True
                break
            p["high"] = max(p.get("high", p["entry"]), h)
            p["sl"] = protected_stop(p["entry"], p["high"], p["sl"])
        if not done and now_ms - p["entry_ms"] >= HOLD_H * 3600000:
            close(c, price[c], "time")

    for c in list(st["pos"]):
        if c in policy.EXCLUDED:
            close(c, price.get(c, st["pos"][c]["entry"]), "policy")
            st["events"].append(dict(time=stamp, text=f"E: أُغلقت صفقة {c} لأن العملة مستبعدة بسياسة العملات"))

    if not st["halted"] and value() <= CAPITAL * (1 - HALT_DD):
        for c in list(st["pos"]):
            close(c, price[c], "halt")
        st["halted"] = True
        st["events"].append(dict(time=stamp, text=f"E: توقف الحساب بعد هبوط {HALT_DD*100:.0f}% من البداية"))

    # ---- learning rule: pause coins / signal types that are losing with confidence (see PAUSE_* above)
    st.setdefault("paused", {})
    sells = [t for t in st["trades"] if t["side"] == "SELL" and t.get("ret") is not None]
    for field in ("coin", "signal"):
        groups = {}
        for t in sells:
            groups.setdefault(t.get(field), []).append(t["ret"])
        for k, r in groups.items():
            key = f"{field}:{k}"
            until = st["paused"].get(key, 0)
            if until and now_ms < until:
                continue
            if len(r) >= PAUSE_MIN and sum(r) / len(r) < 0 and boot_mean_pos(r) < PAUSE_P and not until:
                st["paused"][key] = now_ms + PAUSE_DAYS * 86400000
                st["events"].append(dict(time=stamp, text=f"E: إيقاف {('العملة' if field == 'coin' else 'الإشارة')} {k} لمدة {PAUSE_DAYS} أيام: متوسط {sum(r)/len(r)*100:+.2f}% في {len(r)} صفقة"))
            elif until and now_ms >= until:
                st["paused"][key] = 0
    paused_now = {k for k, v in st["paused"].items() if v and now_ms < v}

    # ---- entries, once per new closed 1h bar per coin
    day_ok = (value() / st["day"]["start_eq"] - 1) > -DAY_LOSS_CAP
    cands = []
    for c in COINS:
        if c not in data or f"coin:{c}" in paused_now:
            continue
        rows = data[c]["rows"]
        bar = rows[-1][0]
        if bar == st["last_bar"].get(c):
            continue
        st["last_bar"][c] = bar
        if st["halted"] or c in st["pos"]:
            continue
        score = signal_score(rows)[-1]
        need = 2 if c in st["day"]["traded"] else 1
        if score < need:
            continue
        mom7 = rows[-1][4] / rows[-168][4] - 1 if len(rows) > 168 else 0.0
        cands.append((score, mom7, c))
    for score, mom7, c in sorted(cands, reverse=True):
        if len(st["pos"]) >= MAX_OPEN or st["day"]["n"] >= MAX_DAY or not day_ok:
            break
        sig = "strong" if score >= 2 else "single"
        if f"signal:{sig}" in paused_now:
            continue
        invest = min(st["cash"], value() * SIZE_FRAC)
        if invest < 50:
            continue
        px = price[c]
        qty = invest * (1 - FEE) / (px * (1 + SLIP))
        st["cash"] -= invest
        st["pos"][c] = dict(qty=qty, basis=invest, entry=px, tp=px * (1 + TP), sl=px * (1 - SL_), entry_ms=now_ms, signal=sig, high=px)
        st["day"]["n"] += 1
        st["day"]["traded"].append(c)
        st["trades"].append(dict(time=stamp, coin=c, side="BUY", price=px, signal=sig))
        print(f"  E BUY {c} @ {px:g} ({sig}, 7d {mom7*100:+.1f}%), lock at {px*(1+LOCK):g}, stop {px*(1-SL_):g}")

    st["trades"], st["events"] = st["trades"][-400:], st["events"][-20:]
    v = value()
    st["prices"] = dict(price)
    st["holdings"] = {c: dict(qty=p["qty"], value=p["qty"] * price.get(c, p["entry"]), entry=p["entry"], tp=p["tp"], sl=p["sl"],
                              pnl=p["qty"] * price.get(c, p["entry"]) * (1 - FEE) / p["basis"] - 1, signal=p["signal"],
                              protected=p["sl"] > p["entry"]) for c, p in st["pos"].items()}
    st["equity"], st["btc_hold"] = v, st["btc_qty"] * price["BTC"]
    st["rules"] = dict(coins=COINS, tp=TP, stop=SL_, hold_hours=HOLD_H, size=SIZE_FRAC, max_open=MAX_OPEN, max_day=MAX_DAY,
                       day_loss_cap=DAY_LOSS_CAP, halt_dd=HALT_DD, fee=FEE, slip=SLIP,
                       protect=dict(be_trigger=BE_TRIGGER, be_lock=BE_LOCK, lock_trigger=LOCK_TRIGGER, lock=LOCK, trail_trigger=TRAIL_TRIGGER, trail=TRAIL), reentry="strong signal only", take_profit=False,
                       learn=dict(min_trades=PAUSE_MIN, p=PAUSE_P, days=PAUSE_DAYS), paused=sorted(paused_now))
    st["stats"] = make_stats(st["trades"])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "paper_equity_E.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "equity", "btc_hold"])
        w.writerow([stamp, round(v, 2), round(st["btc_hold"], 2)])
    json.dump(st, open(STATE, "w"), indent=1)
    print(f"[{stamp}] E account ${v:,.2f} ({(v/CAPITAL-1)*100:+.2f}%) open {list(st['pos']) or 'none'} today {st['day']['n']}/{MAX_DAY}")


# ---------------------------------------------------------------- history test: same rules on 1h candles
def run_protected(rows, score, start, end, cost_mult=1.0, min_score=1, protect=True):
    """Net trade returns.  Re-entry into the same coin on the same UTC day needs a strong signal, as live."""
    fee, slip = FEE * cost_mult, SLIP * cost_mult
    rets, i, traded_day = [], start, (None, False)
    while i < end - 2:
        day = rows[i][0] // 86400000
        need = 2 if (traded_day[0] == day and traded_day[1]) else min_score
        if score[i] < need:
            i += 1
            continue
        px = rows[i + 1][1] * (1 + slip)
        tgt, stp, high, exit_px, j = px * (1 + TP), px * (1 - SL_), px, None, i + 1
        for j in range(i + 1, min(end, i + 1 + (HOLD_H if protect else 24))):
            lo, hi = rows[j][3], rows[j][2]
            if lo <= stp:
                exit_px = stp * (1 - slip)
                break
            if not protect and hi >= tgt:                 # plain variant: fixed +2% take-profit, -3% stop, 24h
                exit_px = tgt * (1 - slip)
                break
            high = max(high, hi)
            if protect:
                stp = protected_stop(px, high, stp)
        if exit_px is None:
            exit_px = rows[j][4] * (1 - slip)
        rets.append(exit_px * (1 - fee) / (px * (1 + fee)) - 1)
        traded_day = (day, True)
        i = j + 1
    return rets


def backtest():
    res, ex_all, rn_all, st_all, plain_all = {}, [], [], [], []
    for c in COINS:
        try:
            rows = sl.klines(c, "1h", 3000)
        except Exception as e:
            print(f"  ! {c}: {e}")
            continue
        n = len(rows)
        score = signal_score(rows)
        sel = int(n * 0.7)
        r_all = run_protected(rows, score, sl.WARMUP, n)
        r_ex = run_protected(rows, score, sel, n)
        r_st = run_protected(rows, score, sel, n, cost_mult=3.0)
        r_pl = run_protected(rows, score, sel, n, protect=False)
        r_rn = hr.run(rows, [True] * n, TP, SL_, HOLD_H, sel, n, step=7)
        days = max(1.0, (n - sl.WARMUP) / 24)
        res[c] = dict(candles=n, all=hr.stats(r_all), exam=hr.stats(r_ex), exam_x3_costs=hr.stats(r_st),
                      exam_without_protection=hr.stats(r_pl), random_entries=hr.stats(r_rn), trades_per_day=len(r_all) / days)
        ex_all += r_ex
        rn_all += r_rn
        st_all += r_st
        plain_all += r_pl
        print(f"  {c}: exam {hr.stats(r_ex)}  no-protection {hr.stats(r_pl)}  random {hr.stats(r_rn)}  x3 {hr.stats(r_st)}")
    ex, rn, s3, pl = hr.stats(ex_all), hr.stats(rn_all), hr.stats(st_all), hr.stats(plain_all)
    out = dict(generated=datetime.now(timezone.utc).isoformat(),
               rules=dict(lock=LOCK, stop=SL_, hold_hours=HOLD_H, be_trigger=BE_TRIGGER, be_lock=BE_LOCK, lock_trigger=LOCK_TRIGGER, trail_trigger=TRAIL_TRIGGER, trail=TRAIL, take_profit=False),
               per_coin=res, exam_all=ex, random_all=rn, exam_x3_all=s3, plain_all=pl, boot_p=boot_mean_pos(ex_all),
               note="signals and parameters fixed in advance; exam = last 30% of each chart; plain_all = same entries with a fixed +2% take-profit / -3% stop / 24h")
    json.dump(out, open(os.path.join(OUT, "backtest_E.json"), "w"), indent=1)
    print(f"ALL exam: n={ex['n']} win {ex['win']*100:.0f}% avg {ex['mean']*100:+.2f}% | without protection: win {pl['win']*100:.0f}% avg {pl['mean']*100:+.2f}% "
          f"| random: avg {rn['mean']*100:+.2f}% | costs x3: avg {s3['mean']*100:+.2f}% | P(avg>0) {out['boot_p']:.2f}")


if __name__ == "__main__":
    backtest() if "backtest" in sys.argv else cycle()
