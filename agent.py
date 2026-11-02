#!/usr/bin/env python3
"""Account G: the trading agent (paper only).  A portfolio manager, not a signal bot.  Rules fixed in advance:

  1. Read the market (every UTC day, on the closed daily candle)
       BTC vs its EMA50/EMA200 and breadth = share of the permissible coins closing above their own EMA50.
       bull   BTC > EMA200 and EMA50 > EMA200      budget 100% if breadth >= 50%, else 50%
       early  BTC > EMA200 (EMA50 still below)      budget 50% if breadth >= 50%, else 25%
       turn   BTC > EMA50, breadth >= 50%           budget 25%
       bear   BTC < EMA200 and EMA50 < EMA200       budget 0%   (cash is a position)
       weak   anything else                         budget 0%
     The budget is the most of the account allowed in coins.
  2. Leaders: every permissible coin ranked by strength against BTC (63-day return minus BTC's, plus closeness to its
     one-year high), kept only if above its EMA50.  The top 8 are the watchlist, published every day, even in a bear
     market, so the agent is ready for the turn.
  3. Entries (only when the budget allows): a watchlist coin with close > EMA20 > EMA50 that either breaks its 20-day
     high or resumes after a pullback to EMA20 (closed under it in the last 5 days, back above today).
  4. Size: each position = min(budget / 4, 0.75% / the coin's daily volatility) of the account: calm coins get more,
     wild coins less; total never above the budget; at most 6 positions.
  5. Exits: initial stop 2.5 ATR under entry; then a trailing stop 3 ATR under the highest close.  At +2R (R = entry -
     initial stop) half is sold and the stop moves to break-even.  If the budget falls, the weakest positions are cut
     until the account is back inside it (a bear regime sells everything).
  6. Every day a decision report in Arabic: the regime and why, the budget, the watchlist, what was done and why,
     what would change the view.  New entries also come as "shadow orders": the exact OKX spot limit order a person
     would place by hand (pair, price, size for a given capital, stop).  Nothing is ever sent to an exchange.

`python agent.py`            one cycle (every 5 minutes: live stop checks; decisions once per new UTC day)
`python agent.py backtest`   the same rules day by day since 2019 against BTC hold and weekly DCA, plus the ablation
                             "same agent without the market reading" (budget always 100%) to show what the regime adds
State: lab_data/paper_G.json, paper_equity_G.csv, agent_report.json, backtest_G.json.  Paper only, no keys.
"""
import csv, json, math, os, sys, time
from datetime import datetime, timezone

import spot_lab as sl
import policy

OUT = sl.OUT
STATE = os.path.join(OUT, "paper_G.json")
REPORT = os.path.join(OUT, "agent_report.json")
CAPITAL = 5000.0
_REL = frozenset(c.upper() for c in policy.load().get("excluded", []))
UNIVERSE = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC", "OP", "DOT", "TRX", "ATOM", "NEAR",
            "XLM", "HBAR", "SUI", "APT", "ARB", "ALGO", "FIL", "ICP", "ETC", "BCH", "ZRO", "ZK", "ZEC", "TAO",
            "RENDER", "QNT", "WLD"]
COINS = [c for c in UNIVERSE if c not in _REL]
WATCH_N, MAX_POS, SLOT = 8, 6, 4
VOL_UNIT = 0.0075
INIT_ATR, TRAIL_ATR, PARTIAL_R = 2.5, 3.0, 2.0
TARGET2_R = 4.0          # second sell target (guide for the remaining half; the trailing stop decides the actual exit)
FEE, SLIP = sl.FEE, sl.SLIP
DAY = 86400000
OKX_FILE = os.path.join(OUT, "okx_instruments.json")
REGIME_AR = {"bull": "صاعد", "early": "بداية صعود", "turn": "محاولة انعطاف", "bear": "هابط", "weak": "ضعيف بلا اتجاه"}


# ---------------------------------------------------------------- indicators per coin (daily)
def indicators(rows):
    c = [r[4] for r in rows]
    n = len(c)
    e20, e50, e200 = sl.ema(c, 20), sl.ema(c, 50), sl.ema(c, 200)
    at = sl.atr(rows)
    vol = [None] * n
    for i in range(21, n):
        rs = [c[j] / c[j - 1] - 1 for j in range(i - 19, i + 1)]
        m = sum(rs) / 20
        vol[i] = math.sqrt(sum((x - m) ** 2 for x in rs) / 19)
    return dict(day=[r[0] // DAY for r in rows], o=[r[1] for r in rows], h=[r[2] for r in rows], l=[r[3] for r in rows],
                c=c, e20=e20, e50=e50, e200=e200, atr=at, vol=vol)


def snap(ind, k):
    """Everything the agent may know about one coin at the close of its bar k."""
    c = ind["c"]
    if k < 60:
        return None
    hi20 = max(ind["h"][k - 20:k])
    hi252 = max(ind["h"][max(0, k - 252):k + 1])
    under_recent = any(c[j] < ind["e20"][j] for j in range(max(0, k - 5), k))
    return dict(c=c[k], o_next=None, e20=ind["e20"][k], e50=ind["e50"][k], e200=ind["e200"][k], atr=ind["atr"][k],
                vol=ind["vol"][k], ret63=(c[k] / c[k - 63] - 1) if k >= 63 else None, near_high=c[k] / hi252 - 1,
                breakout=c[k] > hi20, resume=under_recent and c[k] > ind["e20"][k], low=ind["l"][k], high=ind["h"][k])


def regime(btc, breadth):
    above200, golden, above50 = btc["c"] > btc["e200"], btc["e50"] > btc["e200"], btc["c"] > btc["e50"]
    if above200 and golden:
        return "bull", (1.0 if breadth >= 0.5 else 0.5)
    if above200:
        return "early", (0.5 if breadth >= 0.5 else 0.25)
    if above50 and breadth >= 0.5:
        return "turn", 0.25
    if not golden:
        return "bear", 0.0
    return "weak", 0.0


def plan(snaps, positions, equity, values, force_budget=None):
    """snaps: {coin: snap or None} at today's close.  positions: {coin: {...}} with current values in `values`.
    Returns the decision: regime, budget, watchlist, exits [(coin, frac, why)], entries [(coin, weight, stop)]."""
    btc = snaps.get("BTC")
    live = {k: s for k, s in snaps.items() if s}
    breadth = sum(1 for s in live.values() if s["c"] > s["e50"]) / max(1, len(live))
    reg, budget = regime(btc, breadth) if btc else ("weak", 0.0)
    if force_budget is not None:
        budget = force_budget
    b63 = btc["ret63"] if btc and btc["ret63"] is not None else 0.0
    scored = []
    for k, s in live.items():
        if s["ret63"] is None or s["c"] <= s["e50"]:
            continue
        scored.append(((s["ret63"] - b63) + 0.5 * s["near_high"], k))
    scored.sort(reverse=True)
    watch = [k for _, k in scored[:WATCH_N]]
    score = {k: sc for sc, k in scored}
    exits = []
    # stops and partials are handled by the caller on prices; here: budget discipline
    held_val = sum(values.values())
    excess = held_val - budget * equity
    if excess > 0.05 * equity or budget == 0:
        for k in sorted(positions, key=lambda x: score.get(x, -9)):
            if held_val - budget * equity <= 0.01 * equity:
                break
            exits.append((k, 1.0, "budget"))
            held_val -= values.get(k, 0)
    entries = []
    room = budget * equity - held_val
    n_pos = len(positions) - sum(1 for e in exits if e[1] == 1.0)
    if budget > 0:
        for k in watch:
            s = live[k]
            if k in positions or n_pos >= MAX_POS or room < 0.02 * equity:
                continue
            if not (s["c"] > s["e20"] > s["e50"] and (s["breakout"] or s["resume"])) or not s["vol"] or not s["atr"]:
                continue
            w = min(budget / SLOT, VOL_UNIT / s["vol"], room / equity)
            if w < 0.02:
                continue
            entries.append((k, w, s["c"] - INIT_ATR * s["atr"]))
            room -= w * equity
            n_pos += 1
    return dict(regime=reg, budget=budget, breadth=breadth, watch=watch, score={k: score[k] for k in watch}, exits=exits, entries=entries,
                btc=dict(c=btc["c"], e50=btc["e50"], e200=btc["e200"]) if btc else None)


# ---------------------------------------------------------------- backtest
def backtest_run(data, start_day, end_day, force_budget=None, cost_mult=1.0):
    fee, slip = FEE * cost_mult, SLIP * cost_mult
    inds = {k: indicators(v) for k, v in data.items()}
    idx = {k: {d: i for i, d in enumerate(ind["day"])} for k, ind in inds.items()}
    cash, pos, trades, curve, reg_days = CAPITAL, {}, [], [], {}
    days = [d for d in inds["BTC"]["day"] if start_day <= d <= end_day]
    for t, d in enumerate(days[:-1]):
        nd = days[t + 1]
        # prices today
        px = {k: inds[k]["c"][idx[k][d]] for k in pos if d in idx[k]}
        values = {k: pos[k]["qty"] * px.get(k, pos[k]["entry"]) for k in pos}
        equity = cash + sum(values.values())
        curve.append((d, equity))
        snaps = {k: snap(inds[k], idx[k][d]) if d in idx[k] else None for k in inds}
        dec = plan(snaps, pos, equity, values, force_budget)
        reg_days.setdefault(dec["regime"], []).append(d)
        # execute at next open; then stops on next day's range; trailing on next close
        def sell(k, frac, price, why):
            nonlocal cash
            p = pos[k]
            q = p["qty"] * frac
            proceeds = q * price * (1 - slip) * (1 - fee)
            cash += proceeds
            basis = p["basis"] * frac
            trades.append(dict(coin=k, ret=proceeds / basis - 1, why=why, days=nd - p["day"], regime=p["regime"]))
            p["qty"] -= q
            p["basis"] -= basis
            if frac >= 0.999:
                pos.pop(k)
        for k, frac, why in dec["exits"]:
            if k in pos and nd in idx[k]:
                sell(k, frac, inds[k]["o"][idx[k][nd]], why)
        for k, w, stop in dec["entries"]:
            if nd not in idx[k]:
                continue
            o = inds[k]["o"][idx[k][nd]]
            invest = min(cash, w * equity)
            if invest < 20:
                continue
            cash -= invest
            s = snaps[k]
            stop = o - INIT_ATR * s["atr"]
            pos[k] = dict(qty=invest * (1 - fee) / (o * (1 + slip)), basis=invest, entry=o, stop=stop, r=o - stop, high=o,
                          half=False, day=nd, regime=dec["regime"])
        for k in list(pos):
            if nd not in idx[k]:
                continue
            j = idx[k][nd]
            p, ind = pos[k], inds[k]
            if ind["l"][j] <= p["stop"]:
                sell(k, 1.0, min(p["stop"], ind["o"][j]), "stop")
                continue
            if not p["half"] and ind["h"][j] >= p["entry"] + PARTIAL_R * p["r"]:
                sell(k, 0.5, p["entry"] + PARTIAL_R * p["r"], "partial")
                p["half"], p["stop"] = True, max(p["stop"], p["entry"])
            p["high"] = max(p["high"], ind["c"][j])
            p["stop"] = max(p["stop"], p["high"] - TRAIL_ATR * (ind["atr"][j] or 0))
    # close at the end
    d = days[-1]
    for k in list(pos):
        if d in idx[k]:
            p = pos.pop(k)
            pr = p["qty"] * inds[k]["c"][idx[k][d]] * (1 - slip) * (1 - fee)
            cash += pr
            trades.append(dict(coin=k, ret=pr / p["basis"] - 1, why="end", days=d - p["day"], regime=p["regime"]))
    curve.append((d, cash))
    return curve, trades, reg_days


def curve_stats(curve):
    if len(curve) < 2:
        return dict(ret=0.0, max_dd=0.0, cagr=0.0)
    pk, dd = 0.0, 0.0
    for _, v in curve:
        pk = max(pk, v)
        dd = max(dd, 1 - v / pk)
    yrs = (curve[-1][0] - curve[0][0]) / 365.25
    ret = curve[-1][1] / curve[0][1] - 1
    return dict(ret=ret, max_dd=dd, cagr=(1 + ret) ** (1 / yrs) - 1 if yrs > 0.3 and ret > -1 else None, days=curve[-1][0] - curve[0][0])


def btc_rulers(btc_rows, start_day, end_day):
    rows = [r for r in btc_rows if start_day <= r[0] // DAY <= end_day]
    hold = [(r[0] // DAY, CAPITAL * r[4] / rows[0][4]) for r in rows]
    units, spent, dca = 0.0, 0.0, []
    weekly = CAPITAL / max(1, len(rows) // 7)
    for i, r in enumerate(rows):
        if i % 7 == 0 and spent + weekly <= CAPITAL + 1e-6:
            units += weekly / r[4]
            spent += weekly
        dca.append((r[0] // DAY, CAPITAL - spent + units * r[4]))
    return curve_stats(hold), curve_stats(dca)


def backtest():
    data = {}
    for c in COINS:
        try:
            rows = sl.klines(c, "1d", 3000)
            if len(rows) >= 300:
                data[c] = rows
        except Exception as e:
            print(f"  ! {c}: {e}")
    btc_days = [r[0] // DAY for r in data["BTC"]]
    start, end = btc_days[260], btc_days[-1]
    split = btc_days[260 + int((len(btc_days) - 260) * 0.7)]
    out = dict(generated=datetime.now(timezone.utc).isoformat(), synthetic=sl.SYN, coins=len(data),
               period=[datetime.fromtimestamp(start * 86400, timezone.utc).strftime("%Y-%m-%d"),
                       datetime.fromtimestamp(end * 86400, timezone.utc).strftime("%Y-%m-%d")])
    for label, a, b in (("full", start, end), ("exam", split, end)):
        curve, trades, regd = backtest_run(data, a, b)
        curve_nr, trades_nr, _ = backtest_run(data, a, b, force_budget=1.0)
        curve_x3, _, _ = backtest_run(data, a, b, cost_mult=3.0)
        hold, dca = btc_rulers(data["BTC"], a, b)
        r = [t["ret"] for t in trades]
        out[label] = dict(agent=curve_stats(curve), agent_x3=curve_stats(curve_x3), no_regime=curve_stats(curve_nr),
                          btc_hold=hold, btc_dca=dca, trades=len(trades),
                          win=(sum(x > 0 for x in r) / len(r)) if r else None, avg_trade=(sum(r) / len(r)) if r else None,
                          by_exit={w: dict(n=sum(1 for t in trades if t["why"] == w),
                                           mean=sum(t["ret"] for t in trades if t["why"] == w) / max(1, sum(1 for t in trades if t["why"] == w)))
                                   for w in sorted({t["why"] for t in trades})},
                          regime_share={k: len(v) / max(1, sum(len(x) for x in regd.values())) for k, v in regd.items()},
                          exposure=None,
                          yearly=yearly(curve), btc_yearly=yearly([(r_[0] // DAY, r_[4]) for r_ in data["BTC"] if a <= r_[0] // DAY <= b]))
        A, H, D, N = out[label]["agent"], hold, dca, out[label]["no_regime"]
        print(f"{label}: agent {A['ret']*100:+.1f}% (maxDD {A['max_dd']*100:.0f}%) | no regime {N['ret']*100:+.1f}% (DD {N['max_dd']*100:.0f}%) "
              f"| BTC hold {H['ret']*100:+.1f}% (DD {H['max_dd']*100:.0f}%) | DCA {D['ret']*100:+.1f}% (DD {D['max_dd']*100:.0f}%) | trades {len(trades)}")
    json.dump(out, open(os.path.join(OUT, "backtest_G_null.json" if sl.SYN else "backtest_G.json"), "w"), indent=1)


def yearly(curve):
    by = {}
    for d, v in curve:
        y = datetime.fromtimestamp(d * 86400, timezone.utc).year
        by.setdefault(y, [v, v])
        by[y][1] = v
    return {str(y): v[1] / v[0] - 1 for y, v in by.items()}


# ---------------------------------------------------------------- real-exchange realism (treated as real money)
def okx_rules():
    """OKX spot instrument rules (lot size, minimum size, tick), refreshed once a day; public data, no key."""
    try:
        cache = json.load(open(OKX_FILE))
        if time.time() - cache.get("t", 0) < 86400:
            return cache["rules"]
    except Exception:
        cache = None
    if sl.SYN:
        return {c: dict(lot=1e-6, min=1e-6, tick=1e-6) for c in COINS}
    try:
        import urllib.request
        req = urllib.request.Request("https://www.okx.com/api/v5/public/instruments?instType=SPOT", headers={"User-Agent": "spot-lab"})
        rows = json.load(urllib.request.urlopen(req, timeout=20))["data"]
        rules = {r["baseCcy"]: dict(lot=float(r["lotSz"]), min=float(r["minSz"]), tick=float(r["tickSz"]))
                 for r in rows if r.get("quoteCcy") == "USDT" and r.get("state") == "live"}
        json.dump(dict(t=time.time(), rules=rules), open(OKX_FILE, "w"))
        return rules
    except Exception as e:
        print("  ! OKX instruments:", e)
        return cache["rules"] if cache else {}


def measured_slip(coin):
    """Per-side slippage: the model 0.05%, raised to 1.5x the live OKX measurement when that is higher."""
    try:
        avg = json.load(open(os.path.join(OUT, "okx_check.json"))).get("avg", {}).get(coin, {})
        m = (avg.get("okx_slip_bps") or 0) + (avg.get("okx_spread_bps") or 0) / 2
        return max(SLIP, 1.5 * m / 1e4)
    except Exception:
        return SLIP


def round_down(x, step):
    return math.floor(x / step + 1e-9) * step if step > 0 else x


# ---------------------------------------------------------------- live cycle
def shadow_order(coin, price, weight, stop, rule=None, capital=1000.0):
    usd = capital * weight
    tick = (rule or {}).get("tick", 0)
    lot = (rule or {}).get("lot", 0)
    lim = float(f"{round_down(price, tick):.10g}") if tick else float(f"{price:.6g}")
    qty = float(f"{round_down(usd / lim, lot):.10g}") if lot else float(f"{usd / lim:.6g}")
    r = price - stop
    return dict(pair=f"{coin}-USDT", side="شراء", type="محدد (Limit)", price=lim, usd=round(usd, 2),
                qty=qty, stop=round(stop, 6), stop_pct=round(1 - stop / price, 4),
                t1=float(f"{price + PARTIAL_R * r:.6g}"), t2=float(f"{price + TARGET2_R * r:.6g}"),
                note=f"لكل 1000$ من رأس مالك: {usd:,.0f}$. الوقف عند {stop:.6g} (−{(1 - stop / price) * 100:.1f}%).")


def report_text(dec, actions, pos_lines):
    reg, b = dec["regime"], dec["budget"]
    btc = dec["btc"] or {}
    lines = [f"حالة السوق: {REGIME_AR[reg]}. BTC عند {btc.get('c', 0):,.0f} مقابل متوسط 50 يوماً {btc.get('e50', 0):,.0f} ومتوسط 200 يوم {btc.get('e200', 0):,.0f}. "
             f"اتساع السوق: {dec['breadth']*100:.0f}% من العملات المباحة فوق متوسطها 50 يوماً.",
             f"ميزانية المخاطرة اليوم: {b*100:.0f}% من الحساب كحد أقصى في العملات."]
    if reg == "bear":
        lines.append("الوضع دفاعي: النقد مركز بحد ذاته. التاريخ يقول إن الشراء في السوق الهابط يخسر غالباً، حتى الارتدادات وحتى العملات القوية.")
    lines.append("قائمة القادة (الأقوى أمام BTC): " + ("، ".join(dec["watch"]) if dec["watch"] else "لا عملة فوق متوسطها 50 يوماً بعد."))
    lines.append("ما فعلته اليوم: " + ("؛ ".join(actions) if actions else "لا شيء، ولا تغيير مطلوب."))
    if pos_lines:
        lines.append("المراكز: " + "؛ ".join(pos_lines))
    if reg in ("bear", "weak"):
        lines.append("ما يغيّر رأيي: إغلاق BTC فوق متوسط 50 يوماً مع اتساع 50% أو أكثر (محاولة انعطاف، ميزانية 25%)، ثم فوق متوسط 200 يوم.")
    else:
        lines.append("ما يغيّر رأيي: عودة BTC تحت متوسط 200 يوم مع تقاطع 50 تحت 200 يعيد الحساب إلى النقد.")
    return "\n".join(lines)


def cycle():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    today = int(time.time() * 1000) // DAY
    st = json.load(open(STATE)) if os.path.exists(STATE) else dict(start=datetime.now(timezone.utc).isoformat(), cash=CAPITAL, pos={},
                                                                    trades=[], events=[], last_day=0, orders=[])
    # live prices for held coins (stop and partial checks every cycle)
    price = {}
    for c in list(st["pos"]) + ["BTC"]:
        try:
            price[c] = sl.last_price(c, sl.klines(c, "1h", 3, closed_only=False) if sl.SYN else None)
        except Exception as e:
            print(f"  ! price {c}: {e}")
    def sell(c, frac, px, why):
        p = st["pos"][c]
        q = p["qty"] * frac
        proceeds = q * px * (1 - p.get("slip", SLIP)) * (1 - FEE)
        st["cash"] += proceeds
        basis = p["basis"] * frac
        st["trades"].append(dict(time=stamp, coin=c, side="SELL", price=px, ret=proceeds / basis - 1, why=why, frac=frac,
                                 usd=round(proceeds, 2), cost_usd=round(basis, 2), pnl_usd=round(proceeds - basis, 2), qty=q,
                                 hours=round((time.time() * 1000 - p["entry_ms"]) / 3600000, 1)))
        p["qty"] -= q
        p["basis"] -= basis
        if frac >= 0.999:
            st["pos"].pop(c)
        print(f"  G SELL {c} {frac*100:.0f}% @ {px:g} ({why})")
    for c in list(st["pos"]):
        px = price.get(c)
        if px is None:
            continue
        p = st["pos"][c]
        if px <= p["stop"]:
            sell(c, 1.0, px, "stop")
            continue
        if not p["half"] and px >= p["entry"] + PARTIAL_R * p["r"]:
            sell(c, 0.5, px, "partial")
            p["half"], p["stop"] = True, max(p["stop"], p["entry"])
        if c in _REL and c in st["pos"]:
            sell(c, 1.0, px, "policy")

    # ---- risk limits of a real account: -15% from the peak closes everything and stops new entries for 14 days
    eq_now = st["cash"] + sum(p["qty"] * price.get(k, p["entry"]) for k, p in st["pos"].items())
    st["peak"] = max(st.get("peak", CAPITAL), eq_now)
    if not st.get("halt_until") and eq_now <= st["peak"] * 0.85:
        for c in list(st["pos"]):
            sell(c, 1.0, price.get(c, st["pos"][c]["entry"]), "risk_limit")
        st["halt_until"] = int(time.time() * 1000) + 14 * DAY
        st.setdefault("events", []).append(dict(time=stamp, text="G: حد المخاطرة: هبوط 15% من القمة، أُغلق كل شيء وتوقف الدخول 14 يوماً"))
    if st.get("halt_until") and time.time() * 1000 >= st["halt_until"]:
        st["halt_until"], st["peak"] = 0, eq_now

    # ---- daily decision on a new UTC day (yesterday's daily candle is closed)
    if st["last_day"] != today:
        data = {}
        for c in COINS:
            try:
                rows = sl.klines(c, "1d", 400)
                if len(rows) >= 120:
                    data[c] = rows
            except Exception as e:
                print(f"  ! {c}: {e}")
        if "BTC" in data:
            inds = {k: indicators(v) for k, v in data.items()}
            snaps = {k: snap(ind, len(ind["c"]) - 1) for k, ind in inds.items()}
            for k in st["pos"]:                                  # trailing stop on the daily close
                if k in inds:
                    ind = inds[k]
                    p = st["pos"][k]
                    p["high"] = max(p["high"], ind["c"][-1])
                    p["stop"] = max(p["stop"], p["high"] - TRAIL_ATR * (ind["atr"][-1] or 0))
            for k in snaps:
                if snaps[k] and k not in price:
                    price[k] = snaps[k]["c"]
            values = {k: st["pos"][k]["qty"] * price.get(k, st["pos"][k]["entry"]) for k in st["pos"]}
            equity = st["cash"] + sum(values.values())
            dec = plan(snaps, st["pos"], equity, values, force_budget=0.0 if st.get("halt_until") else None)
            actions = []
            for k, frac, why in dec["exits"]:
                if k in st["pos"]:
                    sell(k, frac, price.get(k, st["pos"][k]["entry"]), why)
                    actions.append(f"بيع {k} لأن الميزانية انخفضت إلى {dec['budget']*100:.0f}%")
            orders = []
            rules = okx_rules()
            for k, w, stop in dec["entries"]:
                try:
                    px = sl.last_price(k, data[k]) if sl.SYN else sl.last_price(k, None)
                except Exception:
                    px = price[k]
                price[k] = px
                rule = rules.get(k)
                if not rule:
                    actions.append(f"تخطي {k}: غير متاحة للتداول الفوري على OKX")
                    continue
                slip = measured_slip(k)
                fill = px * (1 + slip)
                qty = round_down(min(st["cash"], w * equity) * (1 - FEE) / fill, rule["lot"])
                if qty < rule["min"] or qty * fill < 10:
                    actions.append(f"تخطي {k}: الكمية أقل من الحد الأدنى في OKX")
                    continue
                invest = qty * fill / (1 - FEE)
                st["cash"] -= invest
                stop = float(f"{round_down(px - INIT_ATR * snaps[k]['atr'], rule['tick']):.10g}")
                st["pos"][k] = dict(qty=qty, basis=invest, entry=px, stop=stop, r=px - stop, slip=slip,
                                    high=px, half=False, entry_ms=int(time.time() * 1000), regime=dec["regime"])
                st["trades"].append(dict(time=stamp, coin=k, side="BUY", price=px, weight=w, regime=dec["regime"], usd=round(invest, 2), qty=qty,
                                         stop=stop, t1=px + PARTIAL_R * (px - stop), t2=px + TARGET2_R * (px - stop)))
                actions.append(f"شراء {k} بوزن {w*100:.1f}% (قائد، {('كسر قمة 20 يوماً' if snaps[k]['breakout'] else 'عودة فوق المتوسط 20 بعد تصحيح')})")
                orders.append(shadow_order(k, px, w, stop, rule))
                print(f"  G BUY {k} weight {w*100:.1f}% @ {px:g} stop {stop:g}")
            pos_lines = [f"{k} {(price.get(k, p['entry']) / p['entry'] - 1) * 100:+.1f}% (وقف {p['stop']:.6g})" for k, p in st["pos"].items()]
            text = report_text(dec, actions, pos_lines)
            st.update(last_day=today, regime=dec["regime"], budget=dec["budget"], breadth=dec["breadth"], watch=dec["watch"],
                      orders=orders or st.get("orders", [])[-6:], decision=text)
            if not st.get("events") or st["events"][-1].get("regime") != dec["regime"]:
                st.setdefault("events", []).append(dict(time=stamp, regime=dec["regime"], text=f"G: حالة السوق {REGIME_AR[dec['regime']]}، ميزانية المخاطرة {dec['budget']*100:.0f}%"))
            rep = json.load(open(REPORT)) if os.path.exists(REPORT) else {"days": []}
            rep["days"] = [d for d in rep["days"] if d.get("date") != datetime.now(timezone.utc).strftime("%Y-%m-%d")] + [
                dict(date=datetime.now(timezone.utc).strftime("%Y-%m-%d"), regime=dec["regime"], budget=dec["budget"], breadth=dec["breadth"],
                     watch=dec["watch"], score=dec["score"], btc=dec["btc"], text=text, orders=orders)]
            rep["days"] = rep["days"][-90:]
            json.dump(rep, open(REPORT, "w"), indent=1, ensure_ascii=False)
            print(text)
    v = st["cash"] + sum(p["qty"] * price.get(k, p["entry"]) for k, p in st["pos"].items())
    st["trades"], st["events"] = st["trades"][-400:], st.get("events", [])[-30:]
    st["equity"], st["prices"] = v, {k: price[k] for k in st["pos"] if k in price}
    for t in st["trades"]:                                          # older buys: record the dollars from the open position
        if t["side"] == "BUY" and "usd" not in t and t["coin"] in st["pos"]:
            t["usd"], t["qty"] = round(st["pos"][t["coin"]]["basis"], 2), st["pos"][t["coin"]]["qty"]
    st["holdings"] = {k: dict(qty=p["qty"], basis=round(p["basis"], 2), t1=p["entry"] + PARTIAL_R * p["r"], t2=p["entry"] + TARGET2_R * p["r"],
                              half_taken=p["half"], pnl_usd=round(p["qty"] * price.get(k, p["entry"]) * (1 - FEE) - p["basis"], 2),
                              value=p["qty"] * price.get(k, p["entry"]), entry=p["entry"], sl=p["stop"], tp=p["entry"] + PARTIAL_R * p["r"],
                              pnl=price.get(k, p["entry"]) / p["entry"] - 1, signal=p.get("regime", ""), protected=p["stop"] >= p["entry"]) for k, p in st["pos"].items()}
    sells = [t for t in st["trades"] if t["side"] == "SELL"]
    st["stats"] = dict(trades=len(sells), gate_trades=20, gate=f"مغلقة {len(sells)} · الحكم بعد 3 أشهر مقابل الاحتفاظ بـBTC والشراء الدوري")
    st["rules"] = dict(coins=COINS, agent=True, real_money_mode=True, account_dd_limit=0.15, watch_n=WATCH_N, max_pos=MAX_POS, vol_unit=VOL_UNIT, init_atr=INIT_ATR, trail_atr=TRAIL_ATR, partial_r=PARTIAL_R)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "paper_equity_G.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "equity", "btc"])
        w.writerow([stamp, round(v, 2), round(price.get("BTC", 0), 2)])
    json.dump(st, open(STATE, "w"), indent=1, ensure_ascii=False)
    print(f"[{stamp}] G ${v:,.2f} ({(v/CAPITAL-1)*100:+.2f}%) regime {st.get('regime')} budget {st.get('budget')} open {list(st['pos']) or 'none'}")


if __name__ == "__main__":
    backtest() if "backtest" in sys.argv else cycle()
