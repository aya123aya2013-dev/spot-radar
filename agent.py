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
import risk

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


# ---------------------------------------------------------------- backtest (champion = rules alone, challenger = rules + risk engine)
def backtest_run(data, start_day, end_day, force_budget=None, cost_mult=1.0, risk_cfg=None, edge_table=None, inds=None, idx=None):
    """Day by day: decide at the close of day d, execute at the open of d+1, stops on d+1's range, trailing on its close.
    risk_cfg None = champion (the strategy's own sizing, no risk engine).  With risk_cfg: every risk gate that can be
    simulated on daily bars (per-trade risk, stop cap, position cap, correlation cap, daily/weekly loss, drawdown halt,
    loss-streak and asset cooldowns, min-edge gate).  Spread and data-freshness gates cannot be simulated and are live-only."""
    fee, slip = FEE * cost_mult, SLIP * cost_mult
    if inds is None:
        inds = {k: indicators(v) for k, v in data.items()}
        idx = {k: {d: i for i, d in enumerate(ind["day"])} for k, ind in inds.items()}
    cash, pos, trades, curve, reg_days = CAPITAL, {}, [], [], {}
    acc = dict(fees=0.0, slip=0.0, traded=0.0, rejected={})
    peak, halt_until, streak_until, cooldown = CAPITAL, -1, -1, {}
    prev_eq, week_start, week_id, closed = CAPITAL, CAPITAL, None, []
    days = [d for d in inds["BTC"]["day"] if start_day <= d <= end_day]
    for t, d in enumerate(days[:-1]):
        nd = days[t + 1]
        px = {k: inds[k]["c"][idx[k][d]] for k in pos if d in idx[k]}
        values = {k: pos[k]["qty"] * px.get(k, pos[k]["entry"]) for k in pos}
        equity = cash + sum(values.values())
        curve.append((d, equity, sum(values.values())))
        snaps = {k: snap(inds[k], idx[k][d]) if d in idx[k] else None for k in inds}

        def sell(k, frac, price, why):
            nonlocal cash
            p = pos[k]
            q = p["qty"] * frac
            gross = q * price
            proceeds = gross * (1 - slip) * (1 - fee)
            acc["fees"] += gross * (1 - slip) * fee
            acc["slip"] += gross * slip
            acc["traded"] += gross
            cash += proceeds
            basis = p["basis"] * frac
            p["pnl"] = p.get("pnl", 0.0) + proceeds - basis
            p["qty"] -= q
            p["basis"] -= basis
            if frac >= 0.999:
                pos.pop(k)
                tot = p["pnl"] / p["basis0"]
                trades.append(dict(coin=k, ret=tot, pnl=p["pnl"], why=why, days=nd - p["day"], regime=p["regime"]))
                closed.append(tot)
                if risk_cfg and why == "stop":
                    cooldown[k] = nd + risk_cfg["asset_cooldown_days"]
                if risk_cfg and risk.losing_streak(closed) >= risk_cfg["loss_streak"]:
                    nonlocal streak_until
                    streak_until = nd + risk_cfg["streak_cooldown_days"]
                    closed.append(1e-9)                    # the streak has been acted on

        allow, reasons = True, []
        if risk_cfg:
            peak = max(peak, equity)
            if halt_until < d and risk.drawdown_breach(risk_cfg, equity, peak):
                halt_until = d + risk_cfg["halt_days"]
                for k in list(pos):
                    if nd in idx[k]:
                        sell(k, 1.0, inds[k]["o"][idx[k][nd]], "risk_limit")
            if halt_until >= 0 and d > halt_until:
                halt_until, peak = -1, equity
            wk = (d + 3) // 7
            if wk != week_id:
                week_id, week_start = wk, equity
            acct = dict(equity=equity, peak=peak, day_start_eq=prev_eq, week_start_eq=week_start,
                        halt_until=(halt_until + 1) * DAY if halt_until >= d else 0, streak_until=(streak_until + 1) * DAY if streak_until >= d else 0)
            allow, reasons = risk.global_gates(risk_cfg, acct, (d + 1) * DAY - 1, 0, 1.0)
        prev_eq = equity
        dec = plan(snaps, pos, equity, values, 0.0 if (risk_cfg and halt_until >= d) else force_budget)
        reg_days.setdefault(dec["regime"], []).append(d)
        for k, frac, why in dec["exits"]:
            if k in pos and nd in idx[k]:
                sell(k, frac, inds[k]["o"][idx[k][nd]], why)
        held_vals = {k: pos[k]["qty"] * px.get(k, pos[k]["entry"]) for k in pos}
        n_pos = len(pos)
        for k, w, stop in dec["entries"]:
            if nd not in idx[k]:
                continue
            s = snaps[k]
            if risk_cfg:
                if not allow:
                    for r_ in reasons:
                        acc["rejected"][r_] = acc["rejected"].get(r_, 0) + 1
                    continue
                ok_edge, _, _ = risk.edge_eligible(risk_cfg, edge_table, dec["regime"])
                if not ok_edge:
                    acc["rejected"]["insufficient_edge"] = acc["rejected"].get("insufficient_edge", 0) + 1
                    continue
                rets = {}
                for kk in [k] + list(pos):
                    j = idx[kk].get(d)
                    if j is not None and j >= 61:
                        rets[kk] = risk.returns(inds[kk]["c"][:j + 1])
                ctx = dict(equity=equity, cash=cash, budget=dec["budget"], held_values=held_vals, n_positions=n_pos,
                           rets=rets, cooldowns={c: (u + 1) * DAY for c, u in cooldown.items()}, now_ms=(d + 1) * DAY - 1)
                usd, why_not = risk.size_entry(risk_cfg, dict(coin=k, price=s["c"], stop=stop, weight=w), ctx)
                if usd <= 0:
                    for r_ in why_not:
                        acc["rejected"][r_] = acc["rejected"].get(r_, 0) + 1
                    continue
                invest = min(cash, usd)
            else:
                invest = min(cash, w * equity)
            if invest < 20:
                continue
            o = inds[k]["o"][idx[k][nd]]
            cash -= invest
            acc["fees"] += invest * fee
            acc["slip"] += invest * slip
            acc["traded"] += invest
            st_ = o - INIT_ATR * s["atr"]
            pos[k] = dict(qty=invest * (1 - fee) / (o * (1 + slip)), basis=invest, basis0=invest, entry=o, stop=st_, r=o - st_, high=o,
                          half=False, day=nd, regime=dec["regime"])
            held_vals[k] = invest
            n_pos += 1
        for k in list(pos):
            if nd not in idx[k]:
                continue
            j = idx[k][nd]
            p, ind = pos[k], inds[k]
            if ind["l"][j] <= p["stop"]:
                sell(k, 1.0, min(p["stop"], ind["o"][j]), "stop")
                continue
            if not p["half"] and ind["h"][j] >= p["entry"] + PARTIAL_R * p["r"]:
                sell(k, 0.5, max(p["entry"] + PARTIAL_R * p["r"], ind["o"][j]), "partial")
                p["half"], p["stop"] = True, max(p["stop"], p["entry"])
            p["high"] = max(p["high"], ind["c"][j])
            p["stop"] = max(p["stop"], p["high"] - TRAIL_ATR * (ind["atr"][j] or 0))
    d = days[-1]
    for k in list(pos):
        if d in idx[k]:
            p = pos.pop(k)
            pr = p["qty"] * inds[k]["c"][idx[k][d]] * (1 - slip) * (1 - fee)
            cash += pr
            p["pnl"] = p.get("pnl", 0.0) + pr - p["basis"]
            trades.append(dict(coin=k, ret=p["pnl"] / p["basis0"], pnl=p["pnl"], why="end", days=d - p["day"], regime=p["regime"]))
    curve.append((d, cash, 0.0))
    return curve, trades, reg_days, acc


def curve_stats(curve):
    """Return, drawdown, CAGR, Sharpe, Sortino, Calmar (daily samples, 365-day year), average exposure."""
    if len(curve) < 2:
        return dict(ret=0.0, max_dd=0.0, cagr=0.0)
    pk, dd = 0.0, 0.0
    for p in curve:
        pk = max(pk, p[1])
        dd = max(dd, 1 - p[1] / pk)
    yrs = (curve[-1][0] - curve[0][0]) / 365.25
    ret = curve[-1][1] / curve[0][1] - 1
    cagr = (1 + ret) ** (1 / yrs) - 1 if yrs > 0.3 and ret > -1 else None
    r = [curve[i][1] / curve[i - 1][1] - 1 for i in range(1, len(curve)) if curve[i - 1][1] > 0]
    m = sum(r) / len(r) if r else 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in r) / max(1, len(r) - 1)) if len(r) > 1 else 0.0
    dn = math.sqrt(sum(min(0.0, x) ** 2 for x in r) / max(1, len(r))) if r else 0.0
    exp = [p[2] / p[1] for p in curve if len(p) > 2 and p[1] > 0]
    return dict(ret=ret, max_dd=dd, cagr=cagr, days=curve[-1][0] - curve[0][0],
                sharpe=(m / sd * math.sqrt(365)) if sd > 0 else None,
                sortino=(m / dn * math.sqrt(365)) if dn > 0 else None,
                calmar=(cagr / dd) if cagr is not None and dd > 0 else None,
                exposure=(sum(exp) / len(exp)) if exp else None)


def trade_stats(trades, acc=None, avg_equity=CAPITAL, years=1.0):
    r = [t["ret"] for t in trades]
    wins, losses = [x for x in r if x > 0], [x for x in r if x <= 0]
    pnl_w = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    pnl_l = -sum(t["pnl"] for t in trades if t.get("pnl", 0) <= 0)
    out = dict(trades=len(r), win=(len(wins) / len(r)) if r else None, avg_win=(sum(wins) / len(wins)) if wins else None,
               avg_loss=(sum(losses) / len(losses)) if losses else None, expectancy=(sum(r) / len(r)) if r else None,
               profit_factor=(pnl_w / pnl_l) if pnl_l > 0 else None,
               by_exit={w: dict(n=sum(1 for t in trades if t["why"] == w),
                                mean=sum(t["ret"] for t in trades if t["why"] == w) / max(1, sum(1 for t in trades if t["why"] == w)))
                        for w in sorted({t["why"] for t in trades})})
    if acc:
        out.update(fees=acc["fees"], slippage=acc["slip"], turnover_per_year=acc["traded"] / max(1.0, avg_equity) / max(0.1, years),
                   rejected=acc["rejected"])
    return out


def btc_rulers(btc_rows, start_day, end_day):
    rows = [r for r in btc_rows if start_day <= r[0] // DAY <= end_day]
    hold = [(r[0] // DAY, CAPITAL * r[4] / rows[0][4], CAPITAL * r[4] / rows[0][4]) for r in rows]
    units, spent, dca = 0.0, 0.0, []
    weekly = CAPITAL / max(1, len(rows) // 7)
    for i, r in enumerate(rows):
        if i % 7 == 0 and spent + weekly <= CAPITAL + 1e-6:
            units += weekly / r[4]
            spent += weekly
        dca.append((r[0] // DAY, CAPITAL - spent + units * r[4], units * r[4]))
    return curve_stats(hold), curve_stats(dca)


def yearly(curve):
    by = {}
    for p in curve:
        y = datetime.fromtimestamp(p[0] * 86400, timezone.utc).year
        by.setdefault(y, [p[1], p[1]])
        by[y][1] = p[1]
    return {str(y): v[1] / v[0] - 1 for y, v in by.items()}


PROMOTION_RULE = ("challenger (rules + risk engine) replaces champion (rules alone) only if ALL hold: "
                  "(1) full-period max drawdown at least 5 points lower; "
                  "(2) full-period Calmar or Sharpe at least as high; "
                  "(3) positive return in at least 3 of the 5 walk-forward folds; "
                  "(4) full-period return still positive with costs x3. Written before the first run; 2 candidates tested.")


def evaluate(data):
    inds = {k: indicators(v) for k, v in data.items()}
    idx = {k: {d: i for i, d in enumerate(ind["day"])} for k, ind in inds.items()}
    btc_days = inds["BTC"]["day"]
    start, end = btc_days[260], btc_days[-1]
    span = [d for d in btc_days if start <= d <= end]
    folds = [(span[len(span) * i // 5], span[min(len(span) - 1, len(span) * (i + 1) // 5)]) for i in range(5)]
    cfg = risk.load_control()
    run = lambda a, b, **kw: backtest_run(data, a, b, inds=inds, idx=idx, **kw)

    def edge_before(day):
        """Edge table from champion trades CLOSED before `day` (expanding window, no look-ahead)."""
        if day - start < 120:
            return {}
        _, tr, _, _ = run(start, day)
        return risk.edge_table_from(tr)

    out = dict(generated=datetime.now(timezone.utc).isoformat(), synthetic=sl.SYN, coins=len(data),
               period=[datetime.fromtimestamp(start * 86400, timezone.utc).strftime("%Y-%m-%d"),
                       datetime.fromtimestamp(end * 86400, timezone.utc).strftime("%Y-%m-%d")],
               candidates_tested=2, promotion_rule=PROMOTION_RULE, risk_config=cfg,
               caveats=["survivorship: the universe is today's listed permissible coins",
                        "the 2024-2026 'exam' was lived through before the rules were written: not a truly unseen holdout",
                        "spread and data-freshness gates are live-only and not simulated",
                        "the only truly unseen test is the live paper record from 2026-11-02"])
    # full period: champion, challenger (edge gate uses only trades closed before each fold start, recomputed per fold)
    res = {}
    for name, kw in (("champion", {}), ("challenger", dict(risk_cfg=cfg))):
        fold_rows, curve_all, trades_all = [], [], []
        for a, b in folds:
            extra = dict(kw)
            if kw:
                extra["edge_table"] = edge_before(a)
            c, tr, rd, acc = run(a, b, **extra)
            cs = curve_stats(c)
            fold_rows.append(dict(start=datetime.fromtimestamp(a * 86400, timezone.utc).strftime("%Y-%m"),
                                  end=datetime.fromtimestamp(b * 86400, timezone.utc).strftime("%Y-%m"),
                                  ret=cs["ret"], max_dd=cs["max_dd"], trades=len(tr)))
        kwf = dict(kw)
        if kw:
            kwf["edge_table"] = edge_before(folds[0][0])
        c, tr, rd, acc = run(start, end, **kwf)
        c3, _, _, _ = run(start, end, cost_mult=3.0, **kwf)
        cs = curve_stats(c)
        yrs = max(0.1, cs["days"] / 365.25)
        res[name] = dict(full=cs, full_x3=curve_stats(c3), folds=fold_rows,
                         trades=trade_stats(tr, acc, sum(p[1] for p in c) / len(c), yrs),
                         regime_share={k: len(v) / max(1, sum(len(x) for x in rd.values())) for k, v in rd.items()},
                         yearly=yearly(c))
        print(f"{name}: {cs['ret']*100:+.1f}% DD {cs['max_dd']*100:.0f}% CAGR {((cs['cagr'] or 0)*100):+.1f}% Sharpe {cs['sharpe'] or 0:.2f} "
              f"Calmar {cs['calmar'] or 0:.2f} | x3 {res[name]['full_x3']['ret']*100:+.1f}% | folds "
              + " ".join(f"{f['ret']*100:+.0f}%" for f in fold_rows))
    hold, dca = btc_rulers(data["BTC"], start, end)
    nr, _, _, _ = run(start, end, force_budget=1.0)
    out.update(res, btc_hold=hold, btc_dca=dca, no_regime=curve_stats(nr), btc_yearly=yearly([(r[0] // DAY, r[4]) for r in data["BTC"] if start <= r[0] // DAY <= end]))
    ch, cc = res["champion"], res["challenger"]
    rules = [cc["full"]["max_dd"] <= ch["full"]["max_dd"] - 0.05,
             (cc["full"]["calmar"] or -9) >= (ch["full"]["calmar"] or -9) or (cc["full"]["sharpe"] or -9) >= (ch["full"]["sharpe"] or -9),
             sum(1 for f in cc["folds"] if f["ret"] > 0) >= 3,
             cc["full_x3"]["ret"] > 0]
    out["promotion"] = dict(challenger=all(rules), checks=rules)
    # live edge table: every champion-style trade the challenger made, all in the past by construction
    _, tr_all, _, _ = run(start, end, risk_cfg=cfg, edge_table={})
    out["edge_by_regime"] = risk.edge_table_from(tr_all)
    # compatibility with the dashboard's earlier fields
    out["full"] = dict(agent=cc["full"] if out["promotion"]["challenger"] else ch["full"], agent_x3=(cc if out["promotion"]["challenger"] else ch)["full_x3"],
                       no_regime=out["no_regime"], btc_hold=hold, btc_dca=dca,
                       trades=(cc if out["promotion"]["challenger"] else ch)["trades"]["trades"], win=(cc if out["promotion"]["challenger"] else ch)["trades"]["win"])
    last = folds[-1]
    e_c, _, _, _ = run(last[0], last[1], risk_cfg=cfg, edge_table=edge_before(last[0]))
    e_h, _, _, _ = run(last[0], last[1])
    h2, d2 = btc_rulers(data["BTC"], last[0], last[1])
    out["exam"] = dict(agent=curve_stats(e_c if out["promotion"]["challenger"] else e_h), btc_hold=h2, btc_dca=d2)
    print("promotion:", out["promotion"], "| BTC hold", f"{hold['ret']*100:+.1f}% DD {hold['max_dd']*100:.0f}%")
    return out


def backtest():
    data = {}
    for c in COINS:
        try:
            rows = sl.klines(c, "1d", 3000)
            if len(rows) >= 300:
                data[c] = rows
        except Exception as e:
            print(f"  ! {c}: {e}")
    out = evaluate(data)
    json.dump(out, open(os.path.join(OUT, "backtest_G_null.json" if sl.SYN else "backtest_G.json"), "w"), indent=1)


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


def okx_book(coin):
    """Measured OKX spread and slippage in basis points for a 1000$ order, if this coin is monitored."""
    try:
        a = json.load(open(os.path.join(OUT, "okx_check.json"))).get("avg", {}).get(coin)
        if a:
            return a.get("okx_spread_bps"), a.get("okx_slip_bps")
    except Exception:
        pass
    return None, None


def load_edge():
    try:
        return json.load(open(os.path.join(OUT, "backtest_G.json"))).get("edge_by_regime", {})
    except Exception:
        return {}


def cycle():
    import journal, accounting
    cfg = risk.load_control()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    now_ms = int(time.time() * 1000)
    today = now_ms // DAY
    st = json.load(open(STATE)) if os.path.exists(STATE) else dict(start=datetime.now(timezone.utc).isoformat(), cash=CAPITAL, pos={},
                                                                    trades=[], events=[], last_day=0, orders=[])
    st.setdefault("cooldown", {})
    jrec = []
    # live prices for held coins (stop and partial checks every cycle)
    price, price_ok = {}, True
    for c in list(st["pos"]) + ["BTC"]:
        try:
            price[c] = sl.last_price(c, sl.klines(c, "1h", 3, closed_only=False) if sl.SYN else None)
        except Exception as e:
            price_ok = False
            print(f"  ! price {c}: {e}")

    def sell(c, frac, px, why):
        p = st["pos"][c]
        q = risk.can_sell(p["qty"], p["qty"] * frac)
        s_ = p.get("slip", SLIP)
        gross = q * px
        proceeds = gross * (1 - s_) * (1 - FEE)
        st["cash"] += proceeds
        part = q / p["qty"] if p["qty"] > 0 else 1.0
        basis = p["basis"] * part
        st["trades"].append(dict(time=stamp, coin=c, side="SELL", price=px, ret=proceeds / basis - 1, why=why, frac=frac,
                                 usd=round(proceeds, 2), cost_usd=round(basis, 2), pnl_usd=round(proceeds - basis, 2), qty=q,
                                 fee_usd=round(gross * (1 - s_) * FEE, 4), slip_usd=round(gross * s_, 4), pos_id=p.get("pos_id"),
                                 hours=round((now_ms - p["entry_ms"]) / 3600000, 1)))
        jrec.append(dict(t=stamp, type="exit", coin=c, why=why, frac=frac, price=px, pnl_usd=round(proceeds - basis, 2), pos_id=p.get("pos_id")))
        p["qty"] -= q
        p["basis"] -= basis
        if frac >= 0.999 or p["qty"] <= 1e-12:
            st["pos"].pop(c)
            if why == "stop":
                st["cooldown"][c] = now_ms + cfg["asset_cooldown_days"] * DAY
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

    # ---- positions opened before the risk engine existed: if the loss at the stop is above 1.5x the per-trade cap,
    #      sell part so that it is back at the cap (REDUCE EXPOSURE; never adds, never averages down)
    eq_tmp = st["cash"] + sum(p["qty"] * price.get(k, p["entry"]) for k, p in st["pos"].items())
    for c in list(st["pos"]):
        p, px = st["pos"][c], price.get(c)
        if px is None or px <= p["stop"]:
            continue
        at_risk = p["qty"] * (px - p["stop"])
        cap = cfg["risk_per_trade"] * eq_tmp
        if at_risk > 1.5 * cap:
            sell(c, 1 - cap / at_risk, px, "risk_trim")
            st["events"].append(dict(time=stamp, text=f"G: تقليص {c}: الخسارة حتى الوقف كانت ${at_risk:,.0f} فوق حد المخاطرة ${cap:,.0f}، بيع جزء لإعادتها إلى الحد"))

    # ---- account-level limits (risk engine): drawdown halt, daily and weekly loss, losing streak
    eq_now = st["cash"] + sum(p["qty"] * price.get(k, p["entry"]) for k, p in st["pos"].items())
    st["peak"] = max(st.get("peak", CAPITAL), eq_now)
    if not st.get("halt_until") and risk.drawdown_breach(cfg, eq_now, st["peak"]):
        for c in list(st["pos"]):
            sell(c, 1.0, price.get(c, st["pos"][c]["entry"]), "risk_limit")
        st["halt_until"] = now_ms + cfg["halt_days"] * DAY
        st["events"].append(dict(time=stamp, text=f"G: قاطع الهبوط: {cfg['max_drawdown']*100:.0f}% من القمة، أُغلق كل شيء وتوقف الدخول {cfg['halt_days']} يوماً"))
    if st.get("halt_until") and now_ms >= st["halt_until"]:
        st["halt_until"], st["peak"] = 0, eq_now
    day_id, week_id = today, (today + 3) // 7
    if st.get("day_id") != day_id:
        st["day_id"], st["day_start_eq"] = day_id, eq_now
    if st.get("week_id") != week_id:
        st["week_id"], st["week_start_eq"] = week_id, eq_now
    closed = [t["ret"] for t in st["trades"] if t["side"] == "SELL" and t.get("frac", 1.0) >= 0.999]
    if risk.losing_streak(closed) >= cfg["loss_streak"] and st.get("streak_seen") != len(closed):
        st["streak_until"], st["streak_seen"] = now_ms + cfg["streak_cooldown_days"] * DAY, len(closed)
        st["events"].append(dict(time=stamp, text=f"G: {cfg['loss_streak']} خسائر متتالية: لا دخول جديد لمدة {cfg['streak_cooldown_days']} أيام"))

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
            age_h = 0.0 if sl.SYN else (now_ms - (data["BTC"][-1][0] + DAY)) / 3600000   # synthetic candles are not dated like real ones
            coverage = len(data) / len(COINS)
            inds = {k: indicators(v) for k, v in data.items()}
            snaps = {k: snap(ind, len(ind["c"]) - 1) for k, ind in inds.items()}
            for k in st["pos"]:                                  # trailing stop on the daily close
                if k in inds:
                    ind, p = inds[k], st["pos"][k]
                    p["high"] = max(p["high"], ind["c"][-1])
                    p["stop"] = max(p["stop"], p["high"] - TRAIL_ATR * (ind["atr"][-1] or 0))
            for k in snaps:
                if snaps[k] and k not in price:
                    price[k] = snaps[k]["c"]
            values = {k: st["pos"][k]["qty"] * price.get(k, st["pos"][k]["entry"]) for k in st["pos"]}
            equity = st["cash"] + sum(values.values())
            acct = dict(equity=equity, peak=st["peak"], day_start_eq=st.get("day_start_eq"), week_start_eq=st.get("week_start_eq"),
                        halt_until=st.get("halt_until", 0), streak_until=st.get("streak_until", 0))
            allow, gate_reasons = risk.global_gates(cfg, acct, now_ms, age_h, coverage)
            dec = plan(snaps, st["pos"], equity, values, force_budget=0.0 if st.get("halt_until") else None)
            edge_ok, edge_mean, edge_n = risk.edge_eligible(cfg, load_edge(), dec["regime"])
            actions, orders, rejected = [], [], []
            for k, frac, why in dec["exits"]:
                if k in st["pos"]:
                    sell(k, frac, price.get(k, st["pos"][k]["entry"]), why)
                    actions.append(f"بيع {k} لأن الميزانية انخفضت إلى {dec['budget']*100:.0f}%")
            rules = okx_rules()
            proposed = {k for k, _, _ in dec["entries"]}
            for k in dec["watch"]:
                if k not in proposed and k not in st["pos"]:
                    rejected.append(dict(coin=k, reasons=["no_entry_signal"]))
            held_vals = {k: st["pos"][k]["qty"] * price.get(k, st["pos"][k]["entry"]) for k in st["pos"]}
            rets = {k: risk.returns(inds[k]["c"]) for k in set(proposed) | set(st["pos"]) if k in inds and len(inds[k]["c"]) > 61}
            for k, w, _ in dec["entries"]:
                feats = dict(c=snaps[k]["c"], e20=snaps[k]["e20"], e50=snaps[k]["e50"], atr=snaps[k]["atr"], vol=snaps[k]["vol"],
                             ret63=snaps[k]["ret63"], near_high=snaps[k]["near_high"], breakout=snaps[k]["breakout"], score=dec["score"].get(k))
                if not allow:
                    rejected.append(dict(coin=k, reasons=gate_reasons, weight=w, features=feats))
                    continue
                if not edge_ok:
                    rejected.append(dict(coin=k, reasons=["insufficient_edge"], weight=w, features=feats, edge=edge_mean, edge_n=edge_n))
                    continue
                rule = rules.get(k)
                if not rule:
                    rejected.append(dict(coin=k, reasons=["not_on_okx"], weight=w, features=feats))
                    actions.append(f"تخطي {k}: غير متاحة للتداول الفوري على OKX")
                    continue
                try:
                    px = sl.last_price(k, data[k]) if sl.SYN else sl.last_price(k, None)
                except Exception:
                    px = price[k]
                price[k] = px
                stop = float(f"{round_down(px - INIT_ATR * snaps[k]['atr'], rule['tick']):.10g}")
                spread_bps, slip_bps = okx_book(k)
                slip = measured_slip(k)
                cand = dict(coin=k, price=px * (1 + slip), stop=stop, weight=w, spread_bps=spread_bps, slip_bps=slip_bps,
                            min_qty=rule["min"], lot=rule["lot"])
                ctx = dict(equity=equity, cash=st["cash"], budget=dec["budget"], held_values=held_vals, n_positions=len(st["pos"]),
                           rets=rets, cooldowns=st["cooldown"], now_ms=now_ms)
                usd, why_not = risk.size_entry(cfg, cand, ctx)
                if usd <= 0:
                    rejected.append(dict(coin=k, reasons=why_not, weight=w, features=feats, stop_pct=1 - stop / px))
                    actions.append(f"رفض {k}: " + "، ".join(risk.REASON_AR.get(r, r) for r in why_not))
                    continue
                fill = px * (1 + slip)
                qty = round_down(usd * (1 - FEE) / fill, rule["lot"])
                invest = qty * fill / (1 - FEE)
                if invest > st["cash"]:
                    qty = round_down(st["cash"] * (1 - FEE) / fill, rule["lot"])
                    invest = qty * fill / (1 - FEE)
                if qty < rule["min"] or invest < cfg["min_order_usd"]:
                    rejected.append(dict(coin=k, reasons=["below_exchange_minimum"], weight=w, features=feats))
                    continue
                st["cash"] -= invest
                pos_id = f"G-{k}-{stamp.replace(' ', 'T')}"
                st["pos"][k] = dict(qty=qty, basis=invest, entry=px, stop=stop, r=px - stop, slip=slip, pos_id=pos_id,
                                    high=px, half=False, entry_ms=now_ms, regime=dec["regime"])
                held_vals[k] = invest
                risk_usd = invest * (1 - stop / px)
                st["trades"].append(dict(time=stamp, coin=k, side="BUY", price=px, weight=invest / equity, asked_weight=w, regime=dec["regime"],
                                         usd=round(invest, 2), qty=qty, stop=stop, t1=px + PARTIAL_R * (px - stop), t2=px + TARGET2_R * (px - stop),
                                         fee_usd=round(invest * FEE, 4), slip_usd=round(qty * px * slip, 4), risk_usd=round(risk_usd, 2), pos_id=pos_id))
                jrec.append(dict(t=stamp, type="entry", coin=k, regime=dec["regime"], budget=dec["budget"], price=px, stop=stop, usd=round(invest, 2),
                                 asked_weight=w, risk_usd=round(risk_usd, 2), features=feats, edge=edge_mean, edge_n=edge_n, pos_id=pos_id,
                                 approved_by="risk_engine", mode=cfg["mode"]))
                actions.append(f"شراء {k} بـ ${invest:,.0f} ({invest/equity*100:.1f}% من الحساب، مخاطرة ${risk_usd:,.0f} حتى الوقف، "
                               f"{('كسر قمة 20 يوماً' if snaps[k]['breakout'] else 'عودة فوق المتوسط 20 بعد تصحيح')})")
                orders.append(shadow_order(k, px, invest / equity, stop, rule))
                print(f"  G BUY {k} ${invest:,.2f} @ {px:g} stop {stop:g}")
            if not dec["entries"] and dec["budget"] == 0:
                actions.append("لا دخول: الميزانية صفر في هذه الحالة، والنقد هو القرار")
            for r_ in rejected:
                jrec.append(dict(t=stamp, type="rejected", regime=dec["regime"], budget=dec["budget"], **r_))
            jrec.append(dict(t=stamp, type="day", regime=dec["regime"], budget=dec["budget"], breadth=dec["breadth"], watch=dec["watch"],
                             allow_entries=allow, gate_reasons=gate_reasons, data_age_h=round(age_h, 1), coverage=round(coverage, 3),
                             edge=edge_mean, edge_n=edge_n, equity=round(equity, 2), mode=cfg["mode"], emergency_stop=cfg["emergency_stop"]))
            if not allow:
                actions.append("الدخول الجديد موقوف: " + "، ".join(risk.REASON_AR.get(r, r) for r in gate_reasons))
            pos_lines = [f"{k} {(price.get(k, p['entry']) / p['entry'] - 1) * 100:+.1f}% (وقف {p['stop']:.6g})" for k, p in st["pos"].items()]
            text = report_text(dec, actions, pos_lines)
            st.update(last_day=today, regime=dec["regime"], budget=dec["budget"], breadth=dec["breadth"], watch=dec["watch"],
                      orders=orders or st.get("orders", [])[-6:], decision=text,
                      rejected=[dict(coin=r_["coin"], reasons=r_["reasons"], text="، ".join(risk.REASON_AR.get(x, x) for x in r_["reasons"])) for r_ in rejected],
                      health=dict(data_age_h=round(age_h, 1), coverage=round(coverage, 3), allow_entries=allow, gate_reasons=gate_reasons,
                                  gate_text=[risk.REASON_AR.get(x, x) for x in gate_reasons], checked=stamp))
            if not st.get("events") or st["events"][-1].get("regime") != dec["regime"]:
                st["events"].append(dict(time=stamp, regime=dec["regime"], text=f"G: حالة السوق {REGIME_AR[dec['regime']]}، ميزانية المخاطرة {dec['budget']*100:.0f}%"))
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
                              risk_usd=round(p["qty"] * max(0.0, price.get(k, p["entry"]) - p["stop"]), 2),
                              value=p["qty"] * price.get(k, p["entry"]), entry=p["entry"], sl=p["stop"], tp=p["entry"] + PARTIAL_R * p["r"],
                              pnl=price.get(k, p["entry"]) / p["entry"] - 1, signal=p.get("regime", ""), protected=p["stop"] >= p["entry"]) for k, p in st["pos"].items()}
    sells = [t for t in st["trades"] if t["side"] == "SELL"]
    st["stats"] = dict(trades=len(sells), gate_trades=20, gate=f"مغلقة {len(sells)} · الحكم بعد 3 أشهر مقابل الاحتفاظ بـBTC والشراء الدوري")
    st["rules"] = dict(coins=COINS, agent=True, real_money_mode=True, account_dd_limit=cfg["max_drawdown"], watch_n=WATCH_N, max_pos=cfg["max_positions"],
                       vol_unit=VOL_UNIT, init_atr=INIT_ATR, trail_atr=TRAIL_ATR, partial_r=PARTIAL_R, risk=cfg)
    st["control"] = dict(mode=cfg["mode"], emergency_stop=cfg["emergency_stop"], halt_until=st.get("halt_until", 0),
                         streak_until=st.get("streak_until", 0), prices_ok=price_ok)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "paper_equity_G.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "equity", "btc"])
        w.writerow([stamp, round(v, 2), round(price.get("BTC", 0), 2)])
    if jrec:
        st["journal_head"] = journal.append(jrec)
    ok, n, bad = journal.verify()
    st["journal"] = dict(records=n, intact=ok, first_bad_line=bad)
    st["accounting"] = accounting.summarize(st, path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "costs.json"))
    json.dump(st, open(STATE, "w"), indent=1, ensure_ascii=False)
    print(f"[{stamp}] G ${v:,.2f} ({(v/CAPITAL-1)*100:+.2f}%) regime {st.get('regime')} budget {st.get('budget')} open {list(st['pos']) or 'none'}")


if __name__ == "__main__":
    backtest() if "backtest" in sys.argv else cycle()
