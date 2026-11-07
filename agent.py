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
  5. Exits: initial stop 2.5 ATR under entry; then a trailing stop (since 8 Nov 2026, by the research rule: the lowest
     low of the last 20 days, the Turtle exit; before: 3 ATR under the highest close).  At +2R (R = entry - initial
     stop) half is sold and the stop moves to break-even.  If the budget falls, the weakest positions are cut until the
     account is back inside it (a bear regime sells everything).  All of these are parameters in strategy.json.
  6. Every day a decision report in Arabic: the regime and why, the budget, the watchlist, what was done and why,
     what would change the view.  New entries also come as "shadow orders": the exact OKX spot limit order a person
     would place by hand (pair, price, size for a given capital, stop).  Nothing is ever sent to an exchange.

`python agent.py`            one cycle: resting stop/target orders replayed on 15-minute candles since the last
                             check (as the exchange would fill them); decisions once per new UTC day
`python agent.py backtest`   the same rules day by day since 2019 against BTC hold and weekly DCA, plus the ablation
                             "same agent without the market reading" (budget always 100%) to show what the regime adds
State: lab_data/paper_G.json, paper_equity_G.csv, agent_report.json, backtest_G.json.  Paper only, no keys.
"""
import csv, json, math, os, sys, time
from datetime import datetime, timezone

import market as mk
import policy
import risk

OUT = mk.OUT
STATE = os.path.join(OUT, "paper_G.json")
REPORT = os.path.join(OUT, "agent_report.json")
CAPITAL = 5000.0
_RULINGS = policy.site_rulings(OUT)          # refreshed daily from cryptohalal.cc by halal_check.py
UNIVERSE = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC", "OP", "DOT", "TRX", "ATOM", "NEAR",
            "XLM", "HBAR", "SUI", "APT", "ARB", "ALGO", "FIL", "ICP", "ETC", "BCH", "ZRO", "ZK", "ZEC", "TAO",
            "RENDER", "QNT", "WLD", "STX", "GRAM"]
COINS = policy.permitted(UNIVERSE, _RULINGS)   # only coins ruled permissible (مباح); see policy.py
_OK = frozenset(COINS)
HERE = os.path.dirname(os.path.abspath(__file__))
STRATEGY_FILE = os.environ.get("AGENT_STRATEGY") or os.path.join(HERE, "strategy.json")
# The strategy's parameters.  The defaults are the rules the agent started with; strategy.json (versioned in the repo)
# holds the ones adopted since, each adoption decided by the pre-registered research rule (see research()).
DEFAULT_PARAMS = dict(
    watch_n=8,            # leaders kept on the watchlist
    max_pos=6,            # most positions the strategy asks for (the risk engine has its own cap)
    slot=4,               # one position is at most budget / slot of the account
    vol_unit=0.0075,      # ... and at most vol_unit / the coin's daily volatility
    init_atr=2.5,         # initial stop: entry - init_atr x ATR
    trail_atr=3.0,        # trailing stop: highest close - trail_atr x ATR (exit == "trail")
    partial_r=2.0,        # sell half at entry + partial_r x R, then stop to break-even (0 = never)
    target2_r=4.0,        # second target shown to the owner (the trailing stop decides the real exit)
    breakout_n=20,        # entry: close above the highest high of the previous breakout_n days
    resume=True,          # entry: also the resume after a pullback to EMA20
    exit="trail",         # "trail" (ATR trailing) or "donchian" (stop at the lowest low of exit_low_n days)
    exit_low_n=20,
    time_stop_days=0,     # >0: close a position that has not made time_stop_r x R after this many days
    time_stop_r=1.0,
    pyramid=False,        # add one more unit once a position is pyramid_r x R in profit (stop of all moves up)
    pyramid_r=1.0,
    ml=False,             # entry filter: walk-forward logistic model of the chance a signal wins (ml_model_G.json)
)


def load_params(path=STRATEGY_FILE):
    p = dict(DEFAULT_PARAMS)
    try:
        p.update({k: v for k, v in json.load(open(path)).get("params", {}).items() if k in DEFAULT_PARAMS})
    except Exception:
        pass
    return p


PARAMS = load_params()
WATCH_N, MAX_POS, SLOT = PARAMS["watch_n"], PARAMS["max_pos"], PARAMS["slot"]
VOL_UNIT = PARAMS["vol_unit"]
INIT_ATR, TRAIL_ATR, PARTIAL_R = PARAMS["init_atr"], PARAMS["trail_atr"], PARAMS["partial_r"]
TARGET2_R = PARAMS["target2_r"]
ML_FILE = os.path.join(OUT, "ml_model_G.json")
BRK_TXT = f"كسر قمة {PARAMS['breakout_n']} يوماً"
FEE, SLIP = mk.FEE, mk.SLIP
DAY = 86400000
OKX_FILE = os.path.join(OUT, "okx_instruments.json")
REGIME_AR = {"bull": "صاعد", "early": "بداية صعود", "turn": "محاولة انعطاف", "bear": "هابط", "weak": "ضعيف بلا اتجاه"}


# ---------------------------------------------------------------- indicators per coin (daily)
def indicators(rows):
    c = [r[4] for r in rows]
    n = len(c)
    e20, e50, e200 = mk.ema(c, 20), mk.ema(c, 50), mk.ema(c, 200)
    at = mk.atr(rows)
    vol = [None] * n
    for i in range(21, n):
        rs = [c[j] / c[j - 1] - 1 for j in range(i - 19, i + 1)]
        m = sum(rs) / 20
        vol[i] = math.sqrt(sum((x - m) ** 2 for x in rs) / 19)
    return dict(day=[r[0] // DAY for r in rows], o=[r[1] for r in rows], h=[r[2] for r in rows], l=[r[3] for r in rows],
                c=c, e20=e20, e50=e50, e200=e200, atr=at, vol=vol)


def roll(ind, key, n, fn, incl):
    """Rolling max/min of ind[key] over n bars: the previous n (incl=False: k-n..k-1) or the last n (incl=True: k-n+1..k).
    Computed once per series and window, then reused by every backtest run."""
    cache = ind.setdefault("_roll", {})
    ck = (key, n, fn.__name__, incl)
    if ck not in cache:
        a, out = ind[key], [None] * len(ind[key])
        for k in range(len(a)):
            lo, hi = (k - n + 1, k + 1) if incl else (k - n, k)
            if lo >= 0:
                out[k] = fn(a[lo:hi])
        cache[ck] = out
    return cache[ck]


def snap(ind, k, brk_n=20):
    """Everything the agent may know about one coin at the close of its bar k (cached: the same bar is the same snap)."""
    cache = ind.setdefault("_snaps", {}).setdefault(brk_n, {})
    if k in cache:
        return cache[k]
    c = ind["c"]
    if k < max(60, brk_n):
        cache[k] = None
        return None
    hi_brk = roll(ind, "h", brk_n, max, False)[k]
    hi252 = max(ind["h"][max(0, k - 252):k + 1])
    under_recent = any(c[j] < ind["e20"][j] for j in range(max(0, k - 5), k))
    out = dict(c=c[k], o_next=None, e20=ind["e20"][k], e50=ind["e50"][k], e200=ind["e200"][k], atr=ind["atr"][k],
               vol=ind["vol"][k], ret63=(c[k] / c[k - 63] - 1) if k >= 63 else None, near_high=c[k] / hi252 - 1,
               breakout=c[k] > hi_brk, brk_n=brk_n, resume=under_recent and c[k] > ind["e20"][k], low=ind["l"][k], high=ind["h"][k])
    cache[k] = out
    return out


# ---------------------------------------------------------------- ML entry filter (walk-forward, pure Python)
FEATURES = ["rel_strength", "near_high", "atr_pct", "above_e20", "e20_over_e50", "e50_over_e200", "breadth",
            "bull", "early", "turn", "breakout", "btc_over_e200", "rank"]


def features(s, b63, breadth, reg, btc, rank, watch_n):
    """What is known at the close when a signal fires; nothing from the future."""
    return [(s["ret63"] or 0.0) - b63, s["near_high"], (s["atr"] or 0.0) / s["c"], s["c"] / s["e20"] - 1, s["e20"] / s["e50"] - 1,
            (s["e50"] / s["e200"] - 1) if s["e200"] else 0.0, breadth, float(reg == "bull"), float(reg == "early"), float(reg == "turn"),
            float(bool(s["breakout"])), (btc["c"] / btc["e200"] - 1) if btc and btc["e200"] else 0.0, rank / max(1, watch_n)]


def logit_fit(X, y, l2=5.0, iters=300, lr=0.5):
    """Logistic regression by gradient descent on standardized features (ridge penalty).  Returns a plain dict."""
    n, m = len(X), len(X[0])
    mu = [sum(r[j] for r in X) / n for j in range(m)]
    sd = [math.sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / n) or 1.0 for j in range(m)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(m)] for r in X]
    pos = sum(y)
    w, b = [0.0] * m, math.log((pos + 1) / (n - pos + 1))
    for _ in range(iters):
        gw, gb = [0.0] * m, 0.0
        for z, t in zip(Z, y):
            u = b + sum(wj * zj for wj, zj in zip(w, z))
            e = 1 / (1 + math.exp(-max(-30.0, min(30.0, u)))) - t
            gb += e
            for j in range(m):
                gw[j] += e * z[j]
        b -= lr * gb / n
        for j in range(m):
            w[j] -= lr * (gw[j] + l2 * w[j]) / n
    return dict(mu=mu, sd=sd, w=w, b=b, n=n, base=pos / n, features=FEATURES)


def logit_prob(model, x):
    u = model["b"] + sum(wj * (xj - mj) / sj for wj, xj, mj, sj in zip(model["w"], x, model["mu"], model["sd"]))
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, u))))


def train_filter(trades, min_n=60):
    """A model from closed trades that carry entry features (exits at the end of a run are not outcomes: left out)."""
    rows = [(t["feat"], 1 if t["ret"] > 0 else 0) for t in trades if t.get("feat") and t.get("why") != "end"]
    if len(rows) < min_n or len({y for _, y in rows}) < 2:
        return None
    return logit_fit([x for x, _ in rows], [y for _, y in rows])


class EntryFilter:
    """Takes a signal only if the model trained on trades closed BEFORE that day rates it at least as likely to win as
    the average signal it was trained on (threshold = the training base rate, fixed in advance).  No model yet: pass."""
    def __init__(self, models):
        self.models = sorted(models, key=lambda x: x[0])           # [(from_day, model)]
        self.seen = self.kept = 0

    def __call__(self, day, feat):
        m = None
        for d0, mod in self.models:
            if d0 <= day:
                m = mod
        if not m:
            return True
        self.seen += 1
        ok = logit_prob(m, feat) >= m["base"]
        self.kept += ok
        return ok


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


def plan(snaps, positions, equity, values, force_budget=None, prm=None, entry_filter=None, tradable=None):
    """snaps: {coin: snap or None} at today's close.  positions: {coin: {...}} with current values in `values`.
    prm: strategy parameters (default PARAMS).  entry_filter(feat) -> bool (the ML filter, optional).
    tradable: coins that can actually be bought (on OKX spot); the others still count for breadth.
    Returns the decision: regime, budget, watchlist, exits [(coin, frac, why)], entries [(coin, weight, stop)], feats."""
    prm = prm or PARAMS
    btc = snaps.get("BTC")
    live = {k: s for k, s in snaps.items() if s}
    breadth = sum(1 for s in live.values() if s["c"] > s["e50"]) / max(1, len(live))
    reg, budget = regime(btc, breadth) if btc else ("weak", 0.0)
    if force_budget is not None:
        budget = force_budget
    b63 = btc["ret63"] if btc and btc["ret63"] is not None else 0.0
    scored = []
    for k, s in live.items():
        if s["ret63"] is None or s["c"] <= s["e50"] or (tradable is not None and k not in tradable):
            continue
        scored.append(((s["ret63"] - b63) + 0.5 * s["near_high"], k))
    scored.sort(reverse=True)
    watch = [k for _, k in scored[:prm["watch_n"]]]
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
    entries, feats, filtered = [], {}, []
    room = budget * equity - held_val
    n_pos = len(positions) - sum(1 for e in exits if e[1] == 1.0)
    if budget > 0:
        for rank, k in enumerate(watch):
            s = live[k]
            if k in positions or n_pos >= prm["max_pos"] or room < 0.02 * equity:
                continue
            signal = s["breakout"] or (prm["resume"] and s["resume"])
            if not (s["c"] > s["e20"] > s["e50"] and signal) or not s["vol"] or not s["atr"]:
                continue
            f = features(s, b63, breadth, reg, btc, rank, prm["watch_n"])
            if entry_filter is not None and not entry_filter(f):
                filtered.append(k)
                continue
            w = min(budget / prm["slot"], prm["vol_unit"] / s["vol"], room / equity)
            if w < 0.02:
                continue
            entries.append((k, w, s["c"] - prm["init_atr"] * s["atr"]))
            feats[k] = f
            room -= w * equity
            n_pos += 1
    return dict(regime=reg, budget=budget, breadth=breadth, watch=watch, score={k: score[k] for k in watch}, exits=exits, entries=entries,
                feats=feats, filtered=filtered, b63=b63,
                btc=dict(c=btc["c"], e50=btc["e50"], e200=btc["e200"]) if btc else None)


# ---------------------------------------------------------------- backtest (champion = rules alone, challenger = rules + risk engine)
def backtest_run(data, start_day, end_day, force_budget=None, cost_mult=1.0, risk_cfg=None, edge_table=None, inds=None, idx=None, core=None,
                 prm=None, entry_filter=None, tradable=None):
    """Day by day: decide at the close of day d, execute at the open of d+1, stops on d+1's range, trailing on its close.
    risk_cfg None = champion (the strategy's own sizing, no risk engine).  With risk_cfg: every risk gate that can be
    simulated on daily bars (per-trade risk, stop cap, position cap, correlation cap, daily/weekly loss, drawdown halt,
    loss-streak and asset cooldowns, min-edge gate).  Spread and data-freshness gates cannot be simulated and are live-only.
    prm: strategy parameters (default PARAMS).  entry_filter(day, feat) -> bool: the walk-forward ML filter (optional).
    tradable: coins buyable on OKX spot (None = all).  Closed trades carry the entry features for training the filter."""
    prm = prm or PARAMS
    fee, slip = FEE * cost_mult, SLIP * cost_mult
    if inds is None:
        inds = {k: indicators(v) for k, v in data.items()}
        idx = {k: {d: i for i, d in enumerate(ind["day"])} for k, ind in inds.items()}
    cash, pos, trades, curve, reg_days = CAPITAL, {}, [], [], {}
    core_pos = {}                         # core sleeve: {coin: dict(qty, basis, day)}; no ATR stops, trend filter only
    share = core["share"] if core else 0.0
    acc = dict(fees=0.0, slip=0.0, traded=0.0, rejected={}, filtered=0, adds=0)
    peak, halt_until, streak_until, cooldown = CAPITAL, -1, -1, {}
    prev_eq, week_start, week_id, closed = CAPITAL, CAPITAL, None, []
    brk_n, part_r = prm["breakout_n"], prm["partial_r"]
    days = [d for d in inds["BTC"]["day"] if start_day <= d <= end_day]
    for t, d in enumerate(days[:-1]):
        nd = days[t + 1]
        px = {k: inds[k]["c"][idx[k][d]] for k in pos if d in idx[k]}
        values = {k: pos[k]["qty"] * px.get(k, pos[k]["entry"]) for k in pos}
        core_vals = {k: core_pos[k]["qty"] * inds[k]["c"][idx[k][d]] for k in core_pos if d in idx[k]}
        equity = cash + sum(values.values()) + sum(core_vals.values())
        curve.append((d, equity, sum(values.values()) + sum(core_vals.values())))
        snaps = {k: snap(inds[k], idx[k][d], brk_n) if d in idx[k] else None for k in inds}

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
                trades.append(dict(coin=k, ret=tot, pnl=p["pnl"], why=why, days=nd - p["day"], regime=p["regime"], feat=p.get("feat"),
                                   entry_day=p["day"], added=p.get("added", False)))
                closed.append(tot)
                if risk_cfg and why == "stop":
                    cooldown[k] = nd + risk_cfg["asset_cooldown_days"]
                if risk_cfg and risk.losing_streak(closed) >= risk_cfg["loss_streak"]:
                    nonlocal streak_until
                    streak_until = nd + risk_cfg["streak_cooldown_days"]
                    closed.append(1e-9)                    # the streak has been acted on

        allow, reasons = True, []
        if risk_cfg:
            if halt_until >= 0 and d > halt_until:             # halt over: restart the drawdown count from today
                halt_until, peak = -1, equity
            peak = max(peak, equity)
            if halt_until < d and risk.drawdown_breach(risk_cfg, equity, peak):
                halt_until = d + risk_cfg["halt_days"]
                for k in list(pos):
                    if nd in idx[k]:
                        sell(k, 1.0, inds[k]["o"][idx[k][nd]], "risk_limit")
                for k in list(core_pos):
                    if nd in idx[k]:
                        cp = core_pos.pop(k)
                        g = cp["qty"] * inds[k]["o"][idx[k][nd]]
                        cash += g * (1 - slip) * (1 - fee)
                        acc["fees"] += g * fee; acc["slip"] += g * slip; acc["traded"] += g
                        trades.append(dict(coin=k, ret=g * (1 - slip) * (1 - fee) / cp["basis"] - 1, pnl=g * (1 - slip) * (1 - fee) - cp["basis"],
                                           why="risk_limit", days=nd - cp["day"], regime="core"))
            wk = (d + 3) // 7
            if wk != week_id:
                week_id, week_start = wk, equity
            acct = dict(equity=equity, peak=peak, day_start_eq=prev_eq, week_start_eq=week_start,
                        halt_until=(halt_until + 1) * DAY if halt_until >= d else 0, streak_until=(streak_until + 1) * DAY if streak_until >= d else 0)
            allow, reasons = risk.global_gates(risk_cfg, acct, (d + 1) * DAY - 1, 0, 1.0)
        prev_eq = equity
        sat_equity = equity * (1 - share)
        flt = (lambda f, _d=d: entry_filter(_d, f)) if entry_filter else None
        dec = plan(snaps, pos, sat_equity, values, 0.0 if (risk_cfg and halt_until >= d) else force_budget, prm=prm,
                   entry_filter=flt, tradable=tradable)
        acc["filtered"] += len(dec["filtered"])
        # ---- core sleeve: hold BTC/ETH while their own trend is up (close > EMA200 and EMA50 > EMA200), cash when the
        #      close falls under EMA200; decided at the close, executed at the next open
        if core:
            halted = risk_cfg and halt_until >= d
            for k in core["coins"]:
                s_ = snaps.get(k)
                if not s_ or nd not in idx[k]:
                    continue
                o = inds[k]["o"][idx[k][nd]]
                if k in core_pos and (s_["c"] < s_["e200"] or halted):
                    cp = core_pos.pop(k)
                    g = cp["qty"] * o
                    pr = g * (1 - slip) * (1 - fee)
                    cash += pr
                    acc["fees"] += g * fee; acc["slip"] += g * slip; acc["traded"] += g
                    trades.append(dict(coin=k, ret=pr / cp["basis"] - 1, pnl=pr - cp["basis"], why="core_exit", days=nd - cp["day"], regime="core"))
                elif k not in core_pos and not halted and s_["c"] > s_["e200"] and s_["e50"] > s_["e200"]:
                    invest = min(cash, equity * share / len(core["coins"]))
                    if invest > 20:
                        cash -= invest
                        acc["fees"] += invest * fee; acc["slip"] += invest * slip; acc["traded"] += invest
                        core_pos[k] = dict(qty=invest * (1 - fee) / (o * (1 + slip)), basis=invest, day=nd)
        reg_days.setdefault(dec["regime"], []).append(d)
        for k, frac, why in dec["exits"]:
            if k in pos and nd in idx[k]:
                sell(k, frac, inds[k]["o"][idx[k][nd]], why)
        # ---- time stop: a position that has not reached time_stop_r x R after time_stop_days is dead money
        if prm["time_stop_days"]:
            for k in list(pos):
                p = pos[k]
                if nd in idx[k] and d - p["day"] >= prm["time_stop_days"] and not p["half"] and p["high"] < p["entry"] + prm["time_stop_r"] * p["r"]:
                    sell(k, 1.0, inds[k]["o"][idx[k][nd]], "time")
        held_vals = {k: pos[k]["qty"] * px.get(k, pos[k]["entry"]) for k in pos}
        n_pos = len(pos)

        def size(k, w, stop, n_count):
            """Dollars for a new unit: the strategy's ask, or the risk engine's verdict.  0 = refused."""
            s = snaps[k]
            if not risk_cfg:
                return min(cash, w * sat_equity)
            if not allow:
                for r_ in reasons:
                    acc["rejected"][r_] = acc["rejected"].get(r_, 0) + 1
                return 0.0
            ok_edge, _, _ = risk.edge_eligible(risk_cfg, edge_at(edge_table, d), dec["regime"])
            if not ok_edge:
                acc["rejected"]["insufficient_edge"] = acc["rejected"].get("insufficient_edge", 0) + 1
                return 0.0
            rets = {}
            for kk in [k] + list(pos):
                j = idx[kk].get(d)
                if j is not None and j >= 61:
                    rets[kk] = risk.returns(inds[kk]["c"][:j + 1])
            ctx = dict(equity=sat_equity, cash=cash, budget=dec["budget"], held_values=held_vals, n_positions=n_count,
                       rets=rets, cooldowns={c: (u + 1) * DAY for c, u in cooldown.items()}, now_ms=(d + 1) * DAY - 1)
            usd, why_not = risk.size_entry(risk_cfg, dict(coin=k, price=s["c"], stop=stop, weight=w), ctx)
            if usd <= 0:
                for r_ in why_not:
                    acc["rejected"][r_] = acc["rejected"].get(r_, 0) + 1
                return 0.0
            return min(cash, usd)

        for k, w, stop in dec["entries"]:
            if nd not in idx[k]:
                continue
            s = snaps[k]
            invest = size(k, w, stop, n_pos)
            if invest < 20:
                continue
            o = inds[k]["o"][idx[k][nd]]
            cash -= invest
            acc["fees"] += invest * fee
            acc["slip"] += invest * slip
            acc["traded"] += invest
            st_ = o - prm["init_atr"] * s["atr"]
            pos[k] = dict(qty=invest * (1 - fee) / (o * (1 + slip)), basis=invest, basis0=invest, entry=o, stop=st_, r=o - st_, high=o,
                          half=False, day=nd, regime=dec["regime"], feat=dec["feats"].get(k))
            held_vals[k] = invest
            n_pos += 1
        # ---- pyramiding: one more unit once the position is pyramid_r x R up; the stop of the whole position rises to
        #      the new unit's initial stop (so the first unit is at least at break-even)
        if prm["pyramid"] and dec["budget"] > 0 and not (risk_cfg and halt_until >= d):
            room = dec["budget"] * sat_equity - sum(held_vals.values())
            for k in list(pos):
                p, s = pos[k], snaps.get(k)
                if p.get("added") or p["day"] >= nd or not s or nd not in idx[k] or not s["vol"] or not s["atr"]:
                    continue
                if s["c"] < p["entry"] + prm["pyramid_r"] * p["r"] or room < 0.02 * sat_equity:
                    continue
                w = min(dec["budget"] / prm["slot"], prm["vol_unit"] / s["vol"], room / sat_equity)
                new_stop = s["c"] - prm["init_atr"] * s["atr"]
                invest = size(k, w, new_stop, max(0, n_pos - 1))
                if invest < 20:
                    continue
                o = inds[k]["o"][idx[k][nd]]
                cash -= invest
                acc["fees"] += invest * fee; acc["slip"] += invest * slip; acc["traded"] += invest
                p["qty"] += invest * (1 - fee) / (o * (1 + slip))
                p["basis"] += invest
                p["basis0"] += invest
                p["added"] = True
                p["stop"] = max(p["stop"], o - prm["init_atr"] * s["atr"])
                held_vals[k] = held_vals.get(k, 0) + invest
                room -= invest
                acc["adds"] += 1
        for k in list(pos):
            if nd not in idx[k]:
                continue
            j = idx[k][nd]
            p, ind = pos[k], inds[k]
            if ind["l"][j] <= p["stop"]:
                sell(k, 1.0, min(p["stop"], ind["o"][j]), "stop")
                continue
            if part_r > 0 and not p["half"] and ind["h"][j] >= p["entry"] + part_r * p["r"]:
                sell(k, 0.5, max(p["entry"] + part_r * p["r"], ind["o"][j]), "partial")
                p["half"], p["stop"] = True, max(p["stop"], p["entry"])
            p["high"] = max(p["high"], ind["c"][j])
            if prm["exit"] == "donchian":
                lo = roll(ind, "l", prm["exit_low_n"], min, True)[j]
                if lo is not None:
                    p["stop"] = max(p["stop"], lo)
            else:
                p["stop"] = max(p["stop"], p["high"] - prm["trail_atr"] * (ind["atr"][j] or 0))
    d = days[-1]
    for k in list(pos):
        if d in idx[k]:
            p = pos.pop(k)
            pr = p["qty"] * inds[k]["c"][idx[k][d]] * (1 - slip) * (1 - fee)
            cash += pr
            p["pnl"] = p.get("pnl", 0.0) + pr - p["basis"]
            trades.append(dict(coin=k, ret=p["pnl"] / p["basis0"], pnl=p["pnl"], why="end", days=d - p["day"], regime=p["regime"],
                               feat=p.get("feat"), entry_day=p["day"]))
    for k in list(core_pos):
        if d in idx[k]:
            cp = core_pos.pop(k)
            pr = cp["qty"] * inds[k]["c"][idx[k][d]] * (1 - slip) * (1 - fee)
            cash += pr
            trades.append(dict(coin=k, ret=pr / cp["basis"] - 1, pnl=pr - cp["basis"], why="end", days=d - cp["day"], regime="core"))
    curve.append((d, cash, 0.0))
    return curve, trades, reg_days, acc


def edge_at(edge_table, day):
    """The min-edge table in force on `day`: a static dict, or the latest of a walk-forward list [(from_day, table)]."""
    if not isinstance(edge_table, list):
        return edge_table
    cur = {}
    for d0, tab in edge_table:
        if d0 <= day:
            cur = tab
    return cur


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


def okx_tradable():
    """Coins listed for spot trading on OKX (cached daily by okx_rules); None if unknown (then nothing is excluded)."""
    try:
        return set(json.load(open(OKX_FILE))["rules"])
    except Exception:
        return None


def load_data(n=3000):
    data = {}
    for c in COINS:
        try:
            rows = mk.klines(c, "1d", n)
            if len(rows) >= 300:
                data[c] = rows
        except Exception as e:
            print(f"  ! {c}: {e}")
    return data


class Lab:
    """Everything a historical test needs, built once: indicators, folds, OKX-tradable coins, and the rules-alone trades
    closed before each fold start (for the walk-forward min-edge tables and the ML filter's training)."""
    def __init__(self, data):
        self.data = data
        self.inds = {k: indicators(v) for k, v in data.items()}
        self.idx = {k: {d: i for i, d in enumerate(ind["day"])} for k, ind in self.inds.items()}
        days = self.inds["BTC"]["day"]
        self.start, self.end = days[260], days[-1]
        span = [d for d in days if self.start <= d <= self.end]
        self.folds = [(span[len(span) * i // 5], span[min(len(span) - 1, len(span) * (i + 1) // 5)]) for i in range(5)]
        t = None if mk.SYN else okx_tradable()
        self.tradable = {c for c in data if c in t} if t else None
        self._rules = {}

    def run(self, a, b, **kw):
        return backtest_run(self.data, a, b, inds=self.inds, idx=self.idx, tradable=self.tradable, **kw)

    def rules_trades(self, prm, day):
        key = (json.dumps(prm, sort_keys=True), day)
        if key not in self._rules:
            self._rules[key] = [] if day - self.start < 120 else self.run(self.start, day, prm=dict(prm, ml=False))[1]
        return self._rules[key]

    def edges(self, prm):
        return [(a, risk.edge_table_from(self.rules_trades(prm, a))) for a, _ in self.folds]

    def models(self, prm):
        out = []
        for a, _ in self.folds:
            m = train_filter(self.rules_trades(prm, a))
            if m:
                out.append((a, m))
        return out

    def evaluate(self, prm, cfg):
        """Folds, full period and costs x3 for one strategy, all walk-forward (edge tables and ML models only ever use
        trades closed before the day they are applied).  Returns (result, full-period curve)."""
        edges = self.edges(prm)
        models = self.models(prm) if prm.get("ml") else None
        flt = lambda: EntryFilter(models) if models is not None else None
        fr = []
        for a, b in self.folds:
            c, tr, _, _ = self.run(a, b, prm=prm, risk_cfg=cfg, edge_table=edges, entry_filter=flt())
            cs = curve_stats(c)
            fr.append(dict(start=day_str(a)[:7], end=day_str(b)[:7], ret=cs["ret"], max_dd=cs["max_dd"], cagr=cs["cagr"],
                           calmar=cs["calmar"], trades=len(tr)))
        f_full = flt()
        c, tr, rd, acc = self.run(self.start, self.end, prm=prm, risk_cfg=cfg, edge_table=edges, entry_filter=f_full)
        c3, _, _, _ = self.run(self.start, self.end, prm=prm, risk_cfg=cfg, edge_table=edges, entry_filter=flt(), cost_mult=3.0)
        cs = curve_stats(c)
        res = dict(full=cs, full_x3=curve_stats(c3), folds=fr, yearly=yearly(c),
                   trades=trade_stats(tr, acc, sum(p[1] for p in c) / len(c), max(0.1, cs["days"] / 365.25)),
                   regime_share={k: len(v) / max(1, sum(len(x) for x in rd.values())) for k, v in rd.items()},
                   filter=dict(seen=f_full.seen, kept=f_full.kept, models=len(models)) if f_full else None, adds=acc["adds"])
        return res, c


def day_str(d):
    return datetime.fromtimestamp(d * 86400, timezone.utc).strftime("%Y-%m-%d")


def evaluate(data):
    lab = Lab(data)
    start, end, folds = lab.start, lab.end, lab.folds
    cfg = risk.load_control()
    prm = dict(PARAMS)
    run = lab.run
    out = dict(generated=datetime.now(timezone.utc).isoformat(), synthetic=mk.SYN, coins=len(data), tradable=sorted(lab.tradable) if lab.tradable else None,
               period=[day_str(start), day_str(end)], candidates_tested=2, promotion_rule=PROMOTION_RULE, risk_config=cfg, params=prm,
               caveats=["survivorship: the universe is today's listed permissible coins",
                        "the 2024-2026 'exam' was lived through before the rules were written: not a truly unseen holdout",
                        "spread and data-freshness gates are live-only and not simulated",
                        "the only truly unseen test is the live paper record from 2026-11-02"])
    res = {}
    for name, kw in (("champion", None), ("challenger", cfg)):
        if kw is None:                     # rules alone: no risk engine, no edge gate, same parameters
            fold_rows = []
            for a, b in folds:
                c, tr, _, _ = run(a, b, prm=prm)
                cs = curve_stats(c)
                fold_rows.append(dict(start=day_str(a)[:7], end=day_str(b)[:7], ret=cs["ret"], max_dd=cs["max_dd"], trades=len(tr)))
            c, tr, rd, acc = run(start, end, prm=prm)
            c3, _, _, _ = run(start, end, prm=prm, cost_mult=3.0)
            cs = curve_stats(c)
            res[name] = dict(full=cs, full_x3=curve_stats(c3), folds=fold_rows,
                             trades=trade_stats(tr, acc, sum(p[1] for p in c) / len(c), max(0.1, cs["days"] / 365.25)),
                             regime_share={k: len(v) / max(1, sum(len(x) for x in rd.values())) for k, v in rd.items()}, yearly=yearly(c))
        else:
            r, c = lab.evaluate(prm, cfg)
            res[name] = r
            out["challenger_curve_end"] = c[-1][1]
        cs = res[name]["full"]
        print(f"{name}: {cs['ret']*100:+.1f}% DD {cs['max_dd']*100:.0f}% CAGR {((cs['cagr'] or 0)*100):+.1f}% Sharpe {cs['sharpe'] or 0:.2f} "
              f"Calmar {cs['calmar'] or 0:.2f} | x3 {res[name]['full_x3']['ret']*100:+.1f}% | folds "
              + " ".join(f"{f['ret']*100:+.0f}%" for f in res[name]["folds"]))
    hold, dca = btc_rulers(data["BTC"], start, end)
    nr, _, _, _ = run(start, end, force_budget=1.0, prm=prm)
    out.update(res, btc_hold=hold, btc_dca=dca, no_regime=curve_stats(nr), btc_yearly=yearly([(r[0] // DAY, r[4]) for r in data["BTC"] if start <= r[0] // DAY <= end]))
    ch, cc = res["champion"], res["challenger"]
    rules = [cc["full"]["max_dd"] <= ch["full"]["max_dd"] - 0.05,
             (cc["full"]["calmar"] or -9) >= (ch["full"]["calmar"] or -9) or (cc["full"]["sharpe"] or -9) >= (ch["full"]["sharpe"] or -9),
             sum(1 for f in cc["folds"] if f["ret"] > 0) >= 3,
             cc["full_x3"]["ret"] > 0]
    out["promotion"] = dict(challenger=all(rules), checks=rules)
    # live edge table: every rules-alone trade with today's parameters, all in the past by construction
    _, tr_all, _, _ = run(start, end, prm=dict(prm, ml=False))
    out["edge_by_regime"] = risk.edge_table_from(tr_all)
    if prm.get("ml") and not mk.SYN:      # live ML filter: trained on every closed rules-alone trade so far
        m = train_filter(tr_all)
        if m:
            json.dump(dict(m, trained=datetime.now(timezone.utc).isoformat(), trades=m["n"]), open(ML_FILE, "w"), indent=1)
    best = cc if out["promotion"]["challenger"] else ch
    out["full"] = dict(agent=best["full"], agent_x3=best["full_x3"], no_regime=out["no_regime"], btc_hold=hold, btc_dca=dca,
                       trades=best["trades"]["trades"], win=best["trades"]["win"])
    last = folds[-1]
    e_c, _, _, _ = run(last[0], last[1], prm=prm, risk_cfg=cfg, edge_table=lab.edges(prm))
    e_h, _, _, _ = run(last[0], last[1], prm=prm)
    h2, d2 = btc_rulers(data["BTC"], last[0], last[1])
    out["exam"] = dict(agent=curve_stats(e_c if out["promotion"]["challenger"] else e_h), btc_hold=h2, btc_dca=d2)
    print("promotion:", out["promotion"], "| BTC hold", f"{hold['ret']*100:+.1f}% DD {hold['max_dd']*100:.0f}%")
    return out


# ---------------------------------------------------------------- research: candidates against the live strategy
RESEARCH_RULE = (
    "Pre-registered research rule (written 7 Nov 2026 before any run; candidates live in candidates.json with the date each "
    "was added). Baseline = the live strategy (strategy.json + control.json, risk engine on). Same data, same costs, halal "
    "coins tradable on OKX only, min-edge tables and the ML filter strictly walk-forward. "
    "Stage 1 (each candidate changes one thing): eligible only if (a) full-period max drawdown <= 25% (the owner's limit), "
    "(b) positive return in >= 4 of the 5 consecutive folds, (c) full-period return positive with costs x3, (d) full-period "
    "Calmar >= 1.10 x the baseline's, (e) a better Calmar than the baseline in >= 3 of the 5 folds. The eligible candidate "
    "with the highest Calmar wins. Stage 1b: the two best eligible candidates combined (only if they change different "
    "parameters) replace the winner only if the combination passes (a)-(e) and has a higher Calmar. Stage 2 (only if there "
    "is a winner): risk per trade is raised in steps (1.25%, 1.5%, 1.75%, 2%); the highest step is kept whose full-period "
    "drawdown is no deeper than the baseline's (the drawdown the owner already accepted, +0.5 point) and that still passes "
    "(b) and (c). No winner: nothing changes. Monte Carlo ranges, neighbouring parameter values and the same round on "
    "synthetic noise are reported, never used to decide.")
CANDIDATES_FILE = os.path.join(HERE, "candidates.json")
RISK_KEYS = {"max_positions"}            # the only risk-engine limit a candidate may change (risk per trade: stage 2)
NEIGHBOUR_STEP = dict(breakout_n=10, trail_atr=0.5, partial_r=1.0, init_atr=0.5, exit_low_n=5, time_stop_days=10, time_stop_r=0.5,
                      watch_n=2, pyramid_r=0.5)


def monte_carlo(curve, paths=2000, horizon=1095, block=20, seed=7):
    """Block bootstrap of the strategy's own daily returns: what three years could look like if the future resembles
    the past in pieces of 20 days.  A range, not a forecast."""
    import random
    r = [curve[i][1] / curve[i - 1][1] - 1 for i in range(1, len(curve)) if curve[i - 1][1] > 0]
    if len(r) < block * 3:
        return None
    rnd = random.Random(seed)
    cagrs, dds, loss1 = [], [], 0
    for _ in range(paths):
        path = []
        while len(path) < horizon:
            i = rnd.randrange(0, len(r) - block)
            path.extend(r[i:i + block])
        eq, pk, dd, e365 = 1.0, 1.0, 0.0, 1.0
        for t_, x in enumerate(path[:horizon]):
            eq *= 1 + x
            pk = max(pk, eq)
            dd = max(dd, 1 - eq / pk)
            if t_ == 364:
                e365 = eq
        cagrs.append(eq ** (365 / horizon) - 1)
        dds.append(dd)
        loss1 += e365 < 1
    q = lambda a, p_: sorted(a)[int(p_ * (len(a) - 1))]
    return dict(paths=paths, horizon_days=horizon, block_days=block, cagr_p5=q(cagrs, 0.05), cagr_p50=q(cagrs, 0.5), cagr_p95=q(cagrs, 0.95),
                dd_p50=q(dds, 0.5), dd_p95=q(dds, 0.95), p_loss_first_year=loss1 / paths)


def load_candidates():
    try:
        return [c for c in json.load(open(CANDIDATES_FILE)).get("items", []) if c.get("status", "active") == "active"]
    except Exception:
        return []


def research():
    lab = Lab(load_data())
    base_prm, base_cfg = dict(PARAMS), risk.load_control()
    compact = lambda r: dict(full={k: r["full"].get(k) for k in ("ret", "cagr", "max_dd", "sharpe", "calmar", "exposure")},
                             x3=r["full_x3"]["ret"], folds=r["folds"], trades=r["trades"]["trades"], win=r["trades"]["win"],
                             profit_factor=r["trades"]["profit_factor"], yearly=r["yearly"], filter=r.get("filter"), adds=r.get("adds"))

    def checks(r, ref):
        wins = sum(1 for f, g in zip(r["folds"], ref["folds"]) if (f["calmar"] if f["calmar"] is not None else -9) > (g["calmar"] if g["calmar"] is not None else -9))
        cal, rcal = r["full"]["calmar"] or 0, ref["full"]["calmar"] or 0
        return dict(dd=r["full"]["max_dd"] <= 0.25, folds=sum(1 for f in r["folds"] if f["ret"] > 0) >= 4,
                    costs_x3=r["full_x3"]["ret"] > 0, calmar=cal > 0 and cal >= 1.10 * rcal, fold_wins=wins >= 3)

    def line(name, r):
        f = r["full"]
        print(f"{name:16} ret {f['ret']*100:+7.1f}% CAGR {(f['cagr'] or 0)*100:+5.1f}% DD {f['max_dd']*100:4.1f}% Calmar {f['calmar'] or 0:5.2f} "
              f"x3 {r['full_x3']['ret']*100:+6.1f}% trades {r['trades']['trades']:4d} | folds " + " ".join(f"{x['ret']*100:+.0f}%" for x in r["folds"]))

    base, base_curve = lab.evaluate(base_prm, base_cfg)
    line("baseline", base)
    out = dict(generated=datetime.now(timezone.utc).isoformat(), synthetic=mk.SYN, rule=RESEARCH_RULE,
               period=[day_str(lab.start), day_str(lab.end)], coins=len(lab.data), tradable=sorted(lab.tradable) if lab.tradable else None,
               folds=[[day_str(a), day_str(b)] for a, b in lab.folds], baseline=dict(params=base_prm, risk=base_cfg, result=compact(base)),
               candidates={}, caveats=["survivorship: the universe is today's listed permissible coins",
                                       "every run before 2026-11-02 was history the rules could have been fitted to; the live record is the only unseen test",
                                       "spread and data-freshness gates are live-only",
                                       "the more candidates are tried, the likelier one wins by luck: the fold and cost checks are there for that"])
    cands = load_candidates()
    elig = []
    for cand in cands:
        prm = dict(base_prm, **{k: v for k, v in cand.get("params", {}).items() if k in DEFAULT_PARAMS})
        cfg = dict(base_cfg, **{k: v for k, v in cand.get("risk", {}).items() if k in RISK_KEYS})
        r, curve = lab.evaluate(prm, cfg)
        ck = checks(r, base)
        line(cand["id"], r)
        out["candidates"][cand["id"]] = dict(title=cand.get("title"), params=cand.get("params", {}), risk=cand.get("risk", {}),
                                             result=compact(r), checks=ck, eligible=all(ck.values()))
        if all(ck.values()):
            elig.append((r["full"]["calmar"] or 0, cand["id"], prm, cfg, r, curve, cand))
    elig.sort(key=lambda x: -x[0])
    winner = elig[0] if elig else None
    out["stage1"] = dict(eligible=[e[1] for e in elig], winner=winner[1] if winner else None)
    if len(elig) >= 2:
        a_, b_ = elig[0][6], elig[1][6]
        keys_a = set(a_.get("params", {})) | set(a_.get("risk", {}))
        keys_b = set(b_.get("params", {})) | set(b_.get("risk", {}))
        if not keys_a & keys_b:
            prm = dict(base_prm, **a_.get("params", {}), **b_.get("params", {}))
            cfg = dict(base_cfg, **{k: v for k, v in {**a_.get("risk", {}), **b_.get("risk", {})}.items() if k in RISK_KEYS})
            r, curve = lab.evaluate(prm, cfg)
            ck = checks(r, base)
            name = f"{a_['id']}+{b_['id']}"
            line(name, r)
            better = all(ck.values()) and (r["full"]["calmar"] or 0) > winner[0]
            out["stage1b"] = dict(name=name, result=compact(r), checks=ck, adopted=better)
            if better:
                winner = ((r["full"]["calmar"] or 0), name, prm, cfg, r, curve, dict(id=name, params={**a_.get("params", {}), **b_.get("params", {})},
                                                                                       risk={**a_.get("risk", {}), **b_.get("risk", {})}))
    adopted = None
    if winner:
        _, name, prm, cfg, r, curve, cand = winner
        steps, keep = [], (cfg, r, curve)
        for rpt in (0.0125, 0.015, 0.0175, 0.02):
            cfg2 = dict(cfg, risk_per_trade=rpt)
            r2, c2 = lab.evaluate(prm, cfg2)
            ok = (r2["full"]["max_dd"] <= base["full"]["max_dd"] + 0.005 and sum(1 for f in r2["folds"] if f["ret"] > 0) >= 4 and r2["full_x3"]["ret"] > 0)
            steps.append(dict(risk_per_trade=rpt, result=compact(r2), ok=ok))
            line(f"  risk {rpt*100:.2f}%", r2)
            if ok:
                keep = (cfg2, r2, c2)
        cfg, r, curve = keep
        out["stage2"] = steps
        adopted = dict(name=name, params={k: v for k, v in prm.items() if base_prm.get(k) != v},
                       risk={k: v for k, v in cfg.items() if base_cfg.get(k) != v and k in RISK_KEYS | {"risk_per_trade"}},
                       result=compact(r))
        # neighbours of every changed numeric parameter: is the winner a knife edge?
        nb = []
        for k, v in {**cand.get("params", {}), **adopted["params"]}.items():
            stp = NEIGHBOUR_STEP.get(k)
            if stp is None or isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            for nv in (v - stp, v + stp):
                if nv <= 0:
                    continue
                rr, _ = lab.evaluate(dict(prm, **{k: type(v)(nv)}), cfg)
                nb.append(dict(param=k, value=nv, calmar=rr["full"]["calmar"], cagr=rr["full"]["cagr"], max_dd=rr["full"]["max_dd"]))
        out["neighbours"] = nb
        out["monte_carlo"] = dict(adopted=monte_carlo(curve), baseline=monte_carlo(base_curve))
    else:
        out["monte_carlo"] = dict(baseline=monte_carlo(base_curve))
    out["adopted"] = adopted
    # the live strategy's own neighbourhood: does it still work if its key numbers move a little? (reported only)
    rob = []
    keys = ["breakout_n", "init_atr"] + (["exit_low_n"] if base_prm["exit"] == "donchian" else ["trail_atr"]) + (["partial_r"] if base_prm["partial_r"] > 0 else [])
    for k in keys:
        v = base_prm[k]
        for nv in (v - NEIGHBOUR_STEP[k], v + NEIGHBOUR_STEP[k]):
            if nv > 0:
                rr, _ = lab.evaluate(dict(base_prm, **{k: type(v)(nv)}), base_cfg)
                rob.append(dict(param=k, value=nv, calmar=rr["full"]["calmar"], cagr=rr["full"]["cagr"], max_dd=rr["full"]["max_dd"]))
    out["robustness"] = rob
    hold, dca = btc_rulers(lab.data["BTC"], lab.start, lab.end)
    out.update(btc_hold=hold, btc_dca=dca)
    print("adopted:", adopted["name"] if adopted else "none (the live strategy stays)")
    json.dump(out, open(os.path.join(OUT, "research_G_null.json" if mk.SYN else "research_G.json"), "w"), indent=1)
    return out


def backtest():
    out = evaluate(load_data())
    json.dump(out, open(os.path.join(OUT, "backtest_G_null.json" if mk.SYN else "backtest_G.json"), "w"), indent=1)


# ---------------------------------------------------------------- real-exchange realism (treated as real money)
def okx_rules():
    """OKX spot instrument rules (lot size, minimum size, tick), refreshed once a day; public data, no key."""
    try:
        cache = json.load(open(OKX_FILE))
        if time.time() - cache.get("t", 0) < 86400:
            return cache["rules"]
    except Exception:
        cache = None
    if mk.SYN:
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
                t1=float(f"{price + PARTIAL_R * r:.6g}") if PARTIAL_R > 0 else None, t2=float(f"{price + TARGET2_R * r:.6g}"),
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


BAR = 15 * 60 * 1000      # resting-order checks walk 15-minute candles


def last_check_ms(default):
    """Time of the previous cycle (last row of the equity log), so the next cycle replays only what happened since."""
    try:
        with open(os.path.join(OUT, "paper_equity_G.csv")) as f:
            last = f.read().strip().splitlines()[-1].split(",")[0]
        return int(datetime.strptime(last[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
    except Exception:
        return default


def bars_since(coin, start_ms, now_ms, ref_price=None):
    """15-minute candles [t, o, h, l, c, v] from the one containing start_ms to now (the still-forming one included);
    hourly candles if the gap is longer than 9 days (Binance returns at most 1000 rows)."""
    tf, step = ("15m", BAR) if now_ms - start_ms <= 9 * DAY else ("1h", 4 * BAR)
    t0 = start_ms // step * step
    if mk.SYN:                                       # code test only: a synthetic path scaled to the current price
        rows = [r for r in mk.synthetic(coin, tf, 1500) if r[0] >= t0]
        k = (ref_price / rows[-1][4]) if rows and ref_price else 1.0
        return [[r[0], r[1] * k, r[2] * k, r[3] * k, r[4] * k, r[5]] for r in rows], step
    b = mk.api(f"/api/v3/klines?symbol={coin}USDT&interval={tf}&startTime={t0}&limit=1000")
    return [[int(x[0]), *map(float, x[1:6])] for x in b], step


def resting_orders(pos, start_ms, now_ms, bars, sell):
    """Exits as the exchange executes them.  Right after a buy, the stop (stop-market) and target 1 (limit order for
    half) rest on the exchange, so they fill the moment the price touches them, between our checks too.  This replays
    the candles since the previous check in order:
      - low touches the stop: everything is sold at the stop, or at the candle's open if it opened below the stop (gap);
      - high touches target 1 (half not yet sold): half is sold at target 1 (or at the open if it opened above), and
        the stop moves to break-even;
      - both in one candle: the order inside a candle is unknown, so the stop is assumed first (the worse case).
    The candle in which the position was opened is skipped (its low may predate the entry).
    Same rules as the backtest on daily candles.  bars(coin, start) -> (rows, step); sell(coin, frac, price, why, at_ms)."""
    for c in list(pos):
        p = pos[c]
        start = max(start_ms, p["entry_ms"] - p["entry_ms"] % BAR + BAR)
        if start >= now_ms:
            continue
        try:
            rows, step = bars(c, start)
        except Exception as e:
            print(f"  ! candles {c}: {e}")
            continue
        for t, o, h, l, cl, v in rows:
            if t + step <= start or t > now_ms:
                continue
            at = max(t, start)
            if l <= p["stop"]:
                sell(c, 1.0, min(p["stop"], o), "stop", at)
                break
            t1 = p["entry"] + PARTIAL_R * p["r"]
            if PARTIAL_R > 0 and not p["half"] and h >= t1:
                sell(c, 0.5, max(t1, o), "partial", at)
                p["half"], p["stop"] = True, max(p["stop"], p["entry"])


def cycle():
    import journal, accounting
    cfg = risk.load_control()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    now_ms = int(time.time() * 1000)
    today = now_ms // DAY
    if not os.path.exists(STATE) and os.path.exists(os.path.join(OUT, "paper_equity_G.csv")):
        raise SystemExit("state file missing while the account has history: stop instead of starting a new account")
    st = json.load(open(STATE)) if os.path.exists(STATE) else dict(start=datetime.now(timezone.utc).isoformat(), cash=CAPITAL, pos={},
                                                                    trades=[], events=[], last_day=0, orders=[])
    st.setdefault("cooldown", {})
    jrec = []
    # live prices for held coins (stop and partial checks every cycle)
    price, price_ok = {}, True
    for c in list(st["pos"]) + ["BTC"]:
        try:
            price[c] = mk.last_price(c, mk.klines(c, "1h", 3, closed_only=False) if mk.SYN else None)
        except Exception as e:
            price_ok = False
            print(f"  ! price {c}: {e}")

    def sell(c, frac, px, why, at=None):
        at = at or now_ms
        when = datetime.fromtimestamp(at / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
        p = st["pos"][c]
        q = risk.can_sell(p["qty"], p["qty"] * frac)
        s_ = p.get("slip", SLIP)
        gross = q * px
        proceeds = gross * (1 - s_) * (1 - FEE)
        st["cash"] += proceeds
        part = q / p["qty"] if p["qty"] > 0 else 1.0
        basis = p["basis"] * part
        st["trades"].append(dict(time=when, coin=c, side="SELL", price=px, ret=proceeds / basis - 1, why=why, frac=frac,
                                 usd=round(proceeds, 2), cost_usd=round(basis, 2), pnl_usd=round(proceeds - basis, 2), qty=q,
                                 fee_usd=round(gross * (1 - s_) * FEE, 4), slip_usd=round(gross * s_, 4), pos_id=p.get("pos_id"),
                                 hours=round((at - p["entry_ms"]) / 3600000, 1)))
        jrec.append(dict(t=when, type="exit", coin=c, why=why, frac=frac, price=px, pnl_usd=round(proceeds - basis, 2), pos_id=p.get("pos_id")))
        p["qty"] -= q
        p["basis"] -= basis
        if frac >= 0.999 or p["qty"] <= 1e-12:
            st["pos"].pop(c)
            if why == "stop":
                st["cooldown"][c] = at + cfg["asset_cooldown_days"] * DAY
        print(f"  G SELL {c} {frac*100:.0f}% @ {px:g} ({why}) at {when}")

    # ---- resting exchange orders (stop and target 1) replayed on the candles since the previous check
    resting_orders(st["pos"], last_check_ms(now_ms - 3600000), now_ms,
                   lambda c, start: bars_since(c, start, now_ms, price.get(c)), sell)
    for c in list(st["pos"]):                                # fallback on the live price (candles unavailable)
        px = price.get(c)
        if px is None:
            continue
        p = st["pos"][c]
        if px <= p["stop"]:
            sell(c, 1.0, px, "stop")
            continue
        if PARTIAL_R > 0 and not p["half"] and px >= p["entry"] + PARTIAL_R * p["r"]:
            sell(c, 0.5, px, "partial")
            p["half"], p["stop"] = True, max(p["stop"], p["entry"])
        if c not in _OK and c in st["pos"]:                   # ruling changed or no longer permissible: close it
            sell(c, 1.0, px, "policy")
            st["events"].append(dict(time=stamp, text=f"G: بيع {c}: لم تعد مصنّفة مباحة في موقع الفحص الشرعي"))

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
                rows = mk.klines(c, "1d", 400)
                if len(rows) >= 120:
                    data[c] = rows
            except Exception as e:
                print(f"  ! {c}: {e}")
        if "BTC" in data:
            age_h = 0.0 if mk.SYN else (now_ms - (data["BTC"][-1][0] + DAY)) / 3600000   # synthetic candles are not dated like real ones
            coverage = len(data) / len(COINS)
            inds = {k: indicators(v) for k, v in data.items()}
            snaps = {k: snap(ind, len(ind["c"]) - 1, PARAMS["breakout_n"]) for k, ind in inds.items()}
            for k in st["pos"]:                                  # trailing stop on the daily close
                if k in inds:
                    ind, p = inds[k], st["pos"][k]
                    p["high"] = max(p["high"], ind["c"][-1])
                    if PARAMS["exit"] == "donchian":
                        lo = roll(ind, "l", PARAMS["exit_low_n"], min, True)[-1]
                        if lo is not None:
                            p["stop"] = max(p["stop"], lo)
                    else:
                        p["stop"] = max(p["stop"], p["high"] - TRAIL_ATR * (ind["atr"][-1] or 0))
            for k in snaps:
                if snaps[k] and k not in price:
                    price[k] = snaps[k]["c"]
            values = {k: st["pos"][k]["qty"] * price.get(k, st["pos"][k]["entry"]) for k in st["pos"]}
            equity = st["cash"] + sum(values.values())
            acct = dict(equity=equity, peak=st["peak"], day_start_eq=st.get("day_start_eq"), week_start_eq=st.get("week_start_eq"),
                        halt_until=st.get("halt_until", 0), streak_until=st.get("streak_until", 0))
            allow, gate_reasons = risk.global_gates(cfg, acct, now_ms, age_h, coverage)
            rules = okx_rules()
            flt = None
            if PARAMS["ml"]:
                try:
                    flt = EntryFilter([(0, json.load(open(ML_FILE)))])
                except Exception:
                    flt = None                                   # no model yet: no verdict, signals pass (like the edge gate)
            dec = plan(snaps, st["pos"], equity, values, force_budget=0.0 if st.get("halt_until") else None, prm=PARAMS,
                       entry_filter=(lambda f: flt(today, f)) if flt else None, tradable=set(rules) if rules else None)
            edge_ok, edge_mean, edge_n = risk.edge_eligible(cfg, load_edge(), dec["regime"])
            actions, orders, rejected = [], [], []
            for k, frac, why in dec["exits"]:
                if k in st["pos"]:
                    sell(k, frac, price.get(k, st["pos"][k]["entry"]), why)
                    actions.append(f"بيع {k} لأن الميزانية انخفضت إلى {dec['budget']*100:.0f}%")
            if PARAMS["time_stop_days"]:                         # a position that went nowhere for too long
                for k in list(st["pos"]):
                    p = st["pos"][k]
                    if (now_ms - p["entry_ms"]) / DAY >= PARAMS["time_stop_days"] and not p["half"] and p["high"] < p["entry"] + PARAMS["time_stop_r"] * p["r"]:
                        sell(k, 1.0, price.get(k, p["entry"]), "time")
                        actions.append(f"بيع {k}: {PARAMS['time_stop_days']} يوماً بلا ربح يساوي المخاطرة")
            proposed = {k for k, _, _ in dec["entries"]}
            for k in dec.get("filtered", []):
                rejected.append(dict(coin=k, reasons=["ml_filter"]))
            for k in dec["watch"]:
                if k not in proposed and k not in st["pos"] and k not in dec.get("filtered", []):
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
                    px = mk.last_price(k, data[k]) if mk.SYN else mk.last_price(k, None)
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
                                         usd=round(invest, 2), qty=qty, stop=stop, t1=(px + PARTIAL_R * (px - stop)) if PARTIAL_R > 0 else None, t2=px + TARGET2_R * (px - stop),
                                         fee_usd=round(invest * FEE, 4), slip_usd=round(qty * px * slip, 4), risk_usd=round(risk_usd, 2), pos_id=pos_id))
                jrec.append(dict(t=stamp, type="entry", coin=k, regime=dec["regime"], budget=dec["budget"], price=px, stop=stop, usd=round(invest, 2),
                                 asked_weight=w, risk_usd=round(risk_usd, 2), features=feats, edge=edge_mean, edge_n=edge_n, pos_id=pos_id,
                                 approved_by="risk_engine", mode=cfg["mode"]))
                actions.append(f"شراء {k} بـ ${invest:,.0f} ({invest/equity*100:.1f}% من الحساب، مخاطرة ${risk_usd:,.0f} حتى الوقف، "
                               f"{(BRK_TXT if snaps[k]['breakout'] else 'عودة فوق المتوسط 20 بعد تصحيح')})")
                orders.append(shadow_order(k, px, invest / equity, stop, rule))
                print(f"  G BUY {k} ${invest:,.2f} @ {px:g} stop {stop:g}")
            # ---- pyramiding (if adopted): one more unit on a position pyramid_r x R in profit, sized and approved by the
            #      risk engine on its own stop; the stop of the whole position then rises to that unit's stop
            if PARAMS["pyramid"] and allow and edge_ok and dec["budget"] > 0:
                for k in list(st["pos"]):
                    p, s_ = st["pos"][k], snaps.get(k)
                    rule = rules.get(k)
                    if p.get("added") or not s_ or not rule or p["entry_ms"] >= now_ms - DAY or not s_["atr"] or not s_["vol"]:
                        continue
                    px = price.get(k, s_["c"])
                    if px < p["entry"] + PARAMS["pyramid_r"] * p["r"]:
                        continue
                    room = dec["budget"] * equity - sum(held_vals.values())
                    w = min(dec["budget"] / PARAMS["slot"], PARAMS["vol_unit"] / s_["vol"], max(0.0, room) / equity)
                    new_stop = float(f"{round_down(px - INIT_ATR * s_['atr'], rule['tick']):.10g}")
                    slip = measured_slip(k)
                    ctx = dict(equity=equity, cash=st["cash"], budget=dec["budget"], held_values=held_vals, n_positions=max(0, len(st["pos"]) - 1),
                               rets=rets, cooldowns={}, now_ms=now_ms)
                    usd, why_not = risk.size_entry(cfg, dict(coin=k, price=px * (1 + slip), stop=new_stop, weight=w, min_qty=rule["min"], lot=rule["lot"]), ctx)
                    if usd <= 0:
                        continue
                    fill = px * (1 + slip)
                    qty = round_down(min(usd, st["cash"]) * (1 - FEE) / fill, rule["lot"])
                    invest = qty * fill / (1 - FEE)
                    if qty < rule["min"] or invest < cfg["min_order_usd"]:
                        continue
                    st["cash"] -= invest
                    p["qty"] += qty
                    p["basis"] += invest
                    p["added"] = True
                    p["stop"] = max(p["stop"], new_stop)
                    held_vals[k] = held_vals.get(k, 0) + invest
                    st["trades"].append(dict(time=stamp, coin=k, side="BUY", add=True, price=px, usd=round(invest, 2), qty=qty, stop=p["stop"],
                                             regime=dec["regime"], fee_usd=round(invest * FEE, 4), slip_usd=round(qty * px * slip, 4), pos_id=p.get("pos_id")))
                    jrec.append(dict(t=stamp, type="add", coin=k, price=px, usd=round(invest, 2), stop=p["stop"], pos_id=p.get("pos_id"),
                                     approved_by="risk_engine", mode=cfg["mode"]))
                    actions.append(f"زيادة {k} بـ ${invest:,.0f}: الصفقة رابحة بقدر المخاطرة، والوقف ارتفع إلى {p['stop']:.6g}")
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
    st["holdings"] = {k: dict(qty=p["qty"], basis=round(p["basis"], 2), t1=(p["entry"] + PARTIAL_R * p["r"]) if PARTIAL_R > 0 else None,
                              t2=p["entry"] + TARGET2_R * p["r"], added=p.get("added", False),
                              half_taken=p["half"], pnl_usd=round(p["qty"] * price.get(k, p["entry"]) * (1 - FEE) - p["basis"], 2),
                              risk_usd=round(p["qty"] * max(0.0, price.get(k, p["entry"]) - p["stop"]), 2),
                              value=p["qty"] * price.get(k, p["entry"]), entry=p["entry"], sl=p["stop"], tp=(p["entry"] + PARTIAL_R * p["r"]) if PARTIAL_R > 0 else None,
                              pnl=price.get(k, p["entry"]) / p["entry"] - 1, signal=p.get("regime", ""), protected=p["stop"] >= p["entry"]) for k, p in st["pos"].items()}
    sells = [t for t in st["trades"] if t["side"] == "SELL"]
    st["stats"] = dict(trades=len(sells), gate_trades=20, gate=f"مغلقة {len(sells)} · الحكم بعد 3 أشهر مقابل الاحتفاظ بـBTC والشراء الدوري")
    st["rules"] = dict(coins=COINS, agent=True, real_money_mode=True, account_dd_limit=cfg["max_drawdown"], watch_n=WATCH_N, max_pos=cfg["max_positions"],
                       vol_unit=VOL_UNIT, init_atr=INIT_ATR, trail_atr=TRAIL_ATR, partial_r=PARTIAL_R, risk=cfg, params=PARAMS)
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
    if "research" in sys.argv:
        research()
    elif "backtest" in sys.argv:
        backtest()
    else:
        cycle()
