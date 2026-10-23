#!/usr/bin/env python3
"""Monthly scorecard: every paper account since its start against the two simplest rulers, holding BTC and
buying BTC in equal weekly instalments, plus trade counts and win rates.  Read-only; writes lab_data/monthly.json
(a list, one entry per report, newest last).  Runs on the 17th of each month and on demand."""
import csv, json, os
from datetime import datetime, timezone

import spot_lab as sl

OUT = sl.OUT
CAPITAL = 5000.0
ACCOUNTS = [("A", "paper.json", "paper_equity.csv"), ("B", "paper_B.json", "paper_equity_B.csv"),
            ("C", "paper_C.json", "paper_equity_C.csv"), ("D", "paper_D.json", "paper_equity_D.csv")]


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def series(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return []
    rows = list(csv.reader(open(p)))[1:]
    out = []
    for r in rows:
        try:
            out.append((r[0], float(r[1])))
        except (ValueError, IndexError):
            pass
    return out


def equity(st):
    if "equity" in st:
        return st["equity"]
    px = st.get("prices") or {}
    return sum(st["cash"][c] + st["qty"][c] * px.get(c, 0) for c in st["cash"])


def ts(s):
    return datetime.strptime(s[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def btc_rulers(start, end, daily):
    """Hold vs weekly DCA into BTC between two UTC datetimes, from daily closes [(ms, close)]."""
    d = [(t, c) for t, c in daily if start.timestamp() * 1000 <= t <= end.timestamp() * 1000 + 86400000]
    if len(d) < 2:
        return None, None
    hold = d[-1][1] / d[0][1] - 1
    weeks = d[::7]
    qty = sum((1 / len(weeks)) / c for _, c in weeks)
    return hold, qty * d[-1][1] - 1


def main():
    now = datetime.now(timezone.utc)
    try:
        daily = [(r[0], r[4]) for r in sl.klines("BTC", "1d", 400)]
    except Exception as e:
        print("no BTC daily data:", e)
        daily = []
    accounts = {}
    for name, fn, cf in ACCOUNTS:
        st = load(fn)
        if not st:
            continue
        ser = series(cf)
        if not ser:
            continue
        start, end = ts(ser[0][0]), ts(ser[-1][0])
        eq = equity(st)
        pk, dd = 0.0, 0.0
        for _, v in ser:
            pk = max(pk, v)
            dd = max(dd, 1 - v / pk) if pk else 0
        sells = [t for t in st.get("trades", []) if t.get("side") == "SELL" and t.get("ret") is not None]
        wins = [t for t in sells if t["ret"] > 0]
        hold, dca = btc_rulers(start, end, daily)
        accounts[name] = dict(start=ser[0][0], days=round((end - start).total_seconds() / 86400, 1), equity=eq,
                              ret=eq / CAPITAL - 1, max_dd=dd, closed=len(sells),
                              win_rate=(len(wins) / len(sells)) if sells else None,
                              avg_trade=(sum(t["ret"] for t in sells) / len(sells)) if sells else None,
                              btc_hold=hold, btc_dca=dca)
    entry = dict(generated=now.isoformat(), month=now.strftime("%Y-%m"), accounts=accounts,
                 note="Paper accounts only. Rulers: holding BTC and weekly DCA into BTC over the same days. No profit is promised.")
    path = os.path.join(OUT, "monthly.json")
    hist = json.load(open(path)) if os.path.exists(path) else []
    hist = [h for h in hist if h.get("month") != entry["month"]] + [entry]
    os.makedirs(OUT, exist_ok=True)
    json.dump(hist[-24:], open(path, "w"), indent=1)
    for k, a in accounts.items():
        print(f"{k}: ${a['equity']:,.2f} ({a['ret']*100:+.2f}%) dd {a['max_dd']*100:.1f}% closed {a['closed']} "
              f"| BTC hold {a['btc_hold']*100 if a['btc_hold'] is not None else float('nan'):+.1f}% dca {a['btc_dca']*100 if a['btc_dca'] is not None else float('nan'):+.1f}%")


if __name__ == "__main__":
    main()
