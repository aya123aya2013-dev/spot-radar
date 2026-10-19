#!/usr/bin/env python3
"""Public OKX vs Binance comparison (no keys, read-only, never trades).

For each coin: mid price on both exchanges, the gap in basis points, OKX bid/ask spread, and what a market
buy of 1000 USDT would cost on OKX's book beyond the mid price (compared with the 0.05% slippage the
models assume). A rolling history of the last 288 snapshots (about a day at one per 5 minutes) gives averages.
Writes lab_data/okx_check.json.
"""
import json, os, sys, time, urllib.request
from datetime import datetime, timezone

import spot_lab as sl

COINS = ["BTC", "ETH", "SOL", "OP", "POL"]
PATH = os.path.join(sl.OUT, "okx_check.json")
NOTIONAL = 1000.0
KEEP = 288


def okx_book(coin):
    req = urllib.request.Request(f"https://www.okx.com/api/v5/market/books?instId={coin}-USDT&sz=50",
                                 headers={"User-Agent": "spot-radar"})
    d = json.loads(urllib.request.urlopen(req, timeout=15).read())
    b = d["data"][0]
    return [(float(a[0]), float(a[1])) for a in b["asks"]], [(float(a[0]), float(a[1])) for a in b["bids"]]


def binance_book(coin):
    d = sl.api(f"/api/v3/depth?symbol={coin}USDT&limit=50")
    return [(float(p), float(q)) for p, q in d["asks"]], [(float(p), float(q)) for p, q in d["bids"]]


def mid_spread(asks, bids):
    mid = (asks[0][0] + bids[0][0]) / 2
    return mid, (asks[0][0] - bids[0][0]) / mid * 1e4


def buy_slip_bps(asks, mid, notional):
    left, cost, qty = notional, 0.0, 0.0
    for p, q in asks:
        take = min(left, p * q)
        cost += take
        qty += take / p
        left -= take
        if left <= 1e-9:
            break
    return ((cost / qty) / mid - 1) * 1e4 if qty and left <= 1e-9 else None


def main():
    hist = json.load(open(PATH)).get("history", {}) if os.path.exists(PATH) else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snap, errs = {}, []
    for c in COINS:
        try:
            oa, ob = okx_book(c)
            ba, bb = binance_book(c)
            om, osp = mid_spread(oa, ob)
            bm, bsp = mid_spread(ba, bb)
            row = dict(okx_mid=om, binance_mid=bm, gap_bps=(om / bm - 1) * 1e4, okx_spread_bps=osp,
                       binance_spread_bps=bsp, okx_slip_bps=buy_slip_bps(oa, om, NOTIONAL),
                       binance_slip_bps=buy_slip_bps(ba, bm, NOTIONAL))
            snap[c] = row
            h = hist.setdefault(c, [])
            h.append([row["gap_bps"], row["okx_spread_bps"], row["okx_slip_bps"]])
            hist[c] = h[-KEEP:]
        except Exception as e:
            errs.append(f"{c}: {type(e).__name__} {str(e)[:60]}")
    avg = {}
    for c, h in hist.items():
        def m(i):
            v = [x[i] for x in h if x[i] is not None]
            return sum(v) / len(v) if v else None
        avg[c] = dict(samples=len(h), gap_bps=m(0), gap_abs_bps=(sum(abs(x[0]) for x in h) / len(h)) if h else None,
                      okx_spread_bps=m(1), okx_slip_bps=m(2))
    json.dump(dict(time=now, notional=NOTIONAL, now=snap, avg=avg, errors=errs, history=hist,
                   model_slip_bps=sl.SLIP * 1e4, okx_taker_bps=10, okx_maker_bps=8), open(PATH, "w"))
    print(f"okx_check: {len(snap)}/{len(COINS)} coins" + (f"; errors {errs}" if errs else ""))
    for c, r in snap.items():
        print(f"  {c:4} gap {r['gap_bps']:+.1f} bps  OKX spread {r['okx_spread_bps']:.1f} bps  buy 1000$ slip {r['okx_slip_bps']}")


if __name__ == "__main__":
    main()
