#!/usr/bin/env python3
"""Do OKX candles match the Binance candles our tests use? Public data, read-only, never trades.

For the dashboard coins and the 1h / 4h / 1d charts it compares the last ~300 closed candles:
  median and worst close difference (basis points), correlation of candle returns,
  and how often a simple EMA 12/26 trend signal agrees between the two sources.
Writes lab_data/okx_candles.json. If the agreement is high, results measured on Binance candles carry over to OKX.
"""
import json, math, os, urllib.request
from datetime import datetime, timezone

import spot_lab as sl

COINS = ["BTC", "ETH", "SOL", "OP", "POL"]
BAR = {"1h": "1H", "4h": "4H", "1d": "1Dutc"}
N = 300


def okx_klines(coin, tf, n=N):
    out, after = [], ""
    while len(out) < n + 5:
        url = f"https://www.okx.com/api/v5/market/history-candles?instId={coin}-USDT&bar={BAR[tf]}&limit=100{after}"
        req = urllib.request.Request(url, headers={"User-Agent": "spot-radar"})
        d = json.loads(urllib.request.urlopen(req, timeout=20).read()).get("data") or []
        if not d:
            break
        out += [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in d if r[8] == "1"]
        after = f"&after={d[-1][0]}"
    out.sort()
    return out[-n:]


def corr(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb) if va and vb else None


def main():
    out, errs = {}, []
    for coin in COINS:
        for tf in BAR:
            try:
                ok = {r[0]: r for r in okx_klines(coin, tf)}
                bn = {r[0]: r for r in sl.klines(coin, tf)}
                ts = sorted(set(ok) & set(bn))
                if len(ts) < 50:
                    errs.append(f"{coin} {tf}: only {len(ts)} common candles")
                    continue
                diffs = [abs(ok[t][4] / bn[t][4] - 1) * 1e4 for t in ts]
                ro = [ok[b][4] / ok[a][4] - 1 for a, b in zip(ts, ts[1:])]
                rb = [bn[b][4] / bn[a][4] - 1 for a, b in zip(ts, ts[1:])]
                co, cb = [ok[t][4] for t in ts], [bn[t][4] for t in ts]
                eo = (sl.ema(co, 12), sl.ema(co, 26))
                ebn = (sl.ema(cb, 12), sl.ema(cb, 26))
                agree = sum((eo[0][i] > eo[1][i]) == (ebn[0][i] > ebn[1][i]) for i in range(len(ts))) / len(ts)
                out[f"{coin}|{tf}"] = dict(candles=len(ts), median_bps=sorted(diffs)[len(diffs) // 2], worst_bps=max(diffs),
                                           ret_corr=corr(ro, rb), signal_agree=agree)
            except Exception as e:
                errs.append(f"{coin} {tf}: {type(e).__name__} {str(e)[:60]}")
    json.dump(dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"), pairs=out, errors=errs),
              open(os.path.join(sl.OUT, "okx_candles.json"), "w"))
    print(f"okx_candles: {len(out)} charts compared" + (f"; errors {errs}" if errs else ""))
    for k, v in out.items():
        print(f"  {k:8} median {v['median_bps']:.1f} bps  worst {v['worst_bps']:.1f}  corr {v['ret_corr']:.4f}  signal agree {v['signal_agree']*100:.1f}%")


if __name__ == "__main__":
    main()
