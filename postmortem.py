#!/usr/bin/env python3
"""Post-mortem of every closed trade in accounts A and C. Read-only: it never changes an account.

For each closed trade it measures, from the real hourly candles:
  best price reached while holding (MFE), worst (MAE), how much of the peak profit was kept,
  and what the price did 24h and 72h after the exit.
Then it labels the trade (exited late, exited early, gave back a win, clean exit ...) and writes
lab_data/postmortems.json, which the dashboard shows in the journal tab.
"""
import json, os, sys, time
from datetime import datetime, timezone

import spot_lab as sl

OUT = sl.OUT
PATH = os.path.join(OUT, "postmortems.json")
ACCOUNTS = [("A", "paper.json"), ("C", "paper_C.json")]
HOUR = 3600 * 1000


def ts(s):
    return int(datetime.strptime(s[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def bars(coin, t0):
    """Hourly candles from t0 on (up to ~40 days)."""
    if sl.SYN:
        rows = sl.synthetic(coin, "1h", 1500)
        return [r for r in rows if r[0] >= t0]
    b = sl.api(f"/api/v3/klines?symbol={coin}USDT&interval=1h&startTime={t0}&limit=1000")
    return [[int(k[0]), *map(float, k[1:6])] for k in b]


def label(ret, mfe, eff, post24, why):
    """Plain-language lesson for one trade."""
    if ret > 0 and mfe > 0.03 and eff is not None and eff < 0.4:
        return "late", f"خرج متأخراً: بلغ الربح {mfe*100:.1f}% ثم لم يحتفظ إلا بـ {ret*100:.1f}%"
    if ret <= 0 and mfe > 0.02:
        return "gave_back", f"كان رابحاً {mfe*100:.1f}% ثم انقلب إلى {ret*100:.1f}%"
    if ret <= 0 and post24 is not None and post24 > 0.02:
        return "early", f"خرج مبكراً: ارتفع السعر {post24*100:.1f}% في 24 ساعة بعد الخروج"
    if ret > 0 and post24 is not None and post24 < -0.01:
        return "good_exit", f"خروج جيد: هبط السعر {abs(post24)*100:.1f}% بعده"
    if ret <= 0:
        return "plain_loss", f"خسارة عادية {ret*100:.1f}% دون إشارة على خطأ في التوقيت"
    return "plain_win", f"ربح عادي {ret*100:.1f}%"


def pair_trades(trades):
    open_, out = {}, []
    for t in trades:
        if t["side"] == "BUY":
            open_[t["coin"]] = t
        elif t["side"] == "SELL" and t["coin"] in open_:
            out.append((open_.pop(t["coin"]), t))
    return out


def main():
    data = json.load(open(PATH)) if os.path.exists(PATH) else dict(items=[], summary={})
    seen = {i["id"] for i in data["items"]}
    now_ms = int(time.time() * 1000)
    added = 0
    for acct, fn in ACCOUNTS:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        st = json.load(open(p))
        for buy, sell in pair_trades(st.get("trades", [])):
            pid = f"{acct}|{buy['coin']}|{buy['time']}"
            if pid in seen:
                continue
            t0, t1 = ts(buy["time"]), ts(sell["time"])
            try:
                rows = bars(buy["coin"], t0 - HOUR)
            except Exception as e:
                print("  postmortem skipped (no candles):", pid, str(e)[:60])
                continue
            held = [r for r in rows if t0 - HOUR <= r[0] <= t1]
            if not held:
                continue
            entry = buy["price"]
            mfe = max(r[2] for r in held) / entry - 1
            mae = min(r[3] for r in held) / entry - 1
            ret = sell.get("ret", 0.0)
            eff = (ret / mfe) if mfe > 0.005 else None

            def after(h):
                if t1 + h * HOUR > now_ms:
                    return None
                r = next((r for r in rows if r[0] >= t1 + h * HOUR - HOUR), None)
                return (r[4] / sell["price"] - 1) if r else None
            post24, post72 = after(24), after(72)
            kind, text = label(ret, mfe, eff, post24, sell.get("why"))
            data["items"].append(dict(id=pid, account=acct, coin=buy["coin"], bought=buy["time"], sold=sell["time"],
                                      ret=ret, mfe=mfe, mae=mae, efficiency=eff, post24=post24, post72=post72,
                                      why=sell.get("why"), kind=kind, lesson=text))
            added += 1
            print(f"  postmortem {pid}: {text}")
    data["items"] = data["items"][-200:]
    counts = {}
    for i in data["items"]:
        counts[i["kind"]] = counts.get(i["kind"], 0) + 1
    effs = [i["efficiency"] for i in data["items"] if i.get("efficiency") is not None and i["ret"] > 0]
    data["summary"] = dict(trades=len(data["items"]), counts=counts,
                           avg_efficiency=(sum(effs) / len(effs)) if effs else None,
                           updated=datetime.now(timezone.utc).isoformat())
    json.dump(data, open(PATH, "w"), indent=1)
    print(f"post-mortems: {added} new, {len(data['items'])} total")


if __name__ == "__main__":
    main()
