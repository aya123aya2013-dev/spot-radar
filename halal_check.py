#!/usr/bin/env python3
"""Daily copy of the Sharia rulings published by the owner's screening site (cryptohalal.cc, public API, no key).

Writes lab_data/halal_rulings.json: {symbol: {name, ruling}} for every coin the site lists.  The agent buys only a coin
whose ruling is "مباح" (permissible, code 0 in the site's API); anything else (not permissible, suspicious, new,
missing) is never bought, and a position in it is closed.  If the site cannot be read, the previous copy is kept (never replaced by an empty one).
Runs once per UTC day (skips if today's copy exists); --force refreshes anyway.
"""
import json, os, sys, time, urllib.request
from datetime import datetime, timezone

OUT = os.environ.get("SPOT_LAB_DATA") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_data")
PATH = os.path.join(OUT, "halal_rulings.json")
API = "https://api.cryptohalal.cc/api/coins?page={page}&limit={limit}"
UA = {"User-Agent": "Mozilla/5.0 (spot-radar agent; daily ruling check)", "Accept": "application/json"}
# The API gives the ruling as a number ("judgement").  0 is permissible: the site shows BTC, ETH, XRP, SOL, USDT and USDC
# (all 0) as "مباح" and BNB (not 0) as "غير مباح".  Only 0 is ever bought; every other value is treated as not
# permissible, whatever it stands for (forbidden, suspicious, new, unknown), so a new code can never open a trade.
PERMISSIBLE = 0
KNOWN = {0: "مباح", 1: "غير مباح"}      # 1 is BNB's value, shown on the site as غير مباح
ANCHORS = {"BTC": 0, "ETH": 0, "XRP": 0, "SOL": 0}      # sanity check of the meaning of 0 on every run


def get(url):
    err = None
    for i in range(3):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        except Exception as e:
            err = e
            time.sleep(3 * (i + 1))
    raise err


def main():
    now = datetime.now(timezone.utc)
    old = json.load(open(PATH)) if os.path.exists(PATH) else {}
    if old.get("date") == now.strftime("%Y-%m-%d") and "--force" not in sys.argv:
        print("halal rulings: today's copy exists")
        return
    rulings, page, limit = {}, 1, 100
    try:
        while page <= 60:
            d = get(API.format(page=page, limit=limit))
            items = ((d or {}).get("data") or {}).get("items") or []
            for it in items:
                sym = str(it.get("symbol") or "").upper().strip()
                try:
                    j = int(it.get("judgement"))
                except (TypeError, ValueError):
                    j = None
                if sym and sym not in rulings:              # ordered by size: the first entry of a symbol wins
                    rulings[sym] = dict(name=it.get("name"), judgement=j, permissible=(j == PERMISSIBLE),
                                        ruling=KNOWN.get(j, "غير محسوم"), note=(it.get("judgnote") or it.get("note") or "")[:300],
                                        rank=it.get("rank"), slug=it.get("slug"))
            if len(items) < limit:
                break
            page += 1
    except Exception as e:
        print("halal rulings: site not readable:", e)
    bad = [s_ for s_, j in ANCHORS.items() if (rulings.get(s_) or {}).get("judgement") != j]
    if len(rulings) < 20 or bad or (rulings.get("BNB") or {}).get("permissible"):
        print(f"halal rulings: {len(rulings)} coins read, anchor mismatch {bad}; keeping the previous copy")
        return
    dist = {}
    for s_, r in rulings.items():
        dist.setdefault(r["judgement"], []).append(s_)
    print("judgement values:", {k: (len(v), v[:12]) for k, v in sorted(dist.items(), key=lambda x: str(x[0]))})
    out = dict(date=now.strftime("%Y-%m-%d"), fetched=now.isoformat(timespec="seconds"), source="https://www.cryptohalal.cc/currencies",
               api="https://api.cryptohalal.cc/api/coins", permissible_code=PERMISSIBLE, n=len(rulings), rulings=rulings)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(PATH, "w"), ensure_ascii=False, indent=1)
    print(f"halal rulings: {len(rulings)} coins, {sum(r['permissible'] for r in rulings.values())} permissible")


if __name__ == "__main__":
    main()
