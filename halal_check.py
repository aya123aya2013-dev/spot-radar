#!/usr/bin/env python3
"""Daily copy of the Sharia rulings published by the owner's screening site (cryptohalal.cc, public API, no key).

Writes lab_data/halal_rulings.json: {symbol: {name, ruling}} for every coin the site lists.  The agent buys only a coin
whose ruling is exactly "مباح" (permissible); anything else (غير مباح, مشتبه به, missing) is never bought, and a position
in it is closed.  If the site cannot be read, the previous copy is kept (never replaced by an empty one).
Runs once per UTC day (skips if today's copy exists); --force refreshes anyway.
"""
import json, os, sys, time, urllib.request
from datetime import datetime, timezone

OUT = os.environ.get("SPOT_LAB_DATA") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_data")
PATH = os.path.join(OUT, "halal_rulings.json")
API = "https://api.cryptohalal.cc/api/coins?page={page}&limit={limit}"
UA = {"User-Agent": "Mozilla/5.0 (spot-radar agent; daily ruling check)", "Accept": "application/json"}
RULING_WORDS = ("مباح", "غير مباح", "مشتبه", "محظور", "جديد")


def get(url):
    for i in range(3):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
        except Exception as e:
            err = e
            time.sleep(3 * (i + 1))
    raise err


def ruling_of(item):
    """The field holding the ruling text (the API's field name is not documented, so it is found by its values)."""
    for k, v in item.items():
        if isinstance(v, str) and v.strip() in RULING_WORDS or (isinstance(v, str) and any(v.strip().startswith(w) for w in RULING_WORDS) and len(v) < 20):
            return k, v.strip()
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, str) and any(v2.strip().startswith(w) for w in RULING_WORDS) and len(v2) < 20:
                    return f"{k}.{k2}", v2.strip()
    return None, None


def main():
    now = datetime.now(timezone.utc)
    old = json.load(open(PATH)) if os.path.exists(PATH) else {}
    if old.get("date") == now.strftime("%Y-%m-%d") and "--force" not in sys.argv:
        print("halal rulings: today's copy exists")
        return
    rulings, fields, page, limit = {}, set(), 1, 100
    try:
        while page <= 60:
            d = get(API.format(page=page, limit=limit))
            items = ((d or {}).get("data") or {}).get("items") or []
            if page == 1 and items:
                print("item keys:", sorted(items[0].keys()))
                print("sample:", {k: v for k, v in items[0].items() if not isinstance(v, str) or len(v) < 60})
            for it in items:
                sym = str(it.get("symbol") or "").upper().strip()
                f, r = ruling_of(it)
                if f:
                    fields.add(f)
                if sym and r and sym not in rulings:          # the list is ordered by size: the first entry of a symbol wins
                    rulings[sym] = dict(name=it.get("name"), ruling=r, slug=it.get("slug"))
            if len(items) < limit:
                break
            page += 1
    except Exception as e:
        print("halal rulings: site not readable:", e)
    if len(rulings) < 20:
        print(f"halal rulings: only {len(rulings)} coins read; keeping the previous copy")
        return
    out = dict(date=now.strftime("%Y-%m-%d"), fetched=now.isoformat(timespec="seconds"), source="https://www.cryptohalal.cc/currencies",
               api="https://api.cryptohalal.cc/api/coins", ruling_field=sorted(fields), n=len(rulings), rulings=rulings)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(PATH, "w"), ensure_ascii=False, indent=1)
    print(f"halal rulings: {len(rulings)} coins, field {sorted(fields)}")


if __name__ == "__main__":
    main()
