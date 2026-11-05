#!/usr/bin/env python3
"""Sharia coin policy (owner's rule: only coins the screening site cryptohalal.cc rules permissible, "مباح").

Two sources, the site first:
  1. lab_data/halal_rulings.json, refreshed daily from the site's public API by halal_check.py: a listed coin is
     tradable only if its ruling is مباح; any other ruling makes it untradable, whatever was recorded before.
  2. coins_policy.json["reviewed"]: rulings recorded by hand from the same site; used only for a coin the site's list
     does not include today.  A coin with no recorded مباح ruling anywhere is never traded.
coins_policy.json["excluded"] (religious) is always excluded.  An open position in an untradable coin is closed.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "coins_policy.json")
PERMISSIBLE = "مباح"


def load():
    try:
        return json.load(open(PATH))
    except Exception:
        return {"source": "", "checked": "", "excluded": [], "excluded_cost": [], "reviewed": {}}


_P = load()
EXCLUDED = frozenset(c.upper() for c in list(_P.get("excluded", [])) + list(_P.get("excluded_cost", [])))
RELIGIOUS = frozenset(c.upper() for c in _P.get("excluded", []))


def site_rulings(out_dir):
    try:
        return json.load(open(os.path.join(out_dir, "halal_rulings.json"))).get("rulings", {})
    except Exception:
        return {}


def ruling(coin, rulings):
    """(tradable, source) for one coin."""
    c = coin.upper()
    if c in RELIGIOUS:
        return False, "excluded"
    r = rulings.get(c)
    if r is not None:
        return bool(r.get("permissible")), "site"
    if str((_P.get("reviewed") or {}).get(c, "")).strip() == PERMISSIBLE:
        return True, "recorded"
    return False, "unreviewed"


def permitted(coins, rulings):
    return [c for c in coins if ruling(c, rulings)[0]]


def allowed(coins):
    return [c for c in coins if c.upper() not in EXCLUDED]


def policy_recorded(coin):
    """True if coins_policy.json records the coin as permissible."""
    return str((_P.get("reviewed") or {}).get(coin.upper(), "")).strip() == PERMISSIBLE
