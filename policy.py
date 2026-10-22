#!/usr/bin/env python3
"""Coin policy shared by every account: coins listed in coins_policy.json["excluded"] are never bought,
and any open position in them is closed at the next cycle (reason "policy"). Read-only helper."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "coins_policy.json")


def load():
    try:
        return json.load(open(PATH))
    except Exception:
        return {"source": "", "checked": "", "excluded": [], "reviewed": {}}


EXCLUDED = frozenset(c.upper() for c in load().get("excluded", []))


def allowed(coins):
    return [c for c in coins if c.upper() not in EXCLUDED]
