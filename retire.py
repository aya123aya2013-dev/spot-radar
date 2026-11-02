#!/usr/bin/env python3
"""Retire paper accounts the owner decided to stop: close every open position at the current price, mark the state
retired, keep the history as a record.  Idempotent: an account already retired is left untouched.
Usage: python retire.py A B C D E F"""
import json, os, sys
from datetime import datetime

import spot_lab as sl

FILES = {"A": "paper.json", "B": "paper_B.json", "C": "paper_C.json", "D": "paper_D.json", "E": "paper_E.json", "F": "paper_F.json"}
REASONS = {
    "D": "أوقفه المالك في 2 نوفمبر 2026: الضربات السريعة خاسرة على التاريخ وحياً (0 رابحة من 15)، واستُبدل بالوكيل G.",
    "E": "أوقفه المالك في 2 نوفمبر 2026: الضربات السريعة خاسرة على التاريخ وحياً (0 رابحة من 15)، واستُبدل بالوكيل G.",
}
DEFAULT = "حوّل المالك النظام كله إلى الوكيل G في 2 نوفمبر 2026. هذا الحساب محفوظ كسجل."


def px_of(c, fallback):
    try:
        return sl.last_price(c, sl.klines(c, "1h", 3, closed_only=False) if sl.SYN else None)
    except Exception:
        return fallback


def main():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for acct in [a for a in sys.argv[1:] if a in FILES]:
        p = os.path.join(sl.OUT, FILES[acct])
        if not os.path.exists(p):
            continue
        st = json.load(open(p))
        if st.get("retired"):
            continue
        fee, slip = sl.FEE, sl.SLIP
        if isinstance(st.get("qty"), dict):                       # accounts A and C: per-coin cash and quantity
            for c, q in st["qty"].items():
                if q > 0:
                    px = px_of(c, (st.get("prices") or {}).get(c, 0))
                    proceeds = q * px * (1 - slip) * (1 - fee)
                    basis = st.get("basis", {}).get(c) or proceeds
                    st["cash"][c] += proceeds
                    st["qty"][c] = 0.0
                    st["trades"].append(dict(time=stamp, coin=c, side="SELL", price=px, ret=proceeds / basis - 1, why="retired"))
            eq = sum(st["cash"].values())
        elif isinstance(st.get("hold"), dict):                    # account B
            for c, q in list(st["hold"].items()):
                px = px_of(c, 0)
                st["cash"] += q * px * (1 - slip) * (1 - fee)
                st["trades"].append(dict(time=stamp, coin=c, side="SELL", price=px, why="retired"))
                st["hold"].pop(c)
            eq = st["cash"]
        else:                                                     # D, E, F: positions with basis
            for c, pos in list(st.get("pos", {}).items()):
                px = px_of(c, pos["entry"])
                proceeds = pos["qty"] * px * (1 - slip) * (1 - fee)
                st["cash"] += proceeds
                st["trades"].append(dict(time=stamp, coin=c, side="SELL", price=px, ret=proceeds / pos["basis"] - 1, why="retired",
                                         signal=pos.get("signal"), hours=0))
                st["pos"].pop(c)
            eq = st["cash"]
        st["holdings"], st["equity"] = {}, eq
        reason = REASONS.get(acct, DEFAULT)
        st["retired"] = dict(time=stamp, reason=reason, equity=eq)
        st.setdefault("events", []).append(dict(time=stamp, text=f"{acct}: {reason}"))
        json.dump(st, open(p, "w"), indent=1, ensure_ascii=False)
        print(f"{acct} retired at ${eq:,.2f}")


if __name__ == "__main__":
    main()
