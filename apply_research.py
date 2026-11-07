#!/usr/bin/env python3
"""Apply the research round's verdict to the live strategy, mechanically.

Reads lab_data/research_G.json (written by `python agent.py research`).  If the pre-registered rule adopted a
candidate, this writes exactly that into strategy.json (strategy parameters) and control.json (only risk_per_trade and
max_positions, the two risk keys the rule may move), and records it in strategy.json's history, docs/DECISIONS.md,
changelog.json and candidates.json.  No judgement here: no adoption, no change.  Extra sanity checks refuse anything
outside the owner's limits (drawdown over 25%, a loss with costs x3, mode other than paper).
Run by the weekly backtest workflow, which then commits the change to main.
"""
import json, os, sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SPOT_LAB_DATA") or os.path.join(HERE, "lab_data")
PARAM_KEYS = {"watch_n", "max_pos", "slot", "vol_unit", "init_atr", "trail_atr", "partial_r", "target2_r", "breakout_n", "resume",
              "exit", "exit_low_n", "time_stop_days", "time_stop_r", "pyramid", "pyramid_r", "ml"}
RISK_KEYS = {"risk_per_trade", "max_positions"}


def load(name, base=HERE):
    return json.load(open(os.path.join(base, name)))


def save(name, obj):
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def main():
    try:
        r = load("research_G.json", OUT)
    except Exception as e:
        print("no research result:", e)
        return 0
    ad = r.get("adopted")
    if r.get("synthetic") or not ad:
        print("no adoption: the live strategy stays")
        return 0
    res = ad.get("result") or {}
    full = res.get("full") or {}
    if (full.get("max_dd") or 1) > 0.25 or (res.get("x3") or -1) <= 0:
        print("refused: the adopted result is outside the owner's limits", full.get("max_dd"), res.get("x3"))
        return 0
    st, ctl = load("strategy.json"), load("control.json")
    if ctl.get("mode") != "paper":
        print("refused: control.json is not in paper mode")
        return 0
    params = dict(st.get("params", {}), **{k: v for k, v in (ad.get("params") or {}).items() if k in PARAM_KEYS})
    rchg = {k: v for k, v in (ad.get("risk") or {}).items() if k in RISK_KEYS}
    if params == st.get("params", {}) and all(ctl["risk"].get(k) == v for k, v in rchg.items()):
        print("already applied")
        return 0
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    b = (r.get("baseline") or {}).get("result", {}).get("full", {})
    ev = (f"{r['period'][0]}..{r['period'][1]}: CAGR {(b.get('cagr') or 0)*100:.1f}% -> {(full.get('cagr') or 0)*100:.1f}%, "
          f"max drawdown {(b.get('max_dd') or 0)*100:.1f}% -> {full['max_dd']*100:.1f}%, Calmar {(b.get('calmar') or 0):.2f} -> {(full.get('calmar') or 0):.2f}, "
          f"costs x3 {res['x3']*100:+.1f}%")
    st["params"] = params
    st.setdefault("history", []).append(dict(date=day, change=dict(ad.get("params") or {}, **rchg), by=f"research rule, research_G.json {r.get('generated', '')[:10]}",
                                             evidence=ev))
    save("strategy.json", st)
    if rchg:
        ctl["risk"].update(rchg)
        ctl.setdefault("owner_decisions", []).append(dict(date=day, change=", ".join(f"{k} -> {v}" for k, v in rchg.items()),
                                                          authorized_by="pre-registered research rule (within the owner's 25% drawdown line)"))
        save("control.json", ctl)
    title = ((r.get("candidates") or {}).get(ad["name"]) or {}).get("title") or ad["name"]
    risk_txt = f"، والمخاطرة للصفقة صارت {rchg['risk_per_trade']*100:.2f}%" if "risk_per_trade" in rchg else ""
    ar = (f"جولة البحث اعتمدت «{title}»{risk_txt}. على الماضي: العائد السنوي {(b.get('cagr') or 0)*100:.1f}% ← {(full.get('cagr') or 0)*100:.1f}%، "
          f"وأسوأ هبوط {(b.get('max_dd') or 0)*100:.1f}% ← {full['max_dd']*100:.1f}%.")
    try:
        cl = load("changelog.json")
        cl.setdefault("items", []).insert(0, dict(date=day, text=ar))
        save("changelog.json", cl)
    except Exception as e:
        print("changelog:", e)
    try:
        cands = load("candidates.json")
        for it in cands.get("items", []):
            if it["id"] in ad["name"].split("+"):
                it["status"], it["adopted"] = "adopted", day
        save("candidates.json", cands)
    except Exception as e:
        print("candidates:", e)
    try:
        with open(os.path.join(HERE, "docs", "DECISIONS.md"), "a") as f:
            f.write(f"| {day} | {ar} | قاعدة البحث المكتوبة مسبقاً (تطبيق آلي) | {ev} |\n")
    except Exception as e:
        print("decisions:", e)
    print("applied:", ad["name"], params, rchg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
