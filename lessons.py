#!/usr/bin/env python3
"""Daily lesson from the agent's closed trades: measured from real candles, not guessed.

For every sell (a full exit or a partial one) it reads the candles and measures:
  best and worst price while holding (MFE / MAE, from daily candles),
  what the price did in the POST_H hours after the exit (hourly candles),
  and a plain label: stop hit then the price recovered above the entry, stop hit and the fall continued,
  exited and the price ran on, exited and the exit avoided a fall ...
Then it aggregates all closed fills by label, exit reason, coin and entry regime, and writes one Arabic lesson for
the day.  Output: lab_data/lessons.json.  Read-only: it never changes the account.  A rule changes only through a
challenger in agent.py that passes the pre-registered promotion rule.
"""
import json, os
from datetime import datetime, timezone, timedelta

import market as mk

OUT = mk.OUT
PATH = os.path.join(OUT, "lessons.json")
STATE = os.path.join(OUT, "paper_G.json")
HOUR = 3600 * 1000
DAY = 24 * HOUR
POST_H = 72            # the agent trades daily candles: judge an exit on the next 3 days, not the next hour
MOVE = 0.05            # a move of 5% after the exit counts as "the price ran on" / "the exit avoided a fall"
MIN_CASES = 5          # no pattern is called a lesson before this many cases
WHY_AR = {"stop": "وقف", "partial": "جني النصف", "risk_trim": "تقليص حسب حد المخاطرة", "risk_limit": "قاطع الهبوط",
          "budget": "خفض الميزانية", "policy": "سياسة العملات"}
LABEL_AR = {
    "stop_recovered": "ضُرب الوقف ثم عاد السعر فوق سعر الدخول خلال 3 أيام",
    "stop_continued": "ضُرب الوقف ولم يعد السعر إلى الدخول: الوقف حمى الحساب",
    "exit_early": "خرج ثم صعد السعر 5% أو أكثر",
    "exit_saved": "خرج ثم هبط السعر 5% أو أكثر: الخروج جنّب خسارة",
    "exit_neutral": "خرج ولم يتحرك السعر كثيراً بعده",
}


def ts(s):
    return int(datetime.strptime(str(s)[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def candles(coin, tf, t0, n):
    if mk.SYN:
        rows = mk.synthetic(coin, tf, 1500)
        return [r for r in rows if r[0] >= t0][:n]
    b = mk.api(f"/api/v3/klines?symbol={coin}USDT&interval={tf}&startTime={t0}&limit={n}")
    return [[int(k[0]), *map(float, k[1:6])] for k in b]


def fills(trades):
    """(buy, sell) for every sell, matched to its position's buy by pos_id (or the coin's last buy)."""
    by_pos, last_buy, out = {}, {}, []
    for t in trades:
        if t.get("side") == "BUY":
            last_buy[t["coin"]] = t
            if t.get("pos_id"):
                by_pos[t["pos_id"]] = t
        elif t.get("side") == "SELL":
            buy = by_pos.get(t.get("pos_id")) or last_buy.get(t["coin"])
            if buy:
                out.append((buy, t))
    return out


def label(why, entry, exit_px, post_max, post_min):
    if why == "stop":
        return "stop_recovered" if post_max >= entry else "stop_continued"
    if post_max >= exit_px * (1 + MOVE):
        return "exit_early"
    if post_min <= exit_px * (1 - MOVE):
        return "exit_saved"
    return "exit_neutral"


def analyze(st, cache, now_ms):
    out = []
    for buy, sell in fills(st.get("trades", [])):
        key = f"G|{sell['coin']}|{sell['time']}|{sell.get('why')}"
        if key in cache:
            out.append(cache[key])
            continue
        try:
            t0, t1 = ts(buy["time"]), ts(sell["time"])
            if now_ms - t1 < POST_H * HOUR:
                continue                                        # too fresh to judge: the next runs will do it
            held = [r for r in candles(sell["coin"], "1d", t0 - DAY, 1000) if t0 - DAY < r[0] <= t1]
            after = [r for r in candles(sell["coin"], "1h", t1, POST_H + 2) if t1 < r[0] <= t1 + POST_H * HOUR]
            if len(after) < POST_H // 2:
                continue
            entry, exit_px = buy["price"], sell["price"]
            mfe = (max(r[2] for r in held) / entry - 1) if held else 0.0
            mae = (min(r[3] for r in held) / entry - 1) if held else 0.0
            post_max, post_min = max(r[2] for r in after), min(r[3] for r in after)
            item = dict(key=key, account="G", coin=sell["coin"], entry_time=buy["time"], exit_time=sell["time"],
                        regime=buy.get("regime") or "-", why=sell.get("why"), ret=sell.get("ret"),
                        pnl_usd=sell.get("pnl_usd"), mfe=mfe, mae=mae,
                        post_max=post_max / exit_px - 1, post_min=post_min / exit_px - 1,
                        label=label(sell.get("why"), entry, exit_px, post_max, post_min))
            cache[key] = item
            out.append(item)
        except Exception as e:
            print(f"  ! {key}: {e}")
    return out


def agg(items, field):
    d = {}
    for it in items:
        d.setdefault(str(it.get(field)), []).append(it["ret"] or 0.0)
    return {k: dict(n=len(v), win=sum(x > 0 for x in v) / len(v), mean=sum(v) / len(v)) for k, v in d.items()}


def usd(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def lesson_text(today_items, all_items):
    """One honest paragraph.  Patterns are named only with >= MIN_CASES cases."""
    lines, hyps = [], []
    if today_items:
        lines.append(f"صفقات الوكيل التي اكتمل تقييمها اليوم: {len(today_items)}.")
        for i in today_items:
            money = f" ({usd(i['pnl_usd'])})" if i.get("pnl_usd") is not None else ""
            lines.append(f"{i['coin']} {i['ret']*100:+.2f}%{money}، {WHY_AR.get(i['why'], i['why'])}: {LABEL_AR.get(i['label'], i['label'])}.")
    else:
        lines.append("لا صفقات جديدة اكتمل تقييمها اليوم (أي خروج يُقيَّم بعد 3 أيام من حركة السعر).")
    n = len(all_items)
    if n >= MIN_CASES:
        stops = [i for i in all_items if i["why"] == "stop"]
        rec = [i for i in stops if i["label"] == "stop_recovered"]
        if len(stops) >= MIN_CASES:
            share = len(rec) / len(stops)
            lines.append(f"من {len(stops)} وقفاً، عاد السعر فوق الدخول بعد {len(rec)} منها ({share*100:.0f}%).")
            if share >= 0.5:
                hyps.append(dict(id="g_stop_confirmation", title="الوقف يُضرب ثم يعود السعر",
                                 idea=f"في {share*100:.0f}% من الوقفات عاد السعر فوق الدخول خلال 3 أيام. فرضية: وقف أوسع أو إغلاق يومي تحت الوقف بدل لمسه. تُختبر كمتحدٍّ في agent.py بقاعدة الترقية.",
                                 source="درس يومي من صفقات الوكيل", status="pending"))
        parts = [i for i in all_items if i["why"] == "partial"]
        early = [i for i in parts if i["label"] == "exit_early"]
        if len(parts) >= MIN_CASES:
            lines.append(f"جني النصف عند +2R: بعد {len(early)} من {len(parts)} حالة صعد السعر 5% أو أكثر.")
            if len(early) / len(parts) >= 0.6:
                hyps.append(dict(id="g_partial_later", title="جني النصف مبكر", idea="أغلب حالات جني النصف تبعها صعود 5%+. فرضية: جني النصف عند +3R. تُختبر كمتحدٍّ.",
                                 source="درس يومي", status="pending"))
        for field, name in (("why", "سبب الخروج"), ("regime", "حالة السوق عند الدخول"), ("coin", "العملة")):
            a = {k: v for k, v in agg(all_items, field).items() if v["n"] >= MIN_CASES}
            if len(a) >= 2:
                best = max(a, key=lambda k: a[k]["mean"])
                worst = min(a, key=lambda k: a[k]["mean"])
                nm = (lambda k: WHY_AR.get(k, k)) if field == "why" else (lambda k: k)
                lines.append(f"{name}: الأفضل {nm(best)} (متوسط {a[best]['mean']*100:+.2f}% في {a[best]['n']})، الأسوأ {nm(worst)} ({a[worst]['mean']*100:+.2f}% في {a[worst]['n']}).")
        tot = sum(i.get("pnl_usd") or 0 for i in all_items)
        lines.append(f"المجموع: {n} عملية خروج مقيَّمة، الرابحة {sum((i['ret'] or 0) > 0 for i in all_items)/n*100:.0f}%، صافي {usd(tot)}.")
    else:
        lines.append(f"المقيَّم حتى الآن {n} عملية خروج فقط؛ لا أستخلص نمطاً قبل {MIN_CASES} حالات.")
    lines.append("القواعد لا تتغير على درس يوم واحد: ما يتكرر يدخل قائمة الأفكار، ولا يُعتمد إلا كمتحدٍّ يتفوق على الوكيل الحالي بقاعدة الترقية المكتوبة مسبقاً.")
    return "\n".join(lines), hyps


def main():
    now = datetime.now(timezone.utc)
    prev = json.load(open(PATH)) if os.path.exists(PATH) else {}
    if prev.get("scope") != "G":
        prev = {}                                   # lessons of the deleted accounts are not carried over
    cache = dict(prev.get("items") or {})
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"trades": []}
    items = sorted(analyze(st, cache, now.timestamp() * 1000), key=lambda i: i["exit_time"])
    day = now.strftime("%Y-%m-%d")
    known = set((prev.get("items") or {}).keys())
    today_items = [i for i in items if i["key"] not in known] + [i for i in items if i["key"] in known and i.get("judged") == day]
    for i in items:
        i.setdefault("judged", day)
    text, hyps = lesson_text(today_items, items)
    daily = [d for d in prev.get("daily", []) if d.get("date") != day] + [dict(date=day, n_today=len(today_items), text=text)]
    out = dict(scope="G", generated=now.isoformat(), items={i["key"]: i for i in items}, daily=daily[-60:],
               aggregates=dict(by_label=agg(items, "label"), by_why=agg(items, "why"), by_coin=agg(items, "coin"),
                               by_regime=agg(items, "regime"), n=len(items)),
               suggested=hyps, labels=LABEL_AR, why=WHY_AR)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(PATH, "w"), indent=1, ensure_ascii=False)
    print(text)


if __name__ == "__main__":
    main()
