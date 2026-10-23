#!/usr/bin/env python3
"""Daily lesson: what the closed trades of the quick-strike accounts (D and E) teach, measured, not guessed.

For every closed trade it reads the real hourly candles and measures:
  best / worst price while holding (MFE / MAE), what the price did 24h after the exit,
  and a plain label: stop hit then price recovered, stop hit and price kept falling, target hit then price ran on,
  early protected exit before the target, time exit ...
Then it aggregates ALL closed trades by signal, coin, exit reason and entry hour (UTC), and writes one Arabic lesson
for the day.  Output: lab_data/lessons.json (daily entries + aggregates + suggested hypotheses for the backlog).
Read-only.  It never changes any account; rule changes go through the backlog and the weekly test.
"""
import json, os
from datetime import datetime, timezone, timedelta

import spot_lab as sl

OUT = sl.OUT
PATH = os.path.join(OUT, "lessons.json")
ACCOUNTS = [("D", "paper_D.json"), ("E", "paper_E.json")]
HOUR = 3600 * 1000
MIN_CASES = 5          # no pattern is called a lesson before this many cases
WHY_AR = {"stop": "وقف", "target": "هدف", "time": "وقت", "protect": "حماية ربح", "locked": "قفل 2%+", "halt": "قاطع", "policy": "سياسة"}
LABEL_AR = {
    "stop_recovered": "ضُرب الوقف ثم ارتد السعر فوق الدخول خلال 24 ساعة",
    "stop_continued": "ضُرب الوقف واستمر الهبوط",
    "target_ran": "تحقق الهدف واستمر الصعود أكثر من 2% إضافية",
    "target_clean": "تحقق الهدف ثم تراجع السعر: خروج جيد",
    "protect_early": "حماية الربح أخرجت الصفقة ثم بلغ السعر الهدف",
    "protect_good": "حماية الربح أنقذت ربحاً لم يبلغ الهدف",
    "locked_ran": "خرج عند قفل 2%+ ثم صعد السعر 2% إضافية على الأقل",
    "locked_good": "خرج عند قفل 2%+ ولم يصعد السعر أكثر: خروج جيد",
    "time_up": "انتهى الوقت ثم صعد السعر",
    "time_down": "انتهى الوقت ثم هبط السعر",
    "other": "أخرى",
}


def ts(s):
    return int(datetime.strptime(str(s)[:16], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def bars(coin, t0, n=120):
    if sl.SYN:
        rows = sl.synthetic(coin, "1h", 1500)
        return [r for r in rows if r[0] >= t0][:n]
    b = sl.api(f"/api/v3/klines?symbol={coin}USDT&interval=1h&startTime={t0}&limit={n}")
    return [[int(k[0]), *map(float, k[1:6])] for k in b]


def pair(trades):
    """(buy, sell) pairs per coin, in order."""
    open_, out = {}, []
    for t in trades:
        if t.get("side") == "BUY":
            open_[t["coin"]] = t
        elif t.get("side") == "SELL" and t["coin"] in open_:
            out.append((open_.pop(t["coin"]), t))
    return out


def label(sell, mfe, post_max, post_min, tp):
    why, ret = sell.get("why"), sell.get("ret", 0)
    if why == "stop":
        return "stop_recovered" if post_max >= 0.0 else "stop_continued"
    if why == "target":
        return "target_ran" if post_max >= tp else "target_clean"
    if why == "protect":
        return "protect_early" if post_max >= tp else "protect_good"
    if why == "locked":
        return "locked_ran" if post_max >= ret + 0.02 else "locked_good"
    if why == "time":
        return "time_up" if post_max > 0.01 and post_max > -post_min else "time_down"
    return "other"


def analyze(acct, st, cache):
    rules = st.get("rules") or {}
    tp = rules.get("tp", 0.02)
    out = []
    for buy, sell in pair(st.get("trades", [])):
        key = f"{acct}|{sell['coin']}|{sell['time']}"
        if key in cache:
            out.append(cache[key])
            continue
        try:
            t0, t1 = ts(buy["time"]), ts(sell["time"])
            rows = bars(sell["coin"], t0 - HOUR)
            held = [r for r in rows if t0 <= r[0] <= t1]
            after = [r for r in rows if t1 < r[0] <= t1 + 24 * HOUR]
            entry, exit_px = buy["price"], sell["price"]
            mfe = (max(r[2] for r in held) / entry - 1) if held else 0.0
            mae = (min(r[3] for r in held) / entry - 1) if held else 0.0
            if len(after) < 3:
                continue                                    # too fresh to judge: wait for the next run
            post_max = max(r[2] for r in after) / entry - 1    # relative to ENTRY: did the idea work later?
            post_min = min(r[3] for r in after) / entry - 1
            lab = label(sell, mfe, post_max, post_min, tp)
            item = dict(key=key, account=acct, coin=sell["coin"], entry_time=buy["time"], exit_time=sell["time"],
                        entry_hour=int(buy["time"][11:13]), signal=sell.get("signal") or buy.get("signal") or "-",
                        why=sell.get("why"), ret=sell.get("ret"), hours=sell.get("hours"), mfe=mfe, mae=mae,
                        post24_max=post_max, post24_min=post_min, label=lab)
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


def lesson_text(today_items, all_items):
    """One honest paragraph.  Patterns are only named with >= MIN_CASES cases."""
    lines, hyps = [], []
    if today_items:
        wins = [i for i in today_items if (i["ret"] or 0) > 0]
        lines.append(f"أُغلقت اليوم {len(today_items)} صفقة في D وE، منها {len(wins)} رابحة.")
        for i in today_items:
            lines.append(f"{i['account']} {i['coin']} {i['ret']*100:+.2f}% ({WHY_AR.get(i['why'], i['why'])}): {LABEL_AR.get(i['label'], i['label'])}.")
    else:
        lines.append("لم تُغلق صفقات اليوم في D وE.")
    n = len(all_items)
    if n >= MIN_CASES:
        labs = agg(all_items, "label")
        stops = [i for i in all_items if i["why"] == "stop"]
        rec = [i for i in stops if i["label"] == "stop_recovered"]
        if len(stops) >= MIN_CASES:
            share = len(rec) / len(stops)
            lines.append(f"من {len(stops)} وقفاً حتى الآن، ارتد السعر فوق الدخول خلال 24 ساعة بعد {len(rec)} منها ({share*100:.0f}%).")
            if share >= 0.5:
                hyps.append(dict(id="wider_stop_or_confirmation", title="الوقف يُضرب ثم يرتد السعر",
                                 idea=f"في {share*100:.0f}% من الوقفات ارتد السعر فوق الدخول خلال 24 ساعة. فرضية: وقف أوسع (4%) أو دخول بعد شمعة تأكيد. تُختبر على التاريخ مقابل الضوضاء قبل أي تطبيق.",
                                 source="درس يومي من الصفقات الحية", status="pending"))
        pe = [i for i in all_items if i["label"] == "protect_early"]
        pg = [i for i in all_items if i["label"] == "protect_good"]
        if len(pe) + len(pg) >= MIN_CASES:
            lines.append(f"حماية الربح: أنقذت ربحاً في {len(pg)} حالة وأخرجت مبكراً قبل الهدف في {len(pe)}.")
            if len(pe) > len(pg):
                hyps.append(dict(id="protect_later", title="حماية الربح تخرج مبكراً", idea="أكثر الخروجات المحمية سبقت بلوغ الهدف. فرضية: تفعيل الحماية بعد +1.5% بدل +1%.", source="درس يومي", status="pending"))
        for field, name in (("signal", "الإشارة"), ("coin", "العملة"), ("entry_hour", "ساعة الدخول (UTC)")):
            a = {k: v for k, v in agg(all_items, field).items() if v["n"] >= MIN_CASES}
            if a:
                best = max(a, key=lambda k: a[k]["mean"])
                worst = min(a, key=lambda k: a[k]["mean"])
                lines.append(f"{name}: الأفضل {best} (متوسط {a[best]['mean']*100:+.2f}% في {a[best]['n']})، الأسوأ {worst} ({a[worst]['mean']*100:+.2f}% في {a[worst]['n']}).")
        lines.append(f"المجموع: {n} صفقة مغلقة، نسبة النجاح {sum((i['ret'] or 0) > 0 for i in all_items)/n*100:.0f}%، متوسط الصفقة {sum(i['ret'] or 0 for i in all_items)/n*100:+.2f}%. التعادل يحتاج نحو 66%.")
    else:
        lines.append(f"المجموع {n} صفقة مغلقة فقط؛ لا أستخلص نمطاً قبل {MIN_CASES} حالات لكل نمط.")
    lines.append("القواعد لا تتغير على درس يوم واحد: الدروس تتجمع هنا، وما يتكرر يدخل قائمة الأفكار ويُختبر على التاريخ وعلى الضوضاء.")
    return "\n".join(lines), hyps


def main():
    now = datetime.now(timezone.utc)
    prev = json.load(open(PATH)) if os.path.exists(PATH) else {"daily": [], "items": {}}
    cache = {it["key"]: it for it in prev.get("items", {}).values()} if isinstance(prev.get("items"), dict) else {}
    items = []
    for acct, fn in ACCOUNTS:
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            items += analyze(acct, json.load(open(p)), cache)
    items.sort(key=lambda i: i["exit_time"])
    cutoff = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")
    today_items = [i for i in items if i["exit_time"] >= cutoff]
    text, hyps = lesson_text(today_items, items)
    day = now.strftime("%Y-%m-%d")
    daily = [d for d in prev.get("daily", []) if d.get("date") != day] + [dict(date=day, n_today=len(today_items), text=text)]
    out = dict(generated=now.isoformat(), items={i["key"]: i for i in items}, daily=daily[-60:],
               aggregates=dict(by_label=agg(items, "label"), by_signal=agg(items, "signal"), by_coin=agg(items, "coin"),
                               by_why=agg(items, "why"), by_hour=agg(items, "entry_hour"), n=len(items)),
               suggested=hyps, labels=LABEL_AR)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(PATH, "w"), indent=1, ensure_ascii=False)
    print(text)


if __name__ == "__main__":
    main()
