#!/usr/bin/env python3
"""Telegram alerts for the agent (G).  Read-only: it never changes the account.

Needs two GitHub secrets (repository Settings > Secrets and variables > Actions):
  TELEGRAM_BOT_TOKEN   from @BotFather
  TELEGRAM_CHAT_ID     your chat id
Without them this script prints a note and does nothing.

Sends: every buy and sell with dollar amounts, targets and stop; every decision of the agent; one morning summary
(06:00 UTC); and the evening lesson (20:00 UTC).  Flag --dry prints the messages instead of sending them.
"""
import json, os, sys, urllib.parse, urllib.request
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SPOT_LAB_DATA") or os.path.join(HERE, "lab_data")
TOKEN, CHAT = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
DRY = "--dry" in sys.argv
STATE = os.path.join(OUT, "notify_state.json")
CAPITAL = 5000.0
ACCOUNTS = [("G", "paper_G.json")]
SUMMARY_HOUR_UTC = 6
LESSON_HOUR_UTC = 20        # evening lesson, about 22:00-23:00 Jerusalem


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def equity(st):
    return st.get("equity", CAPITAL)


LAST_ERR = [""]


def status(msg):
    """Tiny diagnostic file (never contains the token or chat id)."""
    try:
        json.dump({"time": datetime.now(timezone.utc).isoformat(), "status": msg}, open(os.path.join(OUT, "notify_status.json"), "w"))
    except Exception:
        pass


def send(text):
    if DRY:
        print("DRY>", text.replace("\n", " | "))
        return True
    data = urllib.parse.urlencode({"chat_id": CHAT, "text": text}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data, timeout=20).read()
        return True
    except Exception as e:
        LAST_ERR[0] = type(e).__name__ + " " + str(getattr(e, "code", "")) + " " + str(getattr(e, "reason", ""))[:60]
        try:
            body = json.loads(e.read().decode())          # Telegram explains itself: "chat not found", "Unauthorized"
            LAST_ERR[0] += " " + str(body.get("description", ""))[:80]
        except Exception:
            pass
        print("telegram send failed:", LAST_ERR[0])
        return False


WHY_AR = {"stop": "وقف", "partial": "جني النصف", "budget": "خفض الميزانية", "risk_trim": "تقليص حسب حد المخاطرة",
          "risk_limit": "قاطع الهبوط", "policy": "سياسة العملات"}
REG_AR = {"bull": "صاعد", "early": "بداية صعود", "turn": "محاولة انعطاف", "bear": "هابط", "weak": "ضعيف بلا اتجاه"}


def money(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def morning_summary(now):
    """One message every morning: the account, open positions in dollars, the last 24h, today's market reading."""
    st = load("paper_G.json")
    if not st:
        return None
    cutoff = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")
    v = equity(st)
    acc = st.get("accounting") or {}
    msg = f"☀️ صباح الخير · الوكيل · {now.strftime('%d/%m')}\n"
    msg += f"الحساب: ${v:,.2f} ({money(v - CAPITAL)}، {pct(v / CAPITAL - 1)})"
    if acc.get("cash") is not None:
        msg += f" · نقد ${acc['cash']:,.2f}"
    if st.get("regime"):
        msg += (f"\n🧭 السوق {REG_AR.get(st['regime'], st['regime'])} · ميزانية المخاطرة {st.get('budget', 0)*100:.0f}%"
                f" · القادة: {'، '.join((st.get('watch') or [])[:5]) or '—'}")
    hs = st.get("holdings") or {}
    if hs:
        msg += "\n\nالصفقات المفتوحة:"
        for c, h in hs.items():
            line = f"\n{c}: مستثمر ${h.get('basis', 0):,.2f}، الآن ${h.get('value', 0):,.2f} ({money(h.get('pnl_usd', 0))})"
            if h.get("sl"):
                line += f" · وقف {h['sl']:.6g}"
            if h.get("t1") or h.get("tp"):
                line += f" · هدف {(h.get('t1') or h.get('tp')):.6g}"
            msg += line
    else:
        msg += "\n\nلا صفقات مفتوحة: الحساب نقد."
    opened, closed = [], []
    for t in st.get("trades", []):
        if str(t.get("time", "")) < cutoff:
            continue
        if t.get("side") == "BUY":
            opened.append(f"{t.get('coin')} بـ ${t.get('usd', 0):,.2f} @ {(t.get('price') or 0):.6g}")
        elif t.get("side") == "SELL":
            why = WHY_AR.get(t.get("why"), t.get("why") or "")
            pl = f" {money(t['pnl_usd'])}" if t.get("pnl_usd") is not None else ""
            closed.append(f"{t.get('coin')}{pl} ({pct(t.get('ret') or 0)}، {why})")
    decided = [e.get("text", "") for e in st.get("events", []) if str(e.get("time", "")) >= cutoff]
    if opened:
        msg += "\n\nشراء خلال 24 ساعة:\n" + "\n".join(opened[:8])
    if closed:
        msg += "\n\nبيع خلال 24 ساعة:\n" + "\n".join(closed[:8])
    if decided:
        msg += "\n\nقرارات:\n" + "\n".join(d.replace("G: ", "")[:140] for d in decided[-5:])
    if not (opened or closed or decided):
        msg += "\n\nلا صفقات ولا قرارات خلال 24 ساعة."
    msg += "\n\nحساب ورقي، لا مال حقيقي."
    return msg


def pct(x):
    return f"{x*100:+.2f}%"


def main():
    if not DRY and not (TOKEN and CHAT):
        miss = [n for n, v in (("TELEGRAM_BOT_TOKEN", TOKEN), ("TELEGRAM_CHAT_ID", CHAT)) if not v]
        print("telegram not configured, missing:", miss)
        status("not configured, missing secret(s): " + ", ".join(miss))
        return
    if os.environ.get("NOTIFY_TEST") == "true":
        ok = send("🧪 اختبار تنبيهات الوكيل: إذا وصلتك هذه الرسالة فالتوكن الجديد يعمل.")
        status("test message sent" if ok else "test failed: " + LAST_ERR[0])
        return
    ns = json.load(open(STATE)) if os.path.exists(STATE) else None
    first = ns is None
    ns = ns or {"sent": [], "summary_day": "", "lesson_day": ""}
    sent = set(ns["sent"])
    msgs, keys = [], []
    for name, fn in ACCOUNTS:
        st = load(fn)
        if not st:
            continue
        for t in st.get("trades", []):
            k = f"{name}|{t.get('time')}|{t.get('coin') or ''}|{t.get('side')}"
            if k in sent:
                continue
            keys.append(k)
            if t["side"] == "BUY":
                amt = f"\nالمبلغ ${t['usd']:,.2f} · الكمية {t['qty']:g}" if t.get("usd") else ""
                if t.get("t1"):
                    p0 = t["price"]
                    amt += (f"\nهدف البيع 1: {t['t1']:.6g} (+{(t['t1']/p0-1)*100:.1f}%، بيع النصف)"
                            f"\nهدف البيع 2: {t['t2']:.6g} (+{(t['t2']/p0-1)*100:.1f}%، الباقي بوقف متحرك)"
                            f"\nالوقف: {t['stop']:.6g} (−{(1-t['stop']/p0)*100:.1f}%)")
                msgs.append(f"🟢 شراء {t.get('coin')} · الوكيل\nالسعر {(t.get('price') or 0):g}{amt}")
            elif t["side"] == "SELL":
                r = t.get("ret", 0)
                amt = f"\nدخلت بـ ${t['cost_usd']:,.2f} وخرجت بـ ${t['usd']:,.2f} · {'ربح' if t['pnl_usd'] >= 0 else 'خسارة'} ${abs(t['pnl_usd']):,.2f}" if t.get("pnl_usd") is not None else ""
                msgs.append(f"{'✅' if r >= 0 else '🔻'} بيع {t.get('coin')} · الوكيل · {WHY_AR.get(t.get('why'), t.get('why') or '')}\nالنتيجة {pct(r)} · السعر {(t.get('price') or 0):g} · التنفيذ {t.get('time')} UTC{amt}")
        for e in st.get("events", []):
            k = f"{name}|{e.get('time')}|{e.get('text', '')[:40]}"
            if k not in sent:
                keys.append(k)
                msgs.append(f"⚙️ الوكيل: {e.get('text', '').replace('G: ', '', 1)}")
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    summary = None
    if now.hour >= SUMMARY_HOUR_UTC and ns["summary_day"] != today:
        summary = morning_summary(now)
    if first:
        # first run: remember what already exists so history is not replayed, and say hello
        ns["sent"] = list(sent | set(keys))[-800:]
        ns["summary_day"] = today
        if send("✅ تنبيهات الوكيل تعمل. ستصلك هنا صفقاته وقراراته وملخص يومي.\n" + (summary or "")):
            json.dump(ns, open(STATE, "w"))
            status("ok, hello message sent")
        else:
            status("send failed: " + LAST_ERR[0])
        return
    for m in msgs[:15]:
        send(m)
    if len(msgs) > 15:
        send(f"و{len(msgs) - 15} تنبيهات أخرى، راجع اللوحة.")
    if summary:
        send(summary)
        ns["summary_day"] = today
    if now.hour >= LESSON_HOUR_UTC and ns.get("lesson_day") != today:
        les = load("lessons.json")
        if les and les.get("daily"):
            d = les["daily"][-1]
            if d.get("date") == today:
                send("📚 درس اليوم\n" + d["text"][:3500])
                ns["lesson_day"] = today
    ns["sent"] = list(sent | set(keys))[-800:]
    if not DRY:
        json.dump(ns, open(STATE, "w"))
        status("last send failed: " + LAST_ERR[0] if LAST_ERR[0] else "ok, checked")


if __name__ == "__main__":
    main()
