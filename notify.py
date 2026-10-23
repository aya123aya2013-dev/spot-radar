#!/usr/bin/env python3
"""Telegram alerts for the paper accounts. Read-only: it never changes any account.

Needs two GitHub secrets (repository Settings > Secrets and variables > Actions):
  TELEGRAM_BOT_TOKEN   from @BotFather
  TELEGRAM_CHAT_ID     your chat id (send /start to the bot, then open
                       https://api.telegram.org/bot<TOKEN>/getUpdates in a browser and read "chat":{"id":...})
Without them this script prints a note and does nothing.

Sends: every new buy/sell, every system decision (strategy paused/switched, circuit breaker),
and one morning summary per day (06:00 UTC, about 08:00-09:00 Jerusalem): balances, open positions, last 24h.
Flag --dry prints the messages instead of sending them.
"""
import json, os, sys, urllib.parse, urllib.request
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SPOT_LAB_DATA") or os.path.join(HERE, "lab_data")
TOKEN, CHAT = os.environ.get("TELEGRAM_BOT_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", "")
DRY = "--dry" in sys.argv
STATE = os.path.join(OUT, "notify_state.json")
CAPITAL = 5000.0
ACCOUNTS = [("A", "paper.json"), ("B", "paper_B.json"), ("C", "paper_C.json"), ("D", "paper_D.json"), ("E", "paper_E.json")]
SUMMARY_HOUR_UTC = 6
LESSON_HOUR_UTC = 20        # evening lesson, about 22:00-23:00 Jerusalem


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def equity(st):
    if "equity" in st:                                  # account B stores it directly
        return st["equity"]
    px = st.get("prices") or {}
    return sum(st["cash"][c] + st["qty"][c] * px.get(c, 0) for c in st["cash"])


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


def holdings(name, st):
    """[(coin, pnl or None)] for the open positions of one account."""
    out = []
    px = st.get("prices") or {}
    if name in ("D", "E"):
        return [(c, h.get("pnl")) for c, h in (st.get("holdings") or {}).items()]
    if name == "B":
        return [(c, None) for c in (st.get("hold") or {})]
    for c, q in (st.get("qty") or {}).items():
        if q > 0:
            buy = next((t for t in reversed(st.get("trades", [])) if t.get("coin") == c and t.get("side") == "BUY"), None)
            out.append((c, (px[c] / buy["price"] - 1) if buy and px.get(c) else None))
    return out


def morning_summary(now):
    """One message every morning: balances, open positions, what closed and what was decided in the last 24h."""
    cutoff = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")
    lines, opened, closed, decided = [], [], [], []
    for name, fn in ACCOUNTS:
        st = load(fn)
        if not st:
            continue
        v = equity(st)
        hs = holdings(name, st)
        held = "، ".join(c + (f" {pct(p)}" if p is not None else "") for c, p in hs) if hs else "نقد"
        extra = ""
        if name in ("D", "E"):
            n_closed = (st.get("stats") or {}).get("trades", 0)
            extra = f" · بوابة {n_closed}/30"
        lines.append(f"{name}: ${v:,.2f} ({pct(v / CAPITAL - 1)}) · {held}{extra}")
        for t in st.get("trades", []):
            if str(t.get("time", "")) >= cutoff and t.get("side") == "SELL" and t.get("ret") is not None:
                why = {"stop": "وقف", "target": "هدف", "protect": "حماية ربح", "time": "وقت", "signal": "إشارة", "halt": "قاطع", "pause": "إيقاف", "policy": "سياسة"}.get(t.get("why"), t.get("why") or "")
                closed.append(f"{name} {t.get('coin')} {pct(t['ret'])}" + (f" ({why})" if why else ""))
            elif str(t.get("time", "")) >= cutoff and t.get("side") == "BUY":
                opened.append(f"{name} {t.get('coin')} @ {(t.get('price') or 0):g}")
        for e in st.get("events", []):
            if str(e.get("time", "")) >= cutoff:
                decided.append(e.get("text", ""))
    if not lines:
        return None
    msg = "☀️ صباح الخير · ملخص " + now.strftime("%d/%m") + "\n" + "\n".join(lines)
    if opened:
        msg += "\n\nدخول خلال 24 ساعة:\n" + "\n".join(opened[:8])
    if closed:
        msg += "\n\nإغلاق خلال 24 ساعة:\n" + "\n".join(closed[:8])
    if decided:
        msg += "\n\nقرارات:\n" + "\n".join(d[:120] for d in decided[:5])
    if not (opened or closed or decided):
        msg += "\n\nلا صفقات ولا قرارات خلال 24 ساعة."
    msg += "\n\nحسابات ورقية، لا مال حقيقي."
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
        ok = send("🧪 اختبار تنبيهات Spot Radar: إذا وصلتك هذه الرسالة فالتوكن الجديد يعمل.")
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
                msgs.append(f"🟢 شراء {t.get('coin')} · حساب {name}\nالسعر {(t.get('price') or 0):g}")
            elif t["side"] == "SELL":
                r = t.get("ret", 0)
                msgs.append(f"{'✅' if r >= 0 else '🔻'} بيع {t.get('coin')} · حساب {name}\nالنتيجة {pct(r)} · السعر {(t.get('price') or 0):g}")
            else:                                        # account B rebalance rows
                msgs.append(f"🔄 حساب {name}: {t.get('side')} {t.get('coin') or ''}")
        for e in st.get("events", []):
            k = f"{name}|{e.get('time')}|{e.get('text', '')[:40]}"
            if k not in sent:
                keys.append(k)
                msgs.append(f"⚙️ حساب {name}: {e.get('text')}")
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    summary = None
    if now.hour >= SUMMARY_HOUR_UTC and ns["summary_day"] != today:
        summary = morning_summary(now)
    if first:
        # first run: remember what already exists so history is not replayed, and say hello
        ns["sent"] = list(sent | set(keys))[-800:]
        ns["summary_day"] = today
        if send("✅ تنبيهات Spot Radar تعمل. ستصلك هنا الصفقات والقرارات وملخص يومي.\n" + (summary or "")):
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
