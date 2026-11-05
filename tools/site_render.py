#!/usr/bin/env python3
"""Render a JavaScript page on a GitHub runner and print its text and the JSON calls it makes (exploration helper)."""
import json, sys
from playwright.sync_api import sync_playwright

url = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(locale="ar", user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36")
    calls = []
    def on_resp(r):
        ct = r.headers.get("content-type", "")
        if "json" in ct or "/api" in r.url or "_rsc" in r.url:
            try:
                calls.append((r.url, r.status, ct, r.text()[:600]))
            except Exception as e:
                calls.append((r.url, r.status, ct, f"<{e}>"))
    pg.on("response", on_resp)
    pg.goto(url, wait_until="networkidle", timeout=60000)
    pg.wait_for_timeout(4000)
    print("FINAL URL", pg.url)
    txt = pg.inner_text("body")
    print("TEXT", len(txt)); print(txt[:6000])
    print("INPUTS", pg.eval_on_selector_all("input", "els=>els.map(e=>[e.type,e.placeholder,e.name])"))
    print("LINKS", pg.eval_on_selector_all("a", "els=>[...new Set(els.map(e=>e.getAttribute('href')))].slice(0,120)"))
    for u, s, ct, t in calls[:40]:
        print("CALL", s, ct, u); print("   ", t.replace("\n", " ")[:600])
    b.close()
