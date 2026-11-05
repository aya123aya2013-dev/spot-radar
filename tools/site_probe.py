#!/usr/bin/env python3
"""Probe the halal-screening site from a GitHub runner (the sandbox cannot reach it): prints the page structure."""
import re, sys, urllib.request
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
for url in sys.argv[1:]:
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30)
        body = r.read().decode("utf-8", "replace")
        print(f"== {url} status {r.status} len {len(body)} type {r.headers.get('content-type')}")
        print("title:", re.findall(r"<title>(.*?)</title>", body, re.S)[:1])
        links = sorted(set(re.findall(r'href="([^"#]+)"', body)))
        print("links:", len(links), links[:80])
        print("scripts:", re.findall(r'<script[^>]+src="([^"]+)"', body)[:20])
        for k in ("Bitcoin Cash", "Zcash", "Stellar", "Hedera", "Bittensor", "halal", "Halal", "حلال", "مباح", "haram", "Haram", "__NEXT_DATA__", "api/"):
            i = body.find(k)
            print(f"  [{k}] at {i}:", body[max(0, i - 150):i + 250].replace("\n", " ") if i >= 0 else "-")
    except Exception as e:
        print(f"== {url} ERROR {e}")
