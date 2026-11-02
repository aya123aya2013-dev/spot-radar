#!/usr/bin/env python3
"""Append-only decision journal with a hash chain: every record carries the hash of the previous one, so any later
edit or deletion breaks the chain and verify() reports where.  One JSON object per line in lab_data/journal.jsonl.
Records both executed and REJECTED decisions with their reasons, so missed opportunities can be studied later
without hindsight (the record holds only what was known at decision time)."""
import hashlib, json, os

import spot_lab as sl

PATH = os.path.join(sl.OUT, "journal.jsonl")


def _hash(prev, body):
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()


def head(path=PATH):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            last = f.read().decode().strip().splitlines()[-1]
        return json.loads(last)["hash"]
    except Exception:
        return "genesis"


def append(records, path=PATH):
    prev = head(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for body in records:
            h = _hash(prev, body)
            f.write(json.dumps(dict(body, prev=prev, hash=h), ensure_ascii=False) + "\n")
            prev = h
    return prev


def verify(path=PATH):
    """Returns (ok, n_records, first_bad_line or None)."""
    prev, n = "genesis", 0
    if not os.path.exists(path):
        return True, 0, None
    for i, line in enumerate(open(path), 1):
        rec = json.loads(line)
        body = {k: v for k, v in rec.items() if k not in ("prev", "hash")}
        if rec.get("prev") != prev or rec.get("hash") != _hash(prev, body):
            return False, n, i
        prev, n = rec["hash"], n + 1
    return True, n, None


def tail(n=200, path=PATH):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path).read().strip().splitlines()[-n:] if l.strip()]
