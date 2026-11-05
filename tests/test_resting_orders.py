"""Resting exchange orders: the stop and target 1 fill at their level between checks, exactly as on the exchange."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import agent

B = agent.BAR
T0 = 1_800_000_000_000 // B * B          # an aligned 15-minute boundary


def run(rows, entry=100.0, stop=90.0, half=False, entry_ms=T0 - 5 * B, start=T0, now=None):
    pos = {"X": dict(entry=entry, stop=stop, r=entry - 90.0, half=half, entry_ms=entry_ms, qty=1.0)}
    fills = []

    def sell(c, frac, px, why, at):
        fills.append((why, frac, px, at))
        if frac >= 0.999:
            pos.pop(c)

    agent.resting_orders(pos, start, now or T0 + 100 * B, lambda c, s: (rows, B), sell)
    return fills, pos


def bar(i, o, h, l, c):
    return [T0 + i * B, o, h, l, c, 1.0]


def test_stop_fills_at_stop_between_checks():
    fills, pos = run([bar(0, 99, 100, 95, 96), bar(1, 96, 97, 88, 92), bar(2, 92, 93, 80, 81)])
    assert fills == [("stop", 1.0, 90.0, T0 + B)] and not pos


def test_gap_below_stop_fills_at_open():
    fills, _ = run([bar(0, 99, 100, 95, 96), bar(1, 85, 86, 84, 85)])
    assert fills == [("stop", 1.0, 85.0, T0 + B)]


def test_target1_sells_half_then_stop_moves_to_break_even():
    fills, pos = run([bar(0, 101, 121, 100.5, 118), bar(1, 118, 119, 99, 100)])
    assert fills[0] == ("partial", 0.5, 120.0, T0)                 # target 1 = 100 + 2 * 10
    assert fills[1] == ("stop", 1.0, 100.0, T0 + B)                # the rest stops out at break-even
    assert not pos


def test_stop_assumed_first_when_both_touch_in_one_candle():
    fills, _ = run([bar(0, 100, 125, 89, 110)])
    assert fills == [("stop", 1.0, 90.0, T0)]


def test_entry_candle_is_skipped():
    entry_ms = T0 + 3 * B + 60_000                                 # bought one minute into candle 3
    rows = [bar(3, 100, 101, 85, 100), bar(4, 100, 101, 99, 100)]  # candle 3's low may predate the buy
    fills, pos = run(rows, entry_ms=entry_ms, start=T0)
    assert fills == [] and "X" in pos


def test_candles_before_the_last_check_are_ignored():
    rows = [bar(0, 99, 100, 80, 96), bar(5, 96, 97, 95, 96)]
    fills, pos = run(rows, start=T0 + 5 * B)
    assert fills == [] and "X" in pos


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} resting-order tests passed")
