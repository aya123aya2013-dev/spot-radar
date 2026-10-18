#!/usr/bin/env python3
"""High win-rate research ("quick strikes"): small take-profit, bounded stop, time exit.

Goal asked by the owner: win rate >= 70%. A high win rate is easy to fake (tiny target, far stop) and
means nothing unless the average trade still makes money after costs, so a candidate must pass ALL of:
  selection (first 70%): >= 30 trades, win rate >= 70%, net average trade > 0, positive in >= 2 of 3 folds
  exam (last 30%, never used to choose): >= 15 trades, win rate >= 70%, net average trade > 0,
      still > 0 with fees and slippage x3, bootstrap P(average > 0) >= 0.9, beats RANDOM entries
      with the same exits (so the entry signal itself adds something), worst single loss reported.
Read-only research: writes lab_data/hitrate_C.json. It never trades.
With --synthetic it runs the SAME pipeline on random-walk noise and writes hitrate_null.json: whatever it
'accepts' there is pure multiple-testing luck, the yardstick the real result must clearly beat.
"""
import json, os, random, sys
from datetime import datetime, timezone

import spot_lab as sl
import research as rs

OUT = sl.OUT
FOLDS, SEL_FRAC = 3, 0.7
MIN_SEL, MIN_EXAM, WIN_MIN, BOOT_MIN = 30, 15, 0.70, 0.90
TFS = ("1h", "4h")
TPS = (0.008, 0.012, 0.02)        # take-profit
SLS = (0.03, 0.05, 0.08)          # hard stop
HOLDS = (12, 24, 48)              # time exit in bars


# ---------------------------------------------------------------- entry signals (bool per bar, decided on the closed bar)
def e_rsi_dip(rows, th, trend):
    c = [r[4] for r in rows]
    R, e200 = sl.rsi(c), sl.ema(c, 200)
    return [R[i] is not None and R[i] < th and (not trend or c[i] > e200[i]) for i in range(len(c))]


def e_boll_dip(rows, k, trend):
    c = [r[4] for r in rows]
    _, _, lo = sl.boll(c, 20, k)
    e200 = sl.ema(c, 200)
    return [lo[i] is not None and c[i - 1] >= (lo[i - 1] or 0) and c[i] < lo[i] and (not trend or c[i] > e200[i])
            for i in range(len(c))] if len(c) > 1 else []


def e_ema_pull(rows, band, rmax):
    c = [r[4] for r in rows]
    e20, e50, e200 = sl.ema(c, 20), sl.ema(c, 50), sl.ema(c, 200)
    R, at = sl.rsi(c), sl.atr(rows)
    return [R[i] is not None and e50[i] > e200[i] and c[i] > e200[i] and abs(c[i] - e20[i]) < band * at[i]
            and c[i] < e20[i] * 1.002 and R[i] < rmax for i in range(len(c))]


def e_wick(rows, wick, trend):
    """Long lower wick (buyers rejected the low) after a down close."""
    c = [r[4] for r in rows]
    e200 = sl.ema(c, 200)
    out = []
    for i, r in enumerate(rows):
        rng = r[2] - r[3]
        body_low = min(r[1], r[4])
        out.append(rng > 0 and (body_low - r[3]) / rng >= wick and r[4] < r[1] and (not trend or c[i] > e200[i]))
    return out


FAMILIES = {
    "rsi_dip": (e_rsi_dip, [dict(th=t, trend=tr) for t in (25, 30, 35) for tr in (True, False)]),
    "boll_dip": (e_boll_dip, [dict(k=k, trend=tr) for k in (2.0, 2.5) for tr in (True, False)]),
    "ema_pull": (e_ema_pull, [dict(band=b, rmax=r) for b in (0.5, 1.0) for r in (40, 50)]),
    "wick_reject": (e_wick, [dict(wick=w, trend=tr) for w in (0.6, 0.7) for tr in (True, False)]),
}


# ---------------------------------------------------------------- simulator: one position at a time, intrabar exits
def run(rows, en, tp, sp, hold, start, end, step=1, cost_mult=1.0):
    """Returns the list of net trade returns. If a bar touches both stop and target, the stop is assumed first."""
    fee, slip = sl.FEE * cost_mult, sl.SLIP * cost_mult
    rets, i = [], start
    while i < end - hold - 2:
        if not en[i]:
            i += step
            continue
        px = rows[i + 1][1] * (1 + slip)
        tgt, stp = px * (1 + tp), px * (1 - sp)
        exit_px, j = None, i + 1
        for j in range(i + 1, i + 1 + hold):
            lo, hi = rows[j][3], rows[j][2]
            if lo <= stp:
                exit_px = stp * (1 - slip)
                break
            if hi >= tgt:
                exit_px = tgt * (1 - slip)
                break
        if exit_px is None:
            j = i + hold
            exit_px = rows[j][4] * (1 - slip)
        rets.append(exit_px * (1 - fee) / (px * (1 + fee)) - 1)
        i = j + 1
    return rets


def stats(r):
    if not r:
        return dict(n=0, win=0.0, mean=0.0, worst=0.0)
    return dict(n=len(r), win=sum(x > 0 for x in r) / len(r), mean=sum(r) / len(r), worst=min(r))


def boot_mean_pos(r, trials=1000):
    if len(r) < 5:
        return 0.0
    rnd, ok = random.Random(5), 0
    for _ in range(trials):
        ok += sum(rnd.choice(r) for _ in r) > 0
    return ok / trials


def max_loss_streak(r):
    m = cur = 0
    for x in r:
        cur = cur + 1 if x <= 0 else 0
        m = max(m, cur)
    return m


def main():
    found, tested, passed_sel, passed_exam_only = [], 0, 0, 0
    per_chart = []
    for coin in rs.C_COINS:
        for tf in TFS:
            try:
                rows = sl.klines(coin, tf)
            except Exception as e:
                print(f"! {coin} {tf}: {e}")
                continue
            if len(rows) < 900:
                continue
            n = len(rows)
            sel = int(n * SEL_FRAC)
            edges = [sl.WARMUP + (sel - sl.WARMUP) * k // FOLDS for k in range(FOLDS + 1)]
            best_here = None
            for fam, (fn, grid) in FAMILIES.items():
                for params in grid:
                    en = fn(rows, **params)
                    for tp in TPS:
                        for sp in SLS:
                            for hold in HOLDS:
                                tested += 1
                                folds = [stats(run(rows, en, tp, sp, hold, edges[k], edges[k + 1])) for k in range(FOLDS)]
                                tot = sum(f["n"] for f in folds)
                                if tot < MIN_SEL:
                                    continue
                                allr = [x for k in range(FOLDS) for x in run(rows, en, tp, sp, hold, edges[k], edges[k + 1])]
                                s = stats(allr)
                                pos_folds = sum(f["mean"] > 0 for f in folds if f["n"])
                                if s["win"] < WIN_MIN or s["mean"] <= 0 or pos_folds < 2:
                                    continue
                                passed_sel += 1
                                ex = run(rows, en, tp, sp, hold, sel, n)
                                es = stats(ex)
                                cand = dict(coin=coin, tf=tf, family=fam, params=params, tp=tp, sl=sp, hold=hold,
                                            sel=s, exam=es, sel_score=s["mean"])
                                if es["n"] >= MIN_EXAM and es["win"] >= WIN_MIN and es["mean"] > 0:
                                    st = stats(run(rows, en, tp, sp, hold, sel, n, cost_mult=rs.COST_STRESS))
                                    # same exits, entry on every 7th bar (no signal): what luck alone gives
                                    rnd_en = [True] * n
                                    base = stats(run(rows, rnd_en, tp, sp, hold, sel, n, step=7))
                                    cand.update(stress_mean=st["mean"], boot_p=boot_mean_pos(ex),
                                                random_mean=base["mean"], random_win=base["win"],
                                                loss_streak=max_loss_streak(ex), exam_rets=[round(x, 5) for x in ex])
                                    ok = (st["mean"] > 0 and cand["boot_p"] >= BOOT_MIN and es["mean"] > base["mean"])
                                    cand["accepted"] = ok
                                    if ok:
                                        found.append(cand)
                                else:
                                    passed_exam_only += 1
                                if best_here is None or cand["sel_score"] > best_here["sel_score"]:
                                    best_here = cand
            per_chart.append(dict(coin=coin, tf=tf, best=best_here and dict(
                family=best_here["family"], tp=best_here["tp"], sl=best_here["sl"], hold=best_here["hold"],
                sel_win=best_here["sel"]["win"], exam_win=best_here["exam"]["win"], exam_mean=best_here["exam"]["mean"],
                exam_n=best_here["exam"]["n"])))
            print(f"{coin:4} {tf}  best-by-selection: " + (
                f"{best_here['family']} tp {best_here['tp']*100:.1f}% sl {best_here['sl']*100:.0f}% h{best_here['hold']}  "
                f"sel win {best_here['sel']['win']*100:.0f}%  exam win {best_here['exam']['win']*100:.0f}% "
                f"avg {best_here['exam']['mean']*100:+.2f}% n={best_here['exam']['n']}" if best_here else "nothing passed selection"))
    found.sort(key=lambda c: c["exam"]["mean"] * c["exam"]["n"], reverse=True)
    # honest reading: out of everything that looked good in selection, how many held up on the unseen exam?
    sel_total = sum(1 for c in per_chart if c["best"])
    held = [c for c in per_chart if c["best"] and c["best"]["exam_win"] >= WIN_MIN and c["best"]["exam_mean"] > 0]
    summary = dict(tested=tested, passed_selection=passed_sel, accepted=len(found),
                   charts=len(per_chart), charts_with_selection_pick=sel_total, charts_pick_held_on_exam=len(held),
                   min_win=WIN_MIN, generated=datetime.now(timezone.utc).isoformat())
    json.dump(dict(summary=summary, accepted=found[:30], per_chart=per_chart,
                   note="exam = last 30% of each chart, never used for selection; costs: fee 0.1% + slip 0.05% per side"),
              open(os.path.join(OUT, "hitrate_null.json" if sl.SYN else "hitrate_C.json"), "w"), indent=1)
    print(f"\ntested {tested} combos; passed selection {passed_sel}; accepted after the strict exam {len(found)}")
    print(f"charts whose best-by-selection pick also held >= 70% win and profit on the exam: {len(held)} of {sel_total}")
    for c in found[:10]:
        print(f"  ACCEPT {c['coin']} {c['tf']} {c['family']} {c['params']} tp {c['tp']*100:.1f}% sl {c['sl']*100:.0f}% h{c['hold']}: "
              f"exam win {c['exam']['win']*100:.0f}% avg {c['exam']['mean']*100:+.2f}% n={c['exam']['n']} "
              f"x3 costs {c['stress_mean']*100:+.2f}% random {c['random_mean']*100:+.2f}% boot {c['boot_p']:.2f}")


if __name__ == "__main__":
    main()
