# spot-radar: the agent (G)

A paper-trading crypto SPOT agent, run as if it were real money. Halal coins only (`coins_policy.json`, source cryptohalal.cc). No exchange keys, no real orders.

- `strategy.json`: the live strategy's parameters (defaults = the original rules), changed only by the research rule. `candidates.json`: the challengers tested every week.
- `agent.py`: the agent. Reads the market regime daily (BTC 50/200-day averages and breadth), sets a risk budget, trades the leaders on breakouts or pullback resumes, 2.5 ATR stop, trailing stop at the lowest low of 20 days (adopted 8 Nov 2026 by the research rule), half sold at +2R. `python agent.py backtest` runs the champion/challenger test; `python agent.py research` tests every candidate under the pre-registered RESEARCH_RULE, and `apply_research.py` applies its verdict mechanically.
- `risk.py` + `control.json`: the deterministic risk engine. The agent proposes, the risk engine decides. Limits and the emergency stop live in `control.json`.
- `journal.py` (hash-chained decision journal), `accounting.py` (P&L, fees, slippage, ratios, operating costs from `costs.json`).
- `lessons.py` (daily lesson from closed trades), `monthly.py` (monthly scorecard vs BTC), `notify.py` (Telegram).
- `halal_check.py` + `policy.py`: the Sharia rulings, read daily from cryptohalal.cc's public API; only coins ruled مباح are traded.
- `okx_check.py`, `okx_candles.py`: OKX public order books and candles, to measure real costs. `market.py`: shared data helpers. `tools/`: runner-side site probes.
- `tests/`: invariants of the risk engine, the journal and the agent. Run on every push.
- `index.html`: the dashboard (GitHub Pages), one self-contained file; `changelog.json` feeds its change log. State lives on the `data` branch (`lab_data/`).

Workflows: `paper-trade` (one cycle about every 30-40 minutes), `backtest` (weekly: agent backtest, research round with automatic adoption, noise yardstick), `monthly`, `okx-candles`, `tests`.

State backups: branches `backup-mon` ... `backup-sun` (one per weekday). The old accounts A-F were deleted on 4 Nov 2026; their code is on branch `archive-legacy-code` and their data on `archive-legacy`.

## For the weekly review (read this first)

Since 8 Nov 2026 the old `variants()` / `VARIANT_RULE` / `variants_G.json` no longer exist. The learning loop is:

1. Ideas go into `candidates.json` (one change each, parameter keys from `DEFAULT_PARAMS` in `agent.py`, `added` date, `status: active`). Retire an idea (`status: retired`) after 4 consecutive rejections.
2. Every Sunday the `backtest` workflow runs `python agent.py research`: every active candidate against the live strategy under `RESEARCH_RULE` (pre-registered; never edit the rule or its numbers). Results: `lab_data/research_G.json` on the `data` branch.
3. `apply_research.py` applies the verdict mechanically to `strategy.json` / `control.json`. Nobody edits those two files by hand or by judgement.
4. A new ability (a new parameter) may be added to `agent.py` only if its default keeps today's behaviour exactly: add it to `DEFAULT_PARAMS` and `PARAM_KEYS` in `apply_research.py`, guard every use with it, add a test in `tests/test_strategy.py`, and `test_default_rules_are_frozen` must pass unchanged.
5. Tests: `tests/test_risk.py`, `tests/test_journal_agent.py`, `tests/test_resting_orders.py`, `tests/test_policy.py`, `tests/test_strategy.py`, and `SPOT_LAB_DATA=$(mktemp -d) python3 agent.py research --synthetic` (a code check only).
6. The change log for the owner is `changelog.json` (newest first); the dashboard reads it.
