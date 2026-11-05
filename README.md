# spot-radar: the agent (G)

A paper-trading crypto SPOT agent, run as if it were real money. Halal coins only (`coins_policy.json`, source cryptohalal.cc). No exchange keys, no real orders.

- `agent.py`: the agent. Reads the market regime daily (BTC 50/200-day averages and breadth), sets a risk budget, trades the leaders on breakouts or pullback resumes, 2.5 ATR stop, trailing 3 ATR, half sold at +2R. `python agent.py backtest` and `python agent.py variants` run the champion/challenger tests.
- `risk.py` + `control.json`: the deterministic risk engine. The agent proposes, the risk engine decides. Limits and the emergency stop live in `control.json`.
- `journal.py` (hash-chained decision journal), `accounting.py` (P&L, fees, slippage, ratios, operating costs from `costs.json`).
- `lessons.py` (daily lesson from closed trades), `monthly.py` (monthly scorecard vs BTC), `notify.py` (Telegram).
- `halal_check.py` + `policy.py`: the Sharia rulings, read daily from cryptohalal.cc's public API; only coins ruled مباح are traded.
- `okx_check.py`, `okx_candles.py`: OKX public order books and candles, to measure real costs. `market.py`: shared data helpers. `tools/`: runner-side site probes.
- `tests/`: invariants of the risk engine, the journal and the agent. Run on every push.
- `index.html`: the dashboard (GitHub Pages). State lives on the `data` branch (`lab_data/`).

Workflows: `paper-trade` (one cycle about every 30-40 minutes), `backtest` (weekly: agent backtest, challengers, noise yardstick), `monthly`, `okx-candles`, `tests`.

State backups: branches `backup-mon` ... `backup-sun` (one per weekday). The old accounts A-F were deleted on 4 Nov 2026; their code is on branch `archive-legacy-code` and their data on `archive-legacy`.
