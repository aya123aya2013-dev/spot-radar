# spot-radar
Live crypto spot dashboard (BTC, ETH, SOL, OP, POL) plus a $5000 paper-trading account.

- `index.html`: the dashboard (GitHub Pages).
- `spot_lab.py`: backtests strategies and paper-trades.
- `.github/workflows/backtest.yml`: picks strategies (weekly, or run it by hand).
- `.github/workflows/paper.yml`: one paper-trading cycle every 5 minutes.
- State lives on the `data` branch (`lab_data/`), rewritten as a single commit each time.

Setup: Settings > Pages > Deploy from branch `main` / root. Then Actions > backtest > Run workflow, then Actions > paper-trade > Run workflow.
