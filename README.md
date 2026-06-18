# LIFT-Net

Minimal one-command research pipeline for **LIFT-Net: Liquidity-Fragility Transition Network for Early Warning in Cryptocurrency Markets**.

Run the full daily proxy study:

```bash
python run_all.py
```

Quick smoke test:

```bash
python run_all.py --quick
python run_all.py --quick --make_plots
```

Useful flags:

```bash
python run_all.py --skip_download
python run_all.py --skip_tslib
python run_all.py --make_plots
```

## Data

The pipeline downloads Binance spot daily OHLCV klines for the configured USDT pairs and caches one CSV per symbol in `data/raw/`. The study uses daily liquidity-fragility proxy features only: dollar volume, Amihud-style illiquidity, range, impact proxy, volatility, drawdown, technical controls, and market-wide cross-sectional features. It does not claim true order-book liquidity, bid-ask spread, or market depth.

## THUML Time-Series-Library

If you have THUML Time-Series-Library locally, set:

```bash
export TSLIB_PATH=/path/to/Time-Series-Library
```

The runner checks requested TSLib models and skips unavailable or constructor-incompatible models gracefully.

Classical optional packages are handled with fallbacks. If `xgboost`, `lightgbm`, or `hmmlearn` are missing, the pipeline trains dependency-free sklearn fallback baselines and records the fallback reason in `results/results.csv` and `results/summary.xlsx`.

## Outputs

- `results/results.parquet`: prediction-level long table for all models.
- `results/results.csv`: CSV copy of the prediction table.
- `results/summary.xlsx`: leaderboard, metrics, ablations, economic proxy results, and skipped models.
- `results/figures/`: optional plots when `--make_plots` is passed.

## Model

LIFT-Net predicts whether each asset enters a future daily liquidity-fragility state over horizons H = 1, 3, and 7. It encodes each asset's rolling lookback window with a small temporal convolutional encoder. A dynamic graph is computed from recent liquidity-proxy correlations, with ablations for return correlation, fully connected, and identity graphs. The graph layer aggregates neighbor embeddings without torch-geometric, then a hazard head outputs per-asset transition risk. Training uses masked BCE loss so unbalanced panels are allowed. Baselines include persistence, logistic regression, random forest, optional gradient boosting, optional HMM, optional isolation forest, and attempted TSLib models.
