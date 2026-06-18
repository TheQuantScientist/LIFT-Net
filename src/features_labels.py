from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
PRICE_COLS = {
    "log_return_1d", "log_return_3d", "log_return_7d", "abs_return_1d",
    "high_low_range", "close_to_open_return", "close_position_in_range",
}
LIQUIDITY_COLS = {
    "dollar_volume", "log_dollar_volume", "Amihud_ILLIQ", "log_ILLIQ",
    "impact_ratio", "volume_zscore_30d", "range_zscore_30d", "ILLIQ_zscore_30d",
}
TECHNICAL_COLS = {
    "realized_vol_7d", "realized_vol_14d", "realized_vol_30d", "drawdown_7d",
    "drawdown_30d", "RSI_14", "ATR_14", "Bollinger_band_width_20",
    "moving_average_gap_7_30", "moving_average_gap_30_90",
}
MARKET_COLS = {
    "market_median_return_1d", "market_median_abs_return_1d", "market_median_ILLIQ",
    "market_median_range", "market_median_impact", "pct_assets_ILLIQ_stressed",
    "pct_assets_range_stressed", "BTC_log_return_1d", "BTC_log_ILLIQ",
    "BTC_high_low_range", "ETH_log_return_1d", "ETH_log_ILLIQ", "ETH_high_low_range",
}


def _zscore(s, window):
    mean = s.rolling(window, min_periods=max(5, window // 3)).mean()
    std = s.rolling(window, min_periods=max(5, window // 3)).std()
    return (s - mean) / (std + EPS)


def _rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    return 100 - 100 / (1 + gain / (loss + EPS))


def _symbol_features(g):
    g = g.sort_values("date").copy()
    g["log_close"] = np.log(g["close"].clip(lower=EPS))
    g["log_return_1d"] = g["log_close"].diff()
    g["log_return_3d"] = g["log_close"].diff(3)
    g["log_return_7d"] = g["log_close"].diff(7)
    g["abs_return_1d"] = g["log_return_1d"].abs()
    g["high_low_range"] = (g["high"] - g["low"]) / g["close"].clip(lower=EPS)
    g["close_to_open_return"] = np.log(g["close"].clip(lower=EPS) / g["open"].clip(lower=EPS))
    g["close_position_in_range"] = (g["close"] - g["low"]) / (g["high"] - g["low"]).clip(lower=EPS)
    qv = g["quote_volume"].where(g["quote_volume"].notna() & (g["quote_volume"] > 0), g["close"] * g["volume"])
    g["dollar_volume"] = qv.clip(lower=0)
    g["log_dollar_volume"] = np.log1p(g["dollar_volume"])
    g["Amihud_ILLIQ"] = g["abs_return_1d"] / g["dollar_volume"].clip(lower=EPS)
    g["log_ILLIQ"] = np.log1p(g["Amihud_ILLIQ"])
    g["impact_ratio"] = g["abs_return_1d"] / np.log1p(g["dollar_volume"]).clip(lower=EPS)
    g["volume_zscore_30d"] = _zscore(g["volume"], 30)
    g["range_zscore_30d"] = _zscore(g["high_low_range"], 30)
    g["ILLIQ_zscore_30d"] = _zscore(g["Amihud_ILLIQ"], 30)
    for w in [7, 14, 30]:
        g[f"realized_vol_{w}d"] = g["log_return_1d"].rolling(w, min_periods=max(3, w // 2)).std()
    for w in [7, 30]:
        roll_max = g["close"].rolling(w, min_periods=max(3, w // 2)).max()
        g[f"drawdown_{w}d"] = g["close"] / roll_max.clip(lower=EPS) - 1
    g["RSI_14"] = _rsi(g["close"], 14)
    tr = pd.concat([
        g["high"] - g["low"],
        (g["high"] - g["close"].shift()).abs(),
        (g["low"] - g["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    g["ATR_14"] = tr.rolling(14, min_periods=7).mean() / g["close"].clip(lower=EPS)
    ma20 = g["close"].rolling(20, min_periods=10).mean()
    sd20 = g["close"].rolling(20, min_periods=10).std()
    g["Bollinger_band_width_20"] = (4 * sd20) / ma20.clip(lower=EPS)
    ma7 = g["close"].rolling(7, min_periods=4).mean()
    ma30 = g["close"].rolling(30, min_periods=15).mean()
    ma90 = g["close"].rolling(90, min_periods=45).mean()
    g["moving_average_gap_7_30"] = ma7 / ma30.clip(lower=EPS) - 1
    g["moving_average_gap_30_90"] = ma30 / ma90.clip(lower=EPS) - 1
    return g


def _add_market_features(df):
    df = df.copy()
    illiq_thr = df.groupby("symbol")["Amihud_ILLIQ"].transform(lambda s: s.rolling(365, min_periods=60).quantile(0.9))
    range_thr = df.groupby("symbol")["high_low_range"].transform(lambda s: s.rolling(365, min_periods=60).quantile(0.9))
    df["_illiq_stressed"] = (df["Amihud_ILLIQ"] > illiq_thr).astype(float)
    df["_range_stressed"] = (df["high_low_range"] > range_thr).astype(float)
    m = df.groupby("date").agg(
        market_median_return_1d=("log_return_1d", "median"),
        market_median_abs_return_1d=("abs_return_1d", "median"),
        market_median_ILLIQ=("Amihud_ILLIQ", "median"),
        market_median_range=("high_low_range", "median"),
        market_median_impact=("impact_ratio", "median"),
        pct_assets_ILLIQ_stressed=("_illiq_stressed", "mean"),
        pct_assets_range_stressed=("_range_stressed", "mean"),
    ).reset_index()
    df = df.merge(m, on="date", how="left")
    for base in ["BTC", "ETH"]:
        sym = f"{base}USDT"
        cols = ["date", "log_return_1d", "log_ILLIQ", "high_low_range"]
        b = df.loc[df["symbol"] == sym, cols].rename(columns={
            "log_return_1d": f"{base}_log_return_1d",
            "log_ILLIQ": f"{base}_log_ILLIQ",
            "high_low_range": f"{base}_high_low_range",
        })
        df = df.merge(b, on="date", how="left")
    return df.drop(columns=["_illiq_stressed", "_range_stressed"])


def _add_labels(g, horizons, window, min_periods, k):
    g = g.sort_values("date").copy()
    illiq_q90 = g["Amihud_ILLIQ"].rolling(window, min_periods=min_periods).quantile(0.90)
    range_q90 = g["high_low_range"].rolling(window, min_periods=min_periods).quantile(0.90)
    impact_q90 = g["impact_ratio"].rolling(window, min_periods=min_periods).quantile(0.90)
    ret_q10 = g["log_return_1d"].rolling(window, min_periods=min_periods).quantile(0.10)
    range_q80 = g["high_low_range"].rolling(window, min_periods=min_periods).quantile(0.80)
    conds = pd.concat([
        g["Amihud_ILLIQ"] > illiq_q90,
        g["high_low_range"] > range_q90,
        g["impact_ratio"] > impact_q90,
        (g["log_return_1d"] < ret_q10) & (g["high_low_range"] > range_q80),
    ], axis=1)
    g["fragile_event"] = (conds.sum(axis=1) >= k).astype(int)
    g.loc[illiq_q90.isna() | range_q90.isna() | impact_q90.isna() | ret_q10.isna(), "fragile_event"] = 0
    for h in horizons:
        future = pd.concat([g["fragile_event"].shift(-i) for i in range(1, h + 1)], axis=1)
        g[f"y_{h}"] = future.max(axis=1)
    g["event_start"] = ((g["fragile_event"] == 1) & (g["fragile_event"].shift(fill_value=0) == 0)).astype(int)
    event_pos = np.where(g["fragile_event"].to_numpy() == 1)[0]
    dtn = np.full(len(g), np.nan)
    j = 0
    for i in range(len(g)):
        while j < len(event_pos) and event_pos[j] <= i:
            j += 1
        if j < len(event_pos):
            dtn[i] = event_pos[j] - i
    g["days_to_next_event"] = dtn
    return g


def _assign_split(df, exp):
    d = df["date"]
    split = np.where((d >= pd.Timestamp(exp["train_start"])) & (d <= pd.Timestamp(exp["train_end"])), "train", None)
    split = np.where((d >= pd.Timestamp(exp["val_start"])) & (d <= pd.Timestamp(exp["val_end"])), "val", split)
    test_end = pd.Timestamp(exp["test_end"]) if exp.get("test_end") else pd.Timestamp.max
    split = np.where((d >= pd.Timestamp(exp["test_start"])) & (d <= test_end), "test", split)
    df["split"] = split
    return df


def build_features_and_labels(df, config):
    horizons = config["labels"]["horizons"]
    fcfg = config["features"]
    out = df.groupby("symbol", group_keys=False).apply(_symbol_features)
    out = _add_market_features(out)
    out = out.groupby("symbol", group_keys=False).apply(
        _add_labels,
        horizons=horizons,
        window=fcfg["label_percentile_window"],
        min_periods=fcfg["label_min_periods"],
        k=config["labels"]["fragility_k"],
    )
    out = _assign_split(out, config["experiment"])
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    path = Path(config["data"]["processed_dir"]) / "panel_features_labels.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(path, index=False)
    except Exception:
        out.to_pickle(path.with_suffix(".pkl"))
    return out


def select_feature_columns(df, ablation="full"):
    exclude = {"symbol", "date", "split", "fragile_event", "event_start", "days_to_next_event", "log_close"}
    exclude |= {c for c in df.columns if c.startswith("y_")}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if ablation == "price_only":
        keep = PRICE_COLS | {"realized_vol_7d", "realized_vol_14d", "realized_vol_30d", "drawdown_7d", "drawdown_30d"}
        return [c for c in cols if c in keep]
    if ablation == "liquidity_only":
        return [c for c in cols if c in LIQUIDITY_COLS]
    if ablation == "no_market_features":
        return [c for c in cols if c not in MARKET_COLS]
    if ablation == "no_technical_features":
        return [c for c in cols if c not in TECHNICAL_COLS]
    return cols


def fit_global_scaler(df, feature_cols):
    scaler = StandardScaler()
    train = df[df["split"] == "train"][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler.fit(train)
    return scaler


def make_sequence_dataset(df, feature_cols, horizon, lookback, split, scaler=None):
    X, y, meta = [], [], []
    label_col = f"y_{horizon}"
    for _, g in df[df["split"].notna()].sort_values("date").groupby("symbol"):
        vals = g[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(dtype=np.float32)
        if scaler is not None:
            vals = scaler.transform(pd.DataFrame(vals, columns=feature_cols)).astype(np.float32)
        labels = g[label_col].to_numpy()
        rows = g[["symbol", "date", "split", "fragile_event", "event_start", "days_to_next_event"]].reset_index(drop=True)
        for i in range(lookback - 1, len(g)):
            if rows.loc[i, "split"] != split or pd.isna(labels[i]):
                continue
            X.append(vals[i - lookback + 1 : i + 1])
            y.append(float(labels[i]))
            m = rows.loc[i].to_dict()
            m["y_true"] = float(labels[i])
            meta.append(m)
    if not X:
        return np.empty((0, lookback, len(feature_cols)), dtype=np.float32), np.array([]), pd.DataFrame(meta)
    return np.stack(X), np.asarray(y, dtype=np.float32), pd.DataFrame(meta)


def make_tabular_dataset(df, feature_cols, horizon, lookback, split, scaler=None):
    Xs, y, meta = make_sequence_dataset(df, feature_cols, horizon, lookback, split, scaler)
    if len(Xs) == 0:
        return np.empty((0, len(feature_cols) * 5)), y, meta
    feats = [Xs[:, -1, :], Xs.mean(axis=1), Xs.std(axis=1), Xs.min(axis=1), Xs.max(axis=1)]
    return np.concatenate(feats, axis=1), y, meta


def make_graph_date_dataset(df, feature_cols, horizon, lookback, split, symbols, scaler=None):
    label_col = f"y_{horizon}"
    sdf = df[df["symbol"].isin(symbols)].copy()
    if scaler is not None:
        vals = sdf[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaled = pd.DataFrame(scaler.transform(vals).astype(np.float32), index=sdf.index, columns=feature_cols)
        sdf = sdf.drop(columns=feature_cols).join(scaled)
    by_symbol = {s: g.sort_values("date").set_index("date") for s, g in sdf.groupby("symbol")}
    dates = sorted(sdf.loc[sdf["split"] == split, "date"].dropna().unique())
    X, y, asset_mask, label_mask, metas = [], [], [], [], []
    for date in dates:
        x_date = np.zeros((len(symbols), lookback, len(feature_cols)), dtype=np.float32)
        y_date = np.zeros(len(symbols), dtype=np.float32)
        am = np.zeros(len(symbols), dtype=np.float32)
        lm = np.zeros(len(symbols), dtype=np.float32)
        rows = []
        for j, sym in enumerate(symbols):
            g = by_symbol.get(sym)
            if g is None or date not in g.index:
                rows.append(None)
                continue
            hist = g.loc[:date].tail(lookback)
            if len(hist) < lookback:
                rows.append(None)
                continue
            x_date[j] = hist[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(np.float32)
            row = g.loc[date]
            am[j] = 1.0
            if pd.notna(row[label_col]):
                y_date[j] = float(row[label_col])
                lm[j] = 1.0
            rows.append(row)
        if lm.sum() > 0:
            X.append(x_date)
            y.append(y_date)
            asset_mask.append(am)
            label_mask.append(lm)
            metas.append({"date": pd.Timestamp(date), "rows": rows})
    if not X:
        empty = np.empty((0, len(symbols), lookback, len(feature_cols)), dtype=np.float32)
        return empty, np.empty((0, len(symbols))), np.empty((0, len(symbols))), np.empty((0, len(symbols))), []
    return np.stack(X), np.stack(y), np.stack(asset_mask), np.stack(label_mask), metas


def flatten_graph_predictions(symbols, metas, probs, preds, threshold, model, seed, horizon, lookback, experiment, ablation, graph_type, status="ok", reason=""):
    rows = []
    for i, meta in enumerate(metas):
        for j, sym in enumerate(symbols):
            row = meta["rows"][j]
            if row is None or pd.isna(row.get(f"y_{horizon}", np.nan)):
                continue
            rows.append({
                "row_type": "prediction", "model": model, "model_group": "proposed", "seed": seed,
                "horizon": horizon, "lookback": lookback, "experiment": experiment, "ablation": ablation,
                "graph_type": graph_type, "split": row["split"], "date": meta["date"], "symbol": sym,
                "y_true": float(row[f"y_{horizon}"]), "y_prob": float(probs[i, j]),
                "y_pred": int(preds[i, j]), "threshold": float(threshold),
                "fragile_event": int(row["fragile_event"]), "event_start": int(row["event_start"]),
                "days_to_next_event": row["days_to_next_event"], "status": status, "reason": reason,
            })
    return pd.DataFrame(rows)
