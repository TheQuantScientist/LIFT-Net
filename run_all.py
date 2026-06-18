import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import load_or_download_binance_daily
from src.features_labels import (
    fit_global_scaler,
    make_graph_date_dataset,
    make_tabular_dataset,
    select_feature_columns,
    build_features_and_labels,
)
from src.liftnet import train_liftnet
from src.metrics import compute_metrics_table, write_summary
from src.models import train_classical_baselines
from src.plots import make_all_plots
from src.tslib_runner import run_tslib_baselines


def parse_args():
    p = argparse.ArgumentParser(description="Run LIFT-Net daily liquidity-fragility proxy study.")
    p.add_argument("--skip_download", action="store_true", help="Use cached raw CSVs only.")
    p.add_argument("--skip_tslib", action="store_true", help="Do not attempt THUML Time-Series-Library baselines.")
    p.add_argument("--quick", action="store_true", help="Debug run: 3 symbols, 1 seed, H=3, fewer epochs.")
    p.add_argument("--make_plots", action="store_true", help="Generate optional figures after training.")
    return p.parse_args()


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config):
    for key in ["raw_dir", "processed_dir"]:
        Path(config["data"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["results_csv"]).parent.mkdir(parents=True, exist_ok=True)


def apply_quick(config):
    config = yaml.safe_load(yaml.safe_dump(config))
    config["symbols"] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    config["labels"]["horizons"] = [config["labels"]["primary_horizon"]]
    config["experiment"]["seeds"] = [config["experiment"]["seeds"][0]]
    config["experiment"]["epochs"] = min(int(config["experiment"]["epochs"]), 6)
    config["experiment"]["patience"] = min(int(config["experiment"]["patience"]), 3)
    installed = []
    for pkg, name in [("xgboost", "xgboost"), ("lightgbm", "lightgbm")]:
        try:
            __import__(pkg)
            installed.append(name)
            break
        except Exception:
            pass
    config["models"]["classical"] = ["persistence", "logistic"] + installed
    config["models"]["tslib"] = []
    config["ablations"]["enabled"] = False
    return config


def add_return_column(preds, panel):
    if preds.empty:
        return preds
    ret = panel[["symbol", "date", "log_return_1d"]].rename(columns={"log_return_1d": "return_1d"})
    out = preds.merge(ret, on=["symbol", "date"], how="left")
    return out


def save_results(preds, config):
    parquet_path = Path(config["outputs"]["results_parquet"])
    csv_path = Path(config["outputs"]["results_csv"])
    try:
        preds.to_parquet(parquet_path, index=False)
    except Exception as exc:
        logging.warning("Could not write parquet (%s); writing pickle beside it.", exc)
        preds.to_pickle(parquet_path.with_suffix(".pkl"))
    preds.to_csv(csv_path, index=False)
    metrics, leader = write_summary(preds, config["outputs"]["summary_excel"])
    return metrics, leader


def build_tabular_splits(panel, feature_cols, horizon, lookback, scaler):
    return {
        split: make_tabular_dataset(panel, feature_cols, horizon, lookback, split, scaler)
        for split in ["train", "val", "test"]
    }


def build_graph_splits(panel, feature_cols, horizon, lookback, symbols, scaler):
    return {
        split: make_graph_date_dataset(panel, feature_cols, horizon, lookback, split, symbols, scaler)
        for split in ["train", "val", "test"]
    }


def run_main_experiments(panel, config, args):
    frames = []
    lookback = int(config["experiment"]["lookback"])
    symbols = config["symbols"]
    for horizon in config["labels"]["horizons"]:
        feature_cols = select_feature_columns(panel, "full")
        scaler = fit_global_scaler(panel, feature_cols)
        tab = build_tabular_splits(panel, feature_cols, horizon, lookback, scaler)
        graph = build_graph_splits(panel, feature_cols, horizon, lookback, symbols, scaler)
        for seed in config["experiment"]["seeds"]:
            logging.info("Main experiment H=%s seed=%s", horizon, seed)
            frames.append(train_classical_baselines(
                config["models"]["classical"], tab, seed, horizon, lookback, "main", "full"
            ))
            if not args.skip_tslib and config["models"].get("tslib"):
                frames.append(run_tslib_baselines(
                    config["models"]["tslib"], tab, feature_cols, config, seed, horizon, lookback, "main", "full"
                ))
            frames.append(train_liftnet(
                graph, feature_cols, symbols, config, seed, horizon, lookback,
                experiment="main", ablation="full", graph_type="liquidity_corr"
            ))
    return frames


def run_ablations(panel, config):
    frames = []
    if not config["ablations"].get("enabled", False):
        return frames
    horizon = int(config["labels"]["primary_horizon"])
    lookback = int(config["experiment"]["lookback"])
    seed = int(config["experiment"]["seeds"][0])
    symbols = config["symbols"]
    if config["ablations"].get("run_feature_ablations", True):
        for ablation in config["ablations"]["feature_ablations"]:
            feature_cols = select_feature_columns(panel, ablation)
            if not feature_cols:
                continue
            scaler = fit_global_scaler(panel, feature_cols)
            graph = build_graph_splits(panel, feature_cols, horizon, lookback, symbols, scaler)
            logging.info("Feature ablation %s", ablation)
            frames.append(train_liftnet(
                graph, feature_cols, symbols, config, seed, horizon, lookback,
                experiment="feature_ablation", ablation=ablation, graph_type="liquidity_corr"
            ))
    if config["ablations"].get("run_graph_ablations", True):
        feature_cols = select_feature_columns(panel, "full")
        scaler = fit_global_scaler(panel, feature_cols)
        graph = build_graph_splits(panel, feature_cols, horizon, lookback, symbols, scaler)
        for graph_type in config["ablations"]["graph_ablations"]:
            logging.info("Graph ablation %s", graph_type)
            frames.append(train_liftnet(
                graph, feature_cols, symbols, config, seed, horizon, lookback,
                experiment="graph_ablation", ablation="full", graph_type=graph_type
            ))
    return frames


def print_leaderboard(leader):
    if leader.empty:
        print("\nNo leaderboard rows were produced.")
        return
    cols = ["model", "model_group", "horizon", "seed", "experiment", "ablation", "graph_type", "AUPRC", "AUROC", "F1"]
    show = leader[[c for c in cols if c in leader.columns]].head(25)
    print("\nFinal leaderboard sorted by test AUPRC")
    print(show.to_string(index=False))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    if args.quick:
        config = apply_quick(config)
    ensure_dirs(config)
    raw = load_or_download_binance_daily(
        config["symbols"],
        config["start_date"],
        config.get("end_date"),
        config["data"]["raw_dir"],
        use_cached=(config["data"].get("use_cached", True) or args.skip_download),
        allow_download=not args.skip_download,
    )
    panel = build_features_and_labels(raw, config)
    frames = run_main_experiments(panel, config, args)
    frames.extend(run_ablations(panel, config))
    preds = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    preds = add_return_column(preds, panel)
    _, leader = save_results(preds, config)
    if args.make_plots:
        try:
            saved = make_all_plots(config["outputs"]["results_parquet"], config["outputs"]["summary_excel"], config["outputs"]["figures_dir"])
            logging.info("Saved %s figure(s)", len(saved))
        except Exception as exc:
            logging.warning("Plotting failed: %s", exc)
    print_leaderboard(leader)


if __name__ == "__main__":
    main()
