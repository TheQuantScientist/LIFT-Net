import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_metric(fn, y_true, y_prob_or_pred, default=np.nan):
    try:
        if len(np.unique(y_true)) < 2 and fn in {roc_auc_score, average_precision_score}:
            return default
        return fn(y_true, y_prob_or_pred)
    except Exception:
        return default


def best_f1_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def threshold_for_false_alert_rate(y_true, y_prob, max_far=0.10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    negatives = max((y_true == 0).sum(), 1)
    best = 1.0
    for th in np.unique(np.quantile(y_prob, np.linspace(0, 1, 101))):
        pred = y_prob >= th
        far = ((pred == 1) & (y_true == 0)).sum() / negatives
        if far <= max_far:
            best = float(th)
            break
    return best


def metric_dict(y_true, y_prob, threshold, far_threshold=None, dates=None, days_to_next_event=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    far_threshold = threshold if far_threshold is None else far_threshold
    y_pred_far = (y_prob >= far_threshold).astype(int)
    neg = max((y_true == 0).sum(), 1)
    false_alerts = ((y_pred == 1) & (y_true == 0)).sum()
    months = 1.0
    if dates is not None and len(dates):
        d = pd.to_datetime(pd.Series(dates))
        months = max((d.max() - d.min()).days / 30.4375, 1.0)
    lead_vals = []
    if days_to_next_event is not None:
        dtn = pd.to_numeric(pd.Series(days_to_next_event), errors="coerce").to_numpy()
        lead_vals = dtn[(y_pred == 1) & np.isfinite(dtn)]
    return {
        "AUPRC": safe_metric(average_precision_score, y_true, y_prob),
        "AUROC": safe_metric(roc_auc_score, y_true, y_prob),
        "F1": safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_true, y_pred),
        "Precision": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_true, y_pred),
        "Recall": safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_true, y_pred),
        "Balanced Accuracy": safe_metric(balanced_accuracy_score, y_true, y_pred),
        "Brier Score": safe_metric(brier_score_loss, y_true, y_prob),
        "Precision@10%": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_true, y_pred_far),
        "Recall@10% false-alert rate": safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_true, y_pred_far),
        "False alerts per month": false_alerts / months,
        "Missed event rate": 1 - safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_true, y_pred),
        "Mean lead time": float(np.nanmean(lead_vals)) if len(lead_vals) else np.nan,
        "Median lead time": float(np.nanmedian(lead_vals)) if len(lead_vals) else np.nan,
        "false_alert_rate_threshold": far_threshold,
    }


def compute_metrics_table(predictions):
    rows = []
    ok = predictions[(predictions["row_type"] == "prediction") & (predictions["status"] == "ok")].copy()
    keys = ["model", "model_group", "seed", "horizon", "lookback", "experiment", "ablation", "graph_type", "split"]
    for key, g in ok.groupby(keys, dropna=False):
        vals = metric_dict(
            g["y_true"],
            g["y_prob"],
            float(g["threshold"].dropna().iloc[0]) if g["threshold"].notna().any() else 0.5,
            dates=g["date"],
            days_to_next_event=g.get("days_to_next_event"),
        )
        row = dict(zip(keys, key))
        row.update(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def economic_results(predictions):
    rows = []
    ok = predictions[(predictions["row_type"] == "prediction") & (predictions["status"] == "ok")].copy()
    if "return_1d" not in ok.columns:
        ok["return_1d"] = np.nan
    keys = ["model", "seed", "horizon", "experiment", "ablation", "graph_type", "split"]
    for key, g in ok.groupby(keys, dropna=False):
        r = pd.to_numeric(g.get("return_1d"), errors="coerce").fillna(0).to_numpy()
        prob = g["y_prob"].to_numpy()
        th = float(g["threshold"].dropna().iloc[0]) if g["threshold"].notna().any() else 0.5
        exposure = np.where(prob > th, 0.5, 1.0)
        strat = exposure * r
        wealth = np.cumprod(1 + strat)
        peak = np.maximum.accumulate(wealth) if len(wealth) else np.array([1])
        dd = wealth / peak - 1 if len(wealth) else np.array([0])
        row = dict(zip(keys, key))
        fragile = g["fragile_event"].to_numpy() == 1
        row.update({
            "average return": float(np.mean(strat)) if len(strat) else np.nan,
            "volatility": float(np.std(strat)) if len(strat) else np.nan,
            "max drawdown": float(np.min(dd)) if len(dd) else np.nan,
            "5% tail loss": float(np.quantile(strat, 0.05)) if len(strat) else np.nan,
            "avoided fragile days %": float(np.mean((exposure < 1) & fragile)) if len(strat) else np.nan,
            "turnover proxy": float(np.mean(np.abs(np.diff(exposure)))) if len(exposure) > 1 else 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(predictions, path):
    metrics = compute_metrics_table(predictions)
    econ = economic_results(predictions)
    skipped = predictions[predictions["status"] == "skipped"].drop_duplicates(
        ["model", "seed", "horizon", "experiment", "ablation", "graph_type", "reason"]
    )
    leader = metrics[metrics["split"] == "test"].sort_values("AUPRC", ascending=False) if not metrics.empty else metrics
    main = metrics[(metrics["experiment"] == "main") & (metrics["ablation"] == "full")] if not metrics.empty else metrics
    horizon = main.groupby(["model", "horizon"], dropna=False).mean(numeric_only=True).reset_index() if not main.empty else main
    feat = metrics[metrics["experiment"].eq("feature_ablation")] if not metrics.empty else metrics
    graph = metrics[metrics["experiment"].eq("graph_ablation")] if not metrics.empty else metrics
    with pd.ExcelWriter(path) as writer:
        leader.to_excel(writer, sheet_name="leaderboard", index=False)
        main.to_excel(writer, sheet_name="main_results", index=False)
        horizon.to_excel(writer, sheet_name="horizon_results", index=False)
        feat.to_excel(writer, sheet_name="feature_ablations", index=False)
        graph.to_excel(writer, sheet_name="graph_ablations", index=False)
        econ.to_excel(writer, sheet_name="economic_results", index=False)
        skipped.to_excel(writer, sheet_name="skipped_models", index=False)
    return metrics, leader
