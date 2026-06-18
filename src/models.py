import importlib
import logging
import os
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from .metrics import best_f1_threshold, threshold_for_false_alert_rate


def _prediction_frame(meta, y_prob, threshold, model, group, seed, horizon, lookback, experiment, ablation, graph_type="none", status="ok", reason=""):
    if meta.empty:
        return pd.DataFrame()
    out = meta.copy()
    out["row_type"] = "prediction"
    out["model"] = model
    out["model_group"] = group
    out["seed"] = seed
    out["horizon"] = horizon
    out["lookback"] = lookback
    out["experiment"] = experiment
    out["ablation"] = ablation
    out["graph_type"] = graph_type
    out["y_prob"] = np.asarray(y_prob, dtype=float)
    out["threshold"] = float(threshold)
    out["y_pred"] = (out["y_prob"] >= threshold).astype(int)
    out["status"] = status
    out["reason"] = reason
    return out[[
        "row_type", "model", "model_group", "seed", "horizon", "lookback", "experiment", "ablation",
        "graph_type", "split", "date", "symbol", "y_true", "y_prob", "y_pred", "threshold",
        "fragile_event", "event_start", "days_to_next_event", "status", "reason",
    ]]


def skipped_row(model, group, seed, horizon, lookback, experiment, ablation, reason, graph_type="none"):
    return pd.DataFrame([{
        "row_type": "prediction", "model": model, "model_group": group, "seed": seed, "horizon": horizon,
        "lookback": lookback, "experiment": experiment, "ablation": ablation, "graph_type": graph_type,
        "split": "all", "date": pd.NaT, "symbol": "", "y_true": np.nan, "y_prob": np.nan, "y_pred": np.nan,
        "threshold": np.nan, "fragile_event": np.nan, "event_start": np.nan, "days_to_next_event": np.nan,
        "status": "skipped", "reason": reason,
    }])


def _has_two_classes(y):
    return len(np.unique(y.astype(int))) >= 2


def _calibrated_anomaly_probs(train_scores, val_scores, test_scores):
    lo, hi = np.nanmin(val_scores), np.nanmax(val_scores)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(train_scores), np.nanmax(train_scores)
    scale = max(hi - lo, 1e-12)
    return [np.clip((s - lo) / scale, 0, 1) for s in [train_scores, val_scores, test_scores]]


def _weighted_fit(model, X, y):
    pos = max(float((y == 1).sum()), 1.0)
    neg = max(float((y == 0).sum()), 1.0)
    weights = np.where(y == 1, neg / pos, 1.0)
    model.fit(X, y, sample_weight=weights)
    return model


def _fit_predict_model(name, Xtr, ytr, Xv, yv, Xte, seed):
    if name == "logistic":
        if not _has_two_classes(ytr):
            raise ValueError("training labels contain one class")
        model = LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=seed)
        model.fit(Xtr, ytr)
        return (model.predict_proba(Xtr)[:, 1], model.predict_proba(Xv)[:, 1], model.predict_proba(Xte)[:, 1]), ""
    if name == "random_forest":
        if not _has_two_classes(ytr):
            raise ValueError("training labels contain one class")
        model = RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=1)
        model.fit(Xtr, ytr)
        return (model.predict_proba(Xtr)[:, 1], model.predict_proba(Xv)[:, 1], model.predict_proba(Xte)[:, 1]), ""
    if name == "xgboost":
        if not _has_two_classes(ytr):
            raise ValueError("training labels contain one class")
        try:
            xgb = importlib.import_module("xgboost")
            pos = max(float((ytr == 1).sum()), 1.0)
            neg = max(float((ytr == 0).sum()), 1.0)
            model = xgb.XGBClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=seed, scale_pos_weight=neg / pos, n_jobs=2,
            )
            model.fit(Xtr, ytr)
            reason = ""
        except ImportError:
            model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=seed)
            _weighted_fit(model, Xtr, ytr)
            reason = "xgboost package missing; used sklearn GradientBoosting fallback"
        return (model.predict_proba(Xtr)[:, 1], model.predict_proba(Xv)[:, 1], model.predict_proba(Xte)[:, 1]), reason
    if name == "lightgbm":
        if not _has_two_classes(ytr):
            raise ValueError("training labels contain one class")
        try:
            lgb = importlib.import_module("lightgbm")
            model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, class_weight="balanced", random_state=seed, verbose=-1)
            model.fit(Xtr, ytr)
            reason = ""
        except ImportError:
            model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=seed)
            _weighted_fit(model, Xtr, ytr)
            reason = "lightgbm package missing; used sklearn GradientBoosting fallback"
        return (model.predict_proba(Xtr)[:, 1], model.predict_proba(Xv)[:, 1], model.predict_proba(Xte)[:, 1]), reason
    if name == "isolation_forest":
        model = IsolationForest(n_estimators=250, contamination="auto", random_state=seed)
        model.fit(Xtr)
        scores = [-model.decision_function(x) for x in [Xtr, Xv, Xte]]
        return _calibrated_anomaly_probs(*scores), ""
    raise ValueError(f"unknown model {name}")


def train_classical_baselines(model_names, datasets, seed, horizon, lookback, experiment, ablation):
    Xtr, ytr, mtr = datasets["train"]
    Xv, yv, mv = datasets["val"]
    Xte, yte, mte = datasets["test"]
    frames = []
    for name in model_names:
        try:
            reason = ""
            if name == "persistence":
                probs = [m["fragile_event"].astype(float).to_numpy() for m in [mtr, mv, mte]]
            elif name == "hmm":
                probs, reason = _run_hmm(Xtr, ytr, Xv, Xte, seed)
            else:
                probs, reason = _fit_predict_model(name, Xtr, ytr.astype(int), Xv, yv.astype(int), Xte, seed)
            threshold = best_f1_threshold(yv, probs[1]) if len(yv) else 0.5
            threshold_for_false_alert_rate(yv, probs[1]) if len(yv) else threshold
            if reason:
                logging.info("%s: %s", name, reason)
            for split, meta, prob in zip(["train", "val", "test"], [mtr, mv, mte], probs):
                frames.append(_prediction_frame(meta, prob, threshold, name, "classical", seed, horizon, lookback, experiment, ablation, reason=reason))
        except ImportError as exc:
            logging.warning("Skipping %s: %s", name, exc)
            frames.append(skipped_row(name, "classical", seed, horizon, lookback, experiment, ablation, f"missing dependency: {exc}"))
        except Exception as exc:
            logging.warning("Skipping %s: %s", name, exc)
            frames.append(skipped_row(name, "classical", seed, horizon, lookback, experiment, ablation, str(exc)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _run_hmm(Xtr, ytr, Xv, Xte, seed):
    if not _has_two_classes(ytr):
        raise ValueError("training labels contain one class")
    idx = np.arange(min(20, Xtr.shape[1]))
    try:
        hmm = importlib.import_module("hmmlearn.hmm")
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=seed)
        model.fit(Xtr[:, idx])
        post_tr = model.predict_proba(Xtr[:, idx])
        post_v = model.predict_proba(Xv[:, idx])
        post_te = model.predict_proba(Xte[:, idx])
        reason = ""
    except ImportError:
        model = GaussianMixture(n_components=2, covariance_type="diag", max_iter=200, random_state=seed)
        model.fit(Xtr[:, idx])
        post_tr = model.predict_proba(Xtr[:, idx])
        post_v = model.predict_proba(Xv[:, idx])
        post_te = model.predict_proba(Xte[:, idx])
        reason = "hmmlearn package missing; used sklearn GaussianMixture state fallback"
    state_rates = [ytr[post_tr[:, k] >= post_tr.max(axis=1)].mean() if (post_tr[:, k] >= post_tr.max(axis=1)).any() else 0 for k in range(2)]
    fragile_state = int(np.argmax(state_rates))
    return (post_tr[:, fragile_state], post_v[:, fragile_state], post_te[:, fragile_state]), reason
