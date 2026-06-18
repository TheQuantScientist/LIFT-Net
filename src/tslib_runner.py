import importlib
import os
import sys
from pathlib import Path

import pandas as pd

from .models import skipped_row


def find_tslib_path(config=None):
    candidates = []
    if os.environ.get("TSLIB_PATH"):
        candidates.append(Path(os.environ["TSLIB_PATH"]))
    if config and config.get("tslib_path"):
        candidates.append(Path(config["tslib_path"]))
    candidates.extend([Path("external/Time-Series-Library"), Path("Time-Series-Library"), Path("source_models")])
    for path in candidates:
        if (path / "models").exists():
            return path.resolve()
    return None


def run_tslib_baselines(model_names, datasets, feature_cols, config, seed, horizon, lookback, experiment, ablation):
    path = find_tslib_path(config)
    frames = []
    if path is None:
        return pd.concat([
            skipped_row(m, "tslib", seed, horizon, lookback, experiment, ablation, "TSLib path not found")
            for m in model_names
        ], ignore_index=True)
    sys.path.insert(0, str(path))
    for name in model_names:
        model_file = path / "models" / f"{name}.py"
        if not model_file.exists():
            frames.append(skipped_row(name, "tslib", seed, horizon, lookback, experiment, ablation, f"{model_file} not found"))
            continue
        try:
            mod = importlib.import_module(f"models.{name}")
            if not hasattr(mod, "Model"):
                raise AttributeError("models.<name>.Model missing")
            frames.append(skipped_row(name, "tslib", seed, horizon, lookback, experiment, ablation, "local TSLib model found but constructor is task-specific; skipped by minimal wrapper"))
        except Exception as exc:
            frames.append(skipped_row(name, "tslib", seed, horizon, lookback, experiment, ablation, f"incompatible TSLib import: {exc}"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
