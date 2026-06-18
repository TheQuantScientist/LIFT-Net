import copy
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features_labels import flatten_graph_predictions
from .metrics import best_f1_threshold


class TCNBlock(nn.Module):
    def __init__(self, d_model, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if y.size(1) > x.size(1):
            y = y[:, : x.size(1), :]
        return self.norm(x + self.drop(self.act(y)))


class LiftNet(nn.Module):
    def __init__(self, n_features, d_model=64, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([TCNBlock(d_model, d, dropout) for d in [1, 2, 4]])
        self.graph_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, x, adj):
        b, n, t, f = x.shape
        h = self.proj(x.reshape(b * n, t, f))
        for block in self.blocks:
            h = block(h)
        h = h[:, -1, :].reshape(b, n, -1)
        neigh = torch.bmm(adj, h)
        h = self.graph_norm(h + neigh)
        return self.head(h).squeeze(-1)


def _adjacency_for_sample(x, feature_cols, graph_type):
    n = x.shape[0]
    if graph_type == "identity":
        return np.eye(n, dtype=np.float32)
    if graph_type == "fully_connected":
        return np.ones((n, n), dtype=np.float32) / max(n, 1)
    feat = "log_ILLIQ" if graph_type == "liquidity_corr" else "log_return_1d"
    if feat not in feature_cols:
        return np.eye(n, dtype=np.float32)
    idx = feature_cols.index(feat)
    mat = x[:, -30:, idx]
    if mat.shape[1] < 5:
        return np.eye(n, dtype=np.float32)
    std = np.nanstd(mat, axis=1)
    if np.isfinite(std).sum() < 2 or np.nanmax(std) <= 1e-8:
        return np.eye(n, dtype=np.float32)
    mat = mat.copy()
    mat[~np.isfinite(mat)] = 0.0
    corr = np.eye(n, dtype=np.float32)
    valid = std > 1e-8
    corr[np.ix_(valid, valid)] = np.corrcoef(mat[valid])
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, 0, None)
    np.fill_diagonal(corr, 1.0)
    return (corr / np.maximum(corr.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _make_adj(X, feature_cols, graph_type):
    return np.stack([_adjacency_for_sample(x, feature_cols, graph_type) for x in X]).astype(np.float32)


def _predict(model, X, A, device, batch_size):
    if len(X) == 0:
        return np.empty((0, X.shape[1] if X.ndim > 1 else 0))
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(A, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for xb, ab in loader:
            logits = model(xb.to(device), ab.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_liftnet(datasets, feature_cols, symbols, config, seed, horizon, lookback, experiment="main", ablation="full", graph_type="liquidity_corr"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    exp = config["experiment"]
    Xtr, ytr, amtr, lmtr, mtr = datasets["train"]
    Xv, yv, amv, lmv, mv = datasets["val"]
    Xte, yte, amte, lmte, mte = datasets["test"]
    if len(Xtr) == 0 or len(Xv) == 0:
        from .models import skipped_row
        return skipped_row("LIFT-Net", "proposed", seed, horizon, lookback, experiment, ablation, "empty graph train/val data", graph_type)
    Atr, Av, Ate = [_make_adj(X, feature_cols, graph_type) for X in [Xtr, Xv, Xte]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiftNet(len(feature_cols)).to(device)
    labels = ytr[lmtr > 0]
    pos = max(float((labels == 1).sum()), 1.0)
    neg = max(float((labels == 0).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=exp["learning_rate"], weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    ds = TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Atr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32), torch.tensor(lmtr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=exp["batch_size"], shuffle=True)
    best_state, best_score, wait = None, -np.inf, 0
    for epoch in range(int(exp["epochs"])):
        model.train()
        for xb, ab, yb, mb in loader:
            xb, ab, yb, mb = xb.to(device), ab.to(device), yb.to(device), mb.to(device)
            logits = model(xb, ab)
            loss = (loss_fn(logits, yb) * mb).sum() / mb.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = _predict(model, Xv, Av, device, exp["batch_size"])
        valid = lmv > 0
        try:
            from sklearn.metrics import average_precision_score
            score = average_precision_score(yv[valid], pv[valid]) if len(np.unique(yv[valid])) > 1 else 0.0
        except Exception:
            score = 0.0
        if score > best_score:
            best_score, wait, best_state = score, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= int(exp["patience"]):
                break
    if best_state:
        model.load_state_dict(best_state)
    ptr, pv, pte = [_predict(model, X, A, device, exp["batch_size"]) for X, A in [(Xtr, Atr), (Xv, Av), (Xte, Ate)]]
    valid_v = lmv > 0
    threshold = best_f1_threshold(yv[valid_v], pv[valid_v]) if valid_v.sum() else 0.5
    frames = []
    for metas, probs, split in [(mtr, ptr, "train"), (mv, pv, "val"), (mte, pte, "test")]:
        preds = (probs >= threshold).astype(int)
        frames.append(flatten_graph_predictions(symbols, metas, probs, preds, threshold, "LIFT-Net", seed, horizon, lookback, experiment, ablation, graph_type))
    out = pd.concat(frames, ignore_index=True)
    logging.info("LIFT-Net %s/%s H=%s seed=%s val AUPRC=%.4f", ablation, graph_type, horizon, seed, best_score)
    return out
