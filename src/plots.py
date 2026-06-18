from pathlib import Path
import os

import pandas as pd


def make_all_plots(results_path, summary_path, figures_dir):
    cache_dir = Path(figures_dir).parent / ".mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve

    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    res = pd.read_parquet(results_path) if str(results_path).endswith(".parquet") else pd.read_csv(results_path)
    res = res[(res["row_type"] == "prediction") & (res["status"] == "ok")]
    if res.empty:
        return []
    saved = []
    try:
        leader = pd.read_excel(summary_path, sheet_name="leaderboard")
        plt.figure(figsize=(9, 5))
        top = leader[leader["split"].eq("test")].sort_values("AUPRC", ascending=False).head(15)
        sns.barplot(data=top, x="AUPRC", y="model", hue="horizon", dodge=False)
        plt.tight_layout()
        p = figures / "leaderboard_AUPRC.png"
        plt.savefig(p, dpi=160)
        plt.close()
        saved.append(p)
    except Exception:
        pass
    try:
        plt.figure(figsize=(7, 5))
        for model, g in res[(res["split"] == "test") & (res["experiment"] == "main")].groupby("model"):
            if len(g["y_true"].unique()) < 2:
                continue
            pr, rc, _ = precision_recall_curve(g["y_true"], g["y_prob"])
            plt.plot(rc, pr, label=model)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(fontsize=8)
        plt.tight_layout()
        p = figures / "pr_curves_main_models.png"
        plt.savefig(p, dpi=160)
        plt.close()
        saved.append(p)
    except Exception:
        pass
    try:
        plt.figure(figsize=(10, 5))
        ev = res.drop_duplicates(["date", "symbol"]).groupby("symbol")["fragile_event"].sum().sort_values(ascending=False)
        sns.barplot(x=ev.values, y=ev.index)
        plt.tight_layout()
        p = figures / "fragile_events_by_asset.png"
        plt.savefig(p, dpi=160)
        plt.close()
        saved.append(p)
    except Exception:
        pass
    try:
        plt.figure(figsize=(7, 4))
        dtn = pd.to_numeric(res["days_to_next_event"], errors="coerce").dropna()
        sns.histplot(dtn.clip(upper=60), bins=30)
        plt.tight_layout()
        p = figures / "lead_time_distribution.png"
        plt.savefig(p, dpi=160)
        plt.close()
        saved.append(p)
    except Exception:
        pass
    for sheet, fname, key in [
        ("graph_ablations", "graph_ablation_barplot.png", "graph_type"),
        ("feature_ablations", "feature_ablation_barplot.png", "ablation"),
    ]:
        try:
            df = pd.read_excel(summary_path, sheet_name=sheet)
            df = df[df["split"].eq("test")]
            if df.empty:
                continue
            plt.figure(figsize=(8, 4))
            sns.barplot(data=df, x=key, y="AUPRC")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            p = figures / fname
            plt.savefig(p, dpi=160)
            plt.close()
            saved.append(p)
        except Exception:
            pass
    return saved
