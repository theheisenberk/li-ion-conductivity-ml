import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor

from src.utils import compute_regression_metrics, save_dataframe, setup_logger


def _spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation without requiring scipy."""
    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    mask = s_true.notna() & s_pred.notna()
    if mask.sum() == 0:
        return float("nan")
    r_true = s_true[mask].rank(method="average")
    r_pred = s_pred[mask].rank(method="average")
    return float(np.corrcoef(r_true, r_pred)[0, 1])


def fit_quantile_models_for_model(
    model_name: str,
    preds_filename: str,
    test_df: pd.DataFrame,
    optuna_results_dir: Path,
    qr_results_dir: Path,
    quantiles=(0.05, 0.5, 0.95),
    logger=None,
) -> None:
    """
    Post-hoc 1D quantile regression:
    Learn q(y | y_pred_point) using HistGradientBoostingRegressor with loss='quantile'.

    - Reads existing point predictions from `optuna_results_dir / preds_filename`
    - Fits separate quantile models for Q0.05, Q0.5, Q0.95
    - Saves quantile predictions and parity / interval plots under `qr_results_dir`.
    """
    pred_path = optuna_results_dir / preds_filename
    if not pred_path.is_file():
        msg = f"[{model_name}] predictions file not found: {pred_path}"
        print(msg)
        if logger:
            logger.warning(msg)
        return

    preds_df = pd.read_csv(pred_path)
    if "ID" not in preds_df.columns or "prediction" not in preds_df.columns:
        msg = f"[{model_name}] predictions file has wrong format (need columns 'ID' and 'prediction')."
        print(msg)
        if logger:
            logger.warning(msg)
        return

    # Join ground truth and predictions on ID
    merged = (
        test_df[["ID", "log10_sigma"]]
        .merge(preds_df[["ID", "prediction"]], on="ID", how="inner")
        .dropna(subset=["log10_sigma", "prediction"])
    )
    if merged.empty:
        msg = f"[{model_name}] no overlapping IDs between test_cleaned and {preds_filename}."
        print(msg)
        if logger:
            logger.warning(msg)
        return

    # Original point predictions (mean) from the model
    mean_pred = merged["prediction"].values
    X = mean_pred.reshape(-1, 1)
    y = merged["log10_sigma"].values

    msg = f"[{model_name}] fitting quantile regressors on {len(merged)} test samples..."
    print(msg)
    if logger:
        logger.info(msg)

    quantile_preds = {}
    for q in quantiles:
        reg = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            random_state=42,
        )
        reg.fit(X, y)
        quantile_preds[q] = reg.predict(X)

    if 0.05 in quantile_preds and 0.5 in quantile_preds and 0.95 in quantile_preds:
        lower = quantile_preds[0.05]
        median = quantile_preds[0.5]
        upper = quantile_preds[0.95]
        widths = upper - lower
        coverage = np.mean((y >= lower) & (y <= upper))
        mean_width = np.mean(widths)
        msg = (
            f"[{model_name}] 90% interval: "
            f"coverage={coverage*100:5.1f}% | mean width={mean_width:7.4f}"
        )
        print(msg)
        if logger:
            logger.info(msg)

        out_df = pd.DataFrame(
            {
                "ID": merged["ID"].values,
                "prediction_q05": lower,
                "prediction_q50": median,
                "prediction_q95": upper,
                "interval_width": widths,
            }
        )
        out_path = qr_results_dir / f"{model_name}_qr_predictions.csv"
        save_dataframe(out_df, str(out_path))
        msg = f"[{model_name}] saved quantile predictions to {out_path}"
        print(msg)
        if logger:
            logger.info(msg)

        # Compute metrics:
        # - mean prediction: to stay consistent with main pipeline
        # - median prediction: to see how QR median behaves
        metrics_mean = compute_regression_metrics(y, mean_pred)
        metrics_median = compute_regression_metrics(y, median)
        rho_mean = _spearman_rho(y, mean_pred)
        rho_median = _spearman_rho(y, median)
        msg_mean = (
            f"[{model_name}] mean metrics:   R2={metrics_mean['r2']:.4f}, "
            f"RMSE={metrics_mean['rmse']:.4f}, MAE={metrics_mean['mae']:.4f}, "
            f"rho_mean={rho_mean:.4f}"
        )
        msg_med = (
            f"[{model_name}] median metrics: R2={metrics_median['r2']:.4f}, "
            f"RMSE={metrics_median['rmse']:.4f}, MAE={metrics_median['mae']:.4f}, "
            f"rho_median={rho_median:.4f}"
        )
        print(msg_mean)
        print(msg_med)
        if logger:
            logger.info(msg_mean)
            logger.info(msg_med)

        # Parity plot:
        #   - scatter: mean prediction vs true
        #   - text box: R²_mean, R²_median, rho_mean
        parity_path = qr_results_dir / f"{model_name}_qr_parity_median.png"
        plt.figure(figsize=(8, 8))
        sns.set_theme(style="whitegrid")

        y_true_plot = np.asarray(y, dtype=float)
        y_mean_plot = np.asarray(mean_pred, dtype=float)
        mask_plot = np.isfinite(y_true_plot) & np.isfinite(y_mean_plot)
        y_true_plot = y_true_plot[mask_plot]
        y_mean_plot = y_mean_plot[mask_plot]

        ax = sns.scatterplot(
            x=y_true_plot,
            y=y_mean_plot,
            alpha=0.6,
            s=50,
            edgecolor="k",
            label="Mean prediction",
        )

        # y = x reference line
        min_raw = float(min(np.nanmin(y_true_plot), np.nanmin(y_mean_plot)))
        max_raw = float(max(np.nanmax(y_true_plot), np.nanmax(y_mean_plot)))
        pad = 0.05 * (max_raw - min_raw) if max_raw > min_raw else 1.0
        min_val = min_raw - pad
        max_val = max_raw + pad
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

        ax.set_xlabel(r"Actual log10($\sigma$)", fontsize=14)
        ax.set_ylabel(r"Predicted log10($\sigma$)", fontsize=14)
        ax.set_title(f"Test Set: {model_name} (mean prediction)", fontsize=16)

        text_lines = [
            rf"$R^2_{{mean}} = {metrics_mean['r2']:.3f}$",
            rf"$R^2_{{median}} = {metrics_median['r2']:.3f}$",
            rf"$\rho_{{mean}} = {rho_mean:.3f}$",
            rf"$\rho_{{median}} = {rho_median:.3f}$",
        ]
        ax.text(
            0.05,
            0.95,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.legend()
        plt.tight_layout()
        os.makedirs(parity_path.parent, exist_ok=True)
        plt.savefig(parity_path, dpi=300)
        plt.close()
        msg = f"[{model_name}] saved parity plot to {parity_path}"
        print(msg)
        if logger:
            logger.info(msg)

        # Uncertainty-style plot (similar to stage2_double_model_residual_quantile_uncertainty_test)
        # Sort by true value for a smooth curve
        order = np.argsort(y)
        y_sorted = y[order]
        mean_sorted = mean_pred[order]
        median_sorted = median[order]
        lower_sorted = lower[order]
        upper_sorted = upper[order]

        plt.figure(figsize=(10, 8))
        sns.set_theme(style="whitegrid")

        idx = np.arange(len(y_sorted))
        plt.fill_between(idx, lower_sorted, upper_sorted, color="skyblue", alpha=0.3, label="90% Prediction Interval")
        # Mean (point) prediction curve
        plt.plot(idx, mean_sorted, color="black", lw=2, linestyle="--", label="Mean Prediction")
        # Median (Q50) curve from quantile regression
        plt.plot(idx, median_sorted, color="blue", lw=2, label="Median Prediction (Q50)")
        plt.scatter(idx, y_sorted, color="red", s=20, label="True Values")

        plt.xlabel("Sample Index (sorted by true value)")
        plt.ylabel("log10($\\sigma$)")
        plt.title(
            f"Test Set: {model_name} - 90% Prediction Intervals (Post-hoc Quantile Regression)\n"
            f"Coverage: {coverage*100:.1f}% (target: 90%), "
            f"$R^2_{{mean}}$ = {metrics_mean['r2']:.3f}, "
            f"$R^2_{{median}}$ = {metrics_median['r2']:.3f}, "
            f"$\\rho_{{mean}}$ = {rho_mean:.3f}, "
            f"$\\rho_{{median}}$ = {rho_median:.3f}"
        )
        plt.legend()
        plt.tight_layout()

        unc_path = qr_results_dir / f"{model_name}_qr_uncertainty_test.png"
        os.makedirs(unc_path.parent, exist_ok=True)
        plt.savefig(unc_path, dpi=300)
        plt.close()
        msg = f"[{model_name}] saved uncertainty plot to {unc_path}"
        print(msg)
        if logger:
            logger.info(msg)
    else:
        msg = f"[{model_name}] missing some quantiles; no output written."
        print(msg)
        if logger:
            logger.warning(msg)


def main() -> None:
    project_root = Path(__file__).resolve().parent

    # Load test targets (Stage 2 uses log10_sigma as target)
    test_path = project_root / "data" / "processed" / "test_cleaned.csv"
    if not test_path.is_file():
        raise FileNotFoundError(f"test_cleaned.csv not found at {test_path}")

    test_df = pd.read_csv(test_path)
    if "log10_sigma" not in test_df.columns or "ID" not in test_df.columns:
        raise ValueError("test_cleaned.csv must contain 'ID' and 'log10_sigma' columns.")

    optuna_results_dir = project_root / "results" / "results_optuna"
    qr_results_dir = project_root / "results" / "results_quantile_regression"
    os.makedirs(qr_results_dir, exist_ok=True)

    # Logger dedicated to quantile regression post-processing
    log_path = qr_results_dir / "quantile_regression.log"
    logger = setup_logger("stage2_quantile_regression", log_file=str(log_path))

    # baseline_geometry, stage2_physics, residual, residual_geometry
    models = {
        "stage2_baseline_default_geometry": "stage2_baseline_default_geometry_optuna_predictions.csv",
        "stage2_physics": "stage2_physics_optuna_predictions.csv",
        "stage2_double_model_residual": "stage2_double_model_residual_optuna_predictions.csv",
        "stage2_double_model_residual_geometry": "stage2_double_model_residual_geometry_optuna_predictions.csv",
    }

    header = "Post-hoc quantile regression on existing Stage 2 predictions"
    subheader = "(1D QR: log10_sigma ~ f(point_prediction))"
    print(header)
    print(subheader)
    print("-" * 80)
    logger.info("=" * 80)
    logger.info(header)
    logger.info(subheader)
    logger.info("=" * 80)

    for name, fname in models.items():
        fit_quantile_models_for_model(name, fname, test_df, optuna_results_dir, qr_results_dir, logger=logger)


if __name__ == "__main__":
    main()

