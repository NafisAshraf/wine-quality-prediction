import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wine_quality.entity.config_entity import ReportingConfig


class ProjectReporter:
    def __init__(self, config: ReportingConfig):
        self.config = config
        Path(self.config.image_dir).mkdir(parents=True, exist_ok=True)

    def create_visuals(self):
        data = pd.read_csv(self.config.data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        metrics = self._load_metrics()

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]
        predictions = model.predict(test_x)

        self._target_distribution(data)
        self._correlation_heatmap(data)
        self._actual_vs_predicted(test_y, predictions)
        self._residual_plot(test_y, predictions)
        self._metrics_summary(metrics)
        self._pipeline_diagram()

    def _load_metrics(self):
        with open(self.config.metrics_path) as f:
            return json.load(f)

    def _target_distribution(self, data):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        data[self.config.target_column].value_counts().sort_index().plot(
            kind="bar", color="#8b1e3f", ax=ax
        )
        ax.set_title("Wine Quality Distribution")
        ax.set_xlabel("Quality")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(Path(self.config.image_dir) / "quality_distribution.png", dpi=160)
        plt.close(fig)

    def _correlation_heatmap(self, data):
        corr = data.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Feature Correlation Heatmap")
        fig.tight_layout()
        fig.savefig(Path(self.config.image_dir) / "correlation_heatmap.png", dpi=160)
        plt.close(fig)

    def _actual_vs_predicted(self, actual, predicted):
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.scatter(actual, predicted, color="#2f6690", alpha=0.72)
        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())
        ax.plot([low, high], [low, high], color="#cf4d6f", linewidth=2)
        ax.set_title("Actual vs Predicted Quality")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        fig.tight_layout()
        fig.savefig(Path(self.config.image_dir) / "actual_vs_predicted.png", dpi=160)
        plt.close(fig)

    def _residual_plot(self, actual, predicted):
        residuals = actual - predicted
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.scatter(predicted, residuals, color="#3a7d44", alpha=0.72)
        ax.axhline(0, color="#202020", linewidth=1)
        ax.set_title("Prediction Residuals")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        fig.tight_layout()
        fig.savefig(Path(self.config.image_dir) / "residuals.png", dpi=160)
        plt.close(fig)

    def _metrics_summary(self, metrics):
        names = list(metrics.keys())
        values = [metrics[name] for name in names]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        bars = ax.bar(names, values, color=["#8b1e3f", "#2f6690", "#3a7d44"])
        ax.set_title("Model Evaluation Metrics")
        ax.set_ylabel("Score")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )
        fig.tight_layout()
        fig.savefig(Path(self.config.image_dir) / "metrics_summary.png", dpi=160)
        plt.close(fig)

    def _pipeline_diagram(self):
        labels = [
            "Ingest",
            "Validate",
            "Split",
            "Train",
            "Evaluate",
            "Predict",
        ]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.plot(x, np.zeros_like(x), color="#333333", linewidth=2, zorder=1)
        ax.scatter(x, np.zeros_like(x), s=900, color="#8b1e3f", zorder=2)
        for i, label in enumerate(labels):
            ax.text(i, 0, label, color="white", ha="center", va="center", fontsize=10)
        ax.set_ylim(-0.7, 0.7)
        ax.axis("off")
        ax.set_title("Training and Prediction Flow")
        fig.tight_layout()
        fig.savefig(Path(self.config.image_dir) / "pipeline_flow.png", dpi=160)
        plt.close(fig)
