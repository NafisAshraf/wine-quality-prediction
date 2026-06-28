import json
import os
from pathlib import Path

from flask import Flask, render_template, request

from wine_quality.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

FEATURES = [
    ("fixed_acidity", "Fixed Acidity", "7.4"),
    ("volatile_acidity", "Volatile Acidity", "0.70"),
    ("citric_acid", "Citric Acid", "0.00"),
    ("residual_sugar", "Residual Sugar", "1.9"),
    ("chlorides", "Chlorides", "0.076"),
    ("free_sulfur_dioxide", "Free Sulfur Dioxide", "11.0"),
    ("total_sulfur_dioxide", "Total Sulfur Dioxide", "34.0"),
    ("density", "Density", "0.9978"),
    ("pH", "pH", "3.51"),
    ("sulphates", "Sulphates", "0.56"),
    ("alcohol", "Alcohol", "9.4"),
]


def load_metrics():
    metrics_path = Path("artifacts/model_evaluation/metrics.json")
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def model_ready():
    return Path("artifacts/model_trainer/model.joblib").exists()


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        features=FEATURES,
        metrics=load_metrics(),
        model_ready=model_ready(),
        form_values={},
        prediction=None,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_values = request.form.to_dict()
    try:
        values = [float(form_values[name]) for name, _, _ in FEATURES]
        prediction = PredictionPipeline().predict(values)
        result = round(float(prediction[0]), 3)
        return render_template(
            "index.html",
            features=FEATURES,
            metrics=load_metrics(),
            model_ready=model_ready(),
            form_values=form_values,
            prediction=result,
            error=None,
        )
    except Exception as e:
        return render_template(
            "index.html",
            features=FEATURES,
            metrics=load_metrics(),
            model_ready=model_ready(),
            form_values=form_values,
            prediction=None,
            error=str(e),
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
