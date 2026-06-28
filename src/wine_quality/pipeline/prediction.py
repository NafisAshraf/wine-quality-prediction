from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))
        self.feature_names = [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ]

    def predict(self, data):
        prepared = np.array(data).reshape(1, 11)
        prepared = pd.DataFrame(prepared, columns=self.feature_names)
        return self.model.predict(prepared)
