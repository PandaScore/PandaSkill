from abc import ABC
import numpy as np
import shap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Tuple

class BaseModel(BaseEstimator, ClassifierMixin, ABC):
    def __init__(self, model: Any, **kwargs: Any) -> None:
        self.scaler = MinMaxScaler()
        self.model = model(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        X_normalized = self.scaler.fit_transform(X)
        self.model.fit(X_normalized, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_normalized = self.scaler.transform(X)
        return self.model.predict(X_normalized)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_normalized = self.scaler.transform(X)
        return self.model.predict_proba(X_normalized)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        X_normalized = self.scaler.transform(X)
        return self.model.score(X_normalized, y)
    
    def compute_performance_scores(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        win_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return win_probabilities

    def compute_features_importance(self) -> np.ndarray:
        features_importance = self.model.coef_[0]
        features_importance = features_importance / np.abs(features_importance).sum()
        return features_importance
    
    def compute_shap_values(self, X: np.ndarray) -> Tuple[shap.Explainer, np.ndarray]:
        X_normalized = self.scaler.transform(X)
        explainer = shap.Explainer(self.model, X_normalized)
        shap_values = explainer.shap_values(X_normalized)
        return explainer, shap_values
