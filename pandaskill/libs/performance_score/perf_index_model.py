from pandaskill.libs.performance_score.base_model import BaseModel
from pandaskill.libs.performance_score.percentile_mapper import PercentileMapper
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

class PerformanceIndexModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model=RandomForestClassifier, **kwargs)

    def train_random_forest_classifier(self, X: np.ndarray, y: np.ndarray) -> None:
        X_normalized = self.scaler.fit_transform(X)

        self.model.fit(X_normalized, y)

        result = permutation_importance(
            self.model, X_normalized, y, n_repeats=5, random_state=42
        )
        self.weights = result.importances_mean
        self.weights = self.weights / np.sum(self.weights)

    def learn_data_histograms(self, X: np.ndarray, y: np.ndarray) -> None:
        nb_features = len(X[0])
        self.percentile_mapper_list = []
        for feature_index in range(nb_features):
            self.percentile_mapper_list.append(
                PercentileMapper().train(X[:, feature_index])
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        self.train_random_forest_classifier(X, y)
        self.learn_data_histograms(X, y)
        return self
    
    def compute_features_importance(self) -> np.ndarray:
        return self.model.feature_importances_
    
    def compute_performance_scores(self, X: np.ndarray) -> np.ndarray:
        num_samples = X.shape[0]
        num_features = X.shape[1]
        percentiles = np.zeros((num_samples, num_features))
        for i in range(num_features):
            percentiles[:, i] = self.percentile_mapper_list[i].map(X[:, i])
        scores = np.dot(percentiles, self.weights)
        return scores