from pandaskill.libs.performance_score.base_model import BaseModel
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

class PlayerankModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model=SVC, probability=True, **kwargs)

    def compute_performance_scores(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        weights = self.compute_features_importance()

        performance_scores = weights @ np.array(X_scaled).T

        minmax_scaler = MinMaxScaler()
        performance_scores = minmax_scaler.fit_transform(performance_scores.reshape(-1, 1)).flatten()
        performance_scores *= 100

        return performance_scores