from pandaskill.libs.performance_score.base_model import BaseModel
from pandaskill.libs.performance_score.percentile_mapper import PercentileMapper
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class PScoreModel(BaseModel):
    percentile_mapper = None
    
    def __init__(self, **kwargs):
        if "monotone_constraints" in kwargs and type(kwargs["monotone_constraints"]) == list:
            kwargs["monotone_constraints"] = {
                f"feature_{i}": constraint
                for i, constraint in enumerate(kwargs["monotone_constraints"])
            }
        super().__init__(model=XGBClassifier, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        X_normalized = self.scaler.fit_transform(X)
        X_normalized = pd.DataFrame(
            X_normalized,
            columns=[f"feature_{i}" for i in range(len(X[0]))]
        )
        self.model.fit(X_normalized, y)

        win_prob = self.model.predict_proba(X_normalized)[:, 1]
        percentile_mapper = PercentileMapper().train(win_prob)
        self.percentile_mapper = percentile_mapper

        return self
    
    def compute_performance_scores(self, X: np.ndarray) -> np.ndarray:
        win_probabilities = self.predict_proba(X)[:, 1]
        performance_scores = self.percentile_mapper.map(win_probabilities)            
        return performance_scores
    
    def compute_features_importance(self) -> np.ndarray:
        feature_importance_dict = self.model.get_booster().get_score(importance_type="gain")
        feature_importance_list = []
        for feature_name in self.model.feature_names_in_:
            feature_importance = feature_importance_dict.get(feature_name, 0)
            feature_importance_list.append(feature_importance)
        return np.array(feature_importance_list)
