import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from pandaskill.libs.performance_score.base_model import BaseModel

class MockModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model=LogisticRegression, **kwargs)

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y

@pytest.fixture
def base_model():
    return MockModel()

def test_fit(sample_data, base_model):
    X, y = sample_data
    base_model.fit(X, y)
    assert base_model.model is not None

def test_predict(sample_data, base_model):
    X, y = sample_data
    base_model.fit(X, y)
    predictions = base_model.predict(X)
    assert predictions.shape[0] == X.shape[0]

def test_predict_proba(sample_data, base_model):
    X, y = sample_data
    base_model.fit(X, y)
    proba = base_model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)

def test_score(sample_data, base_model):
    X, y = sample_data
    base_model.fit(X, y)
    score = base_model.score(X, y)
    assert 0 <= score <= 1

def test_compute_performance_scores(sample_data, base_model):
    X, y = sample_data
    base_model.fit(X, y)
    performance_scores = base_model.compute_performance_scores(X)
    assert performance_scores.shape[0] == X.shape[0]

def test_not_fitted_error_on_predict(sample_data, base_model):
    X, _ = sample_data
    with pytest.raises(NotFittedError):
        base_model.predict(X)

def test_not_fitted_error_on_predict_proba(sample_data, base_model):
    X, _ = sample_data
    with pytest.raises(NotFittedError):
        base_model.predict_proba(X)

def test_compute_features_importance(sample_data, base_model):
    X, y = sample_data
    base_model.fit(X, y)
    features_importance = base_model.compute_features_importance()
    
    assert features_importance.shape[0] == X.shape[1]
    assert np.isclose(np.abs(features_importance).sum(), 1), "Feature importances should sum up to 1"
    assert np.any(features_importance < 0), "Some features should have negative importance"
    assert np.any(features_importance > 0), "Some features should have positive importance"

def test_compute_shap_values(sample_data, base_model):
    X, y = sample_data
    base_model.fit(X, y)
    explainer, shap_values = base_model.compute_shap_values(X)
    
    assert explainer is not None
    assert shap_values.shape == (X.shape[0], X.shape[1])