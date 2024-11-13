import pytest
import numpy as np
from sklearn.datasets import make_classification
from pandaskill.libs.performance_score.pscore_model import PScoreModel

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y

@pytest.fixture
def pscore_model_fixture():
    return PScoreModel()

def test_pscore_fit(sample_data, pscore_model_fixture):
    X, y = sample_data
    pscore_model_fixture.fit(X, y)
    assert pscore_model_fixture.model is not None
    assert pscore_model_fixture.percentile_mapper is not None

def test_pscore_compute_performance_scores(sample_data, pscore_model_fixture):
    X, y = sample_data
    pscore_model_fixture.fit(X, y)
    performance_scores = pscore_model_fixture.compute_performance_scores(X)

    assert performance_scores.shape[0] == X.shape[0]
    assert np.all(performance_scores >= 0) and np.all(performance_scores <= 100)

def test_pscore_compute_features_importance(sample_data, pscore_model_fixture):
    X, y = sample_data
    pscore_model_fixture.fit(X, y)
    
    features_importance = pscore_model_fixture.compute_features_importance()

    assert isinstance(features_importance, np.ndarray), "Feature importances should be a numpy array"
    assert features_importance.shape[0] == X.shape[1], "Feature importances should match the number of features"
