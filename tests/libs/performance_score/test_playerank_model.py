
from pandaskill.libs.performance_score.playerank_model import PlayerankModel
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y

@pytest.fixture
def playerank_model():
    return PlayerankModel()

def test_playerank_model_initialization(playerank_model):
    assert playerank_model.model is not None

def test_playerank_model_fit(sample_data, playerank_model):
    X, y = sample_data
    playerank_model.fit(X, y)
    assert playerank_model.model is not None

def test_playerank_model_compute_performance_scores(sample_data, playerank_model, mocker):
    X, y = sample_data
    playerank_model.fit(X, y)

    mocker.patch.object(playerank_model, 'compute_features_importance', return_value=np.ones(X.shape[1]))
    performance_scores = playerank_model.compute_performance_scores(X)

    assert performance_scores.shape[0] == X.shape[0]
    assert np.all(performance_scores >= 0) and np.all(performance_scores <= 100)

def test_not_fitted_error_on_compute_performance_scores(sample_data, playerank_model):
    X, _ = sample_data
    
    with pytest.raises(NotFittedError):
        playerank_model.compute_performance_scores(X)