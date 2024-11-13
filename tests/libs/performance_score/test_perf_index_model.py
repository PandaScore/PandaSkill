import pytest
import numpy as np
from sklearn.datasets import make_classification
from pandaskill.libs.performance_score.perf_index_model import PerformanceIndexModel

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y

@pytest.fixture
def perf_index_model_fixture():
    return PerformanceIndexModel()

def test_perf_index_fit(sample_data, perf_index_model_fixture):
    X, y = sample_data
    perf_index_model_fixture.fit(X, y)
    assert perf_index_model_fixture.model is not None
    assert perf_index_model_fixture.percentile_mapper_list is not None

def test_perf_index_compute_performance_scores(sample_data, perf_index_model_fixture):
    X, y = sample_data
    perf_index_model_fixture.fit(X, y)
    performance_scores = perf_index_model_fixture.compute_performance_scores(X)

    assert performance_scores.shape[0] == X.shape[0]
    assert np.all(performance_scores >= 0) and np.all(performance_scores <= 100)

def test_perf_index_compute_features_importance(sample_data, perf_index_model_fixture):
    X, y = sample_data
    perf_index_model_fixture.fit(X, y)
    
    features_importance = perf_index_model_fixture.compute_features_importance()

    assert isinstance(features_importance, np.ndarray), "Feature importances should be a numpy array"
    assert features_importance.shape[0] == X.shape[1], "Feature importances should match the number of features"

def test_perf_index_learn_data_histograms(sample_data, perf_index_model_fixture):
    X, y = sample_data
    perf_index_model_fixture.fit(X, y)
    assert len(perf_index_model_fixture.percentile_mapper_list) == X.shape[1]

def test_perf_index_train_random_forest_classifier(sample_data, perf_index_model_fixture):
    X, y = sample_data
    perf_index_model_fixture.train_random_forest_classifier(X, y)
    assert perf_index_model_fixture.weights is not None
    assert np.all(perf_index_model_fixture.weights >= 0) and np.all(perf_index_model_fixture.weights <= 1)

def test_perf_index_compute_features_importance(sample_data, perf_index_model_fixture):
    X, y = sample_data
    perf_index_model_fixture.fit(X, y)
    
    features_importance = perf_index_model_fixture.compute_features_importance()

    assert isinstance(features_importance, np.ndarray), "Feature importances should be a numpy array"
    assert features_importance.shape[0] == X.shape[1], "Feature importances should match the number of features"

def test_perf_index_compute_performance_scores(sample_data, perf_index_model_fixture):
    X, y = sample_data
    perf_index_model_fixture.fit(X, y)
    performance_scores = perf_index_model_fixture.compute_performance_scores(X)

    assert performance_scores.shape[0] == X.shape[0]
    assert np.all(performance_scores >= 0) and np.all(performance_scores <= 100)