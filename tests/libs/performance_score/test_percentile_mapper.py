
from pandaskill.libs.performance_score.percentile_mapper import PercentileMapper
import numpy as np
import pytest

@pytest.fixture
def probabilities():
    return np.array([0.00001 * i for i in range(1, 100001)])

@pytest.fixture
def percentile_mapper():
    return PercentileMapper()

def test_percentile_mapper_train(probabilities, percentile_mapper):
    trained_mapper = percentile_mapper.train(probabilities)
    assert np.array_equal(trained_mapper.reference_probabilities, probabilities)

def test_percentile_mapper_map_values(probabilities, percentile_mapper):
    percentile_mapper.train(probabilities)
    new_probabilities = np.array([0.15, 0.25, 0.35])
    expected_percentiles = np.array([15, 25, 35])
    mapped_scores = percentile_mapper.map(new_probabilities)
    
    assert np.allclose(mapped_scores, expected_percentiles, rtol=1e-3)

def test_percentile_mapper_empty_input(percentile_mapper):
    percentile_mapper.train(np.array([0.10, 0.20, 0.30]))
    empty_scores = np.array([])
    
    mapped_scores = percentile_mapper.map(empty_scores)
    
    assert mapped_scores.size == 0
