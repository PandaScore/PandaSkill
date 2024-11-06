import numpy as np
from scipy import stats

class PercentileMapper():
    def train(self, probabilities):
        self.reference_probabilities = probabilities
        return self

    def map(self, probabilities):
        performance_scores = np.array([
            stats.percentileofscore(self.reference_probabilities, probability) 
            for probability in probabilities
        ])
        return performance_scores