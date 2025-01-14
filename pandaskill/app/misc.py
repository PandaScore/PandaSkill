import os
    
ARTIFACTS_DIR = os.path.join("pandaskill", "artifacts")

def compute_rating_lower_bound(rating_mu, rating_sigma):
    """ 
    Compute a single-value estimate of the rating modeled as a gaussian distribution.
    Effectively, it represents a 99.7% confidence that the rating of the player is higher that this value.
    In that sense, it's a conservative estimate of the player's rating. See the paper for more information.
    """
    return rating_mu - 3 * rating_sigma