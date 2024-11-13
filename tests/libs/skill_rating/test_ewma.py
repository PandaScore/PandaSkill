import pytest
import pandas as pd
from pandaskill.libs.skill_rating.ewma import compute_ewma_ratings, _compute_ewma_ratings_for_player

def test_compute_ewma_ratings():
    data = {
        'player_id': [1, 1, 1, 2, 2, 2],
        'game_id': [1, 2, 3, 2, 1, 3],
        'date': [
            '2021-01-01', '2021-01-02', '2021-01-03',
            '2021-01-02', '2021-01-01',  '2021-01-03'
        ],
        'performance_score': [10, 20, 30, 50, 40, 60]
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['game_id', 'player_id'])
    
    alpha = 0.5
    ratings_df = compute_ewma_ratings(df, alpha)
    
    expected_data = {
        'skill_rating_before': [0.0, 10.0, 15.0, 0.0, 40.0, 45.0],
        'skill_rating_after': [10.0, 15.0, 22.5, 40.0, 45.0, 52.5]
    }
    expected_df = pd.DataFrame(expected_data, index=df.index)
    
    pd.testing.assert_frame_equal(ratings_df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test__compute_ewma_ratings_for_player():
    data = {
        'performance_score': [10, 20, 30],
        'date': ['2021-01-01', '2021-01-02', '2021-01-03']
    }
    player_game_performance_score = pd.DataFrame(data)
    player_game_performance_score['date'] = pd.to_datetime(player_game_performance_score['date'])
    player_game_performance_score = player_game_performance_score.set_index('date')
    
    alpha = 0.5
    skill_ratings = _compute_ewma_ratings_for_player(player_game_performance_score, alpha)
    
    expected_data = {
        'skill_rating_before': [0.0, 10.0, 15.0],
        'skill_rating_after': [10.0, 15.0, 22.5]
    }
    expected_index = player_game_performance_score.index
    expected_df = pd.DataFrame(expected_data, index=expected_index)
    
    pd.testing.assert_frame_equal(skill_ratings, expected_df)

if __name__ == '__main__':
    pytest.main([__file__])
