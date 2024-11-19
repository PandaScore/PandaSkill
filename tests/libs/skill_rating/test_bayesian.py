import pytest
import numpy as np
import pandas as pd
from pandaskill.libs.skill_rating.bayesian import (
    compute_bayesian_ratings,
    combine_contextual_and_meta_ratings,
    _initialize_ratings,
    _reset_rating_sigma_when_region_change,
    _compute_ratings_before_game,
    _compute_ratings_after_game,
    _compute_ratings_updates,
    _compute_ratings_after_contextual_game,
    _compute_ratings_after_meta_game,
    _apply_rating_updates,
    _instantiate_rater_model,
    lower_bound_rating,
    PlackettLuce,
    TrueSkill
)


DEFAULT_MU = 25.0
DEFAULT_SIGMA = 25 / 3

player_ids = [1, 2, 3, 4, 5]
regions = ['NA', 'EU']
model = PlackettLuce()

def test_initialize_ratings():
    df = pd.DataFrame([
            [1, 1, "NA"],
            [1, 2, "NA"],
            [1, 3, "EU"],
            [1, 4, "EU"],
            [1, 5, "EU"],
        ], columns=["game_id", "player_id", "region"]
    )
    df = df.set_index(["game_id", "player_id"])
    skill_ratings, region_ratings = _initialize_ratings(df)

    for player_id in player_ids:
        assert player_id in skill_ratings
        assert skill_ratings[player_id]['mu'] == DEFAULT_MU
        assert skill_ratings[player_id]['sigma'] == DEFAULT_SIGMA
        assert skill_ratings[player_id]['lower_bound'] == lower_bound_rating(DEFAULT_MU, DEFAULT_SIGMA)

    for region in regions:
        assert region in region_ratings
        assert region_ratings[region]['mu'] == DEFAULT_MU
        assert region_ratings[region]['sigma'] == DEFAULT_SIGMA
        assert region_ratings[region]['lower_bound'] == lower_bound_rating(DEFAULT_MU, DEFAULT_SIGMA)

def test__instantiate_rater_model():
    assert type(_instantiate_rater_model("openskill") == PlackettLuce)
    assert type(_instantiate_rater_model("trueskill") == TrueSkill)
    

def test_lower_bound():
    mu = 30.0
    sigma = 5.0
    expected_lower_bound = mu - 3 * sigma
    assert lower_bound_rating(mu, sigma) == expected_lower_bound

def test_reset_rating_sigma_when_region_change():
    player_contextual_ratings = [
        {'mu': 27.0, 'sigma': 8.0, 'lower_bound': lower_bound_rating(27.0, 8.0)},
        {'mu': 25.0, 'sigma': 7.0, 'lower_bound': lower_bound_rating(25.0, 7.0)}
    ]
    region_changes = [True, False]
    updated_ratings = _reset_rating_sigma_when_region_change(player_contextual_ratings, region_changes)

    assert updated_ratings[0]['sigma'] == DEFAULT_SIGMA
    assert updated_ratings[0]['lower_bound'] == lower_bound_rating(updated_ratings[0]['mu'], DEFAULT_SIGMA)

    assert updated_ratings[1]['sigma'] == 7.0
    assert updated_ratings[1]['lower_bound'] == player_contextual_ratings[1]['lower_bound']

def test_compute_ratings_before_game():
    player_contextual_ratings = [
        {'mu': 27.0, 'sigma': 8.0, 'lower_bound': lower_bound_rating(27.0, 8.0)},
        {'mu': 25.0, 'sigma': 7.0, 'lower_bound': lower_bound_rating(25.0, 7.0)}
    ]
    player_meta_ratings = [
        {'mu': 26.0, 'sigma': 6.0, 'lower_bound': lower_bound_rating(26.0, 6.0)},
        {'mu': 24.0, 'sigma': 5.0, 'lower_bound': lower_bound_rating(24.0, 5.0)}
    ]

    # Test for meta game
    meta_game = True
    ratings_before = _compute_ratings_before_game(
        player_contextual_ratings, player_meta_ratings, model, meta_game
    )
    expected_mu_0 = player_meta_ratings[0]['mu'] + player_contextual_ratings[0]['lower_bound']
    assert ratings_before[0].mu == expected_mu_0
    assert ratings_before[0].sigma == player_meta_ratings[0]['sigma']

    # Test for non-meta game
    meta_game = False
    ratings_before = _compute_ratings_before_game(
        player_contextual_ratings, player_meta_ratings, model, meta_game
    )
    assert ratings_before[0].mu == player_contextual_ratings[0]['mu']
    assert ratings_before[0].sigma == player_contextual_ratings[0]['sigma']

def test_combine_contextual_and_meta_ratings():
    overall_rating_mu, overall_rating_sigma = combine_contextual_and_meta_ratings(27.0, 8.0, 26.0, 6.0)
    expected_mu = 53.0
    expected_sigma = np.sqrt(8.0**2 + 6.0**2)
    assert overall_rating_mu == expected_mu
    assert overall_rating_sigma == expected_sigma

def test_apply_rating_updates():
    rating_updates = [
        [1, 'NA', {}, {}, {'mu': 28.0, 'sigma': 7.0}, {'mu': 26.0, 'sigma': 6.0}],
        [2, 'EU', {}, {}, {'mu': 27.0, 'sigma': 8.0}, {'mu': 25.0, 'sigma': 5.0}],
    ]
    contextual_ratings_dict = {1: {}, 2: {}}
    meta_ratings_dict = {'NA': {}, 'EU': {}}
    updated_contextual_ratings, updated_meta_ratings = _apply_rating_updates(
        rating_updates, contextual_ratings_dict, meta_ratings_dict
    )

    assert updated_contextual_ratings[1]['mu'] == 28.0
    assert updated_contextual_ratings[1]['sigma'] == 7.0
    assert updated_contextual_ratings[2]['mu'] == 27.0
    assert updated_contextual_ratings[2]['sigma'] == 8.0
    assert updated_meta_ratings['NA']['mu'] == 26.0
    assert updated_meta_ratings['NA']['sigma'] == 6.0
    assert updated_meta_ratings['EU']['mu'] == 25.0
    assert updated_meta_ratings['EU']['sigma'] == 5.0

def test_compute_ratings_after_game_ffa():
    ratings_before_game = [model.rating(DEFAULT_MU, DEFAULT_SIGMA), model.rating(DEFAULT_MU, DEFAULT_SIGMA)]
    game_performance_scores = [1, 2]
    
    use_ffa_setting = True
    ratings_after_game = _compute_ratings_after_game(
        ratings_before_game, game_performance_scores, model, use_ffa_setting
    )

    assert len(ratings_after_game) == 2
    assert ratings_after_game[0].mu < ratings_before_game[0].mu
    assert ratings_after_game[0].sigma < ratings_before_game[0].sigma
    assert ratings_after_game[1].mu > ratings_before_game[1].mu
    assert ratings_after_game[1].sigma < ratings_before_game[1].sigma

def test_compute_ratings_after_game_team():
    ratings_before_game = [model.rating(DEFAULT_MU, DEFAULT_SIGMA) for _ in range(10)]
    game_performance_scores = [i for i in range(1, 11)] # not used
    use_ffa_setting = False

    ratings_after_game = _compute_ratings_after_game(
        ratings_before_game, game_performance_scores, model, use_ffa_setting
    )

    assert len(ratings_after_game) == 10
    assert np.all(np.array(ratings_after_game[:5]) == ratings_after_game[0])
    assert np.all(np.array(ratings_after_game[5:]) == ratings_after_game[-1])
    assert ratings_after_game[0].mu > ratings_before_game[0].mu
    assert ratings_after_game[0].sigma < ratings_before_game[0].sigma
    assert ratings_after_game[-1].mu < ratings_before_game[-1].mu
    assert ratings_after_game[-1].sigma < ratings_before_game[-1].sigma

def test_compute_ratings_updates_non_meta(mocker):
    ratings_after_meta_mock = mocker.patch(
        'pandaskill.libs.skill_rating.bayesian._compute_ratings_after_meta_game', 
        return_value=[]
    )
    ratings_after_contextual_mock = mocker.patch(
        'pandaskill.libs.skill_rating.bayesian._compute_ratings_after_contextual_game', 
        return_value=[]
    )

    _compute_ratings_updates([], [], [], [], [], meta_game=False)

    ratings_after_meta_mock.assert_not_called()
    ratings_after_contextual_mock.assert_called_once()

    _compute_ratings_updates([], [], [], [], [], meta_game=True)

    ratings_after_meta_mock.assert_called_once()
    
def test_compute_ratings_after_contextual_game():
    full_ratings_after_game = [model.rating(28.0, 7.0), model.rating(26.0, 6.0)]
    player_ids_in_game = [1, 2]
    player_regions_in_game = ['NA', 'EU']
    contextual_ratings_in_game = [
        p1_contextual_rating := {'mu': 27.0, 'sigma': 8.0, 'lower_bound': lower_bound_rating(27.0, 8.0)},
        p2_contextual_rating := {'mu': 25.0, 'sigma': 7.0, 'lower_bound': lower_bound_rating(25.0, 7.0)}
    ]
    meta_ratings_in_game = [
        p1_meta_rating := {'mu': 26.0, 'sigma': 6.0, 'lower_bound': lower_bound_rating(26.0, 6.0)},
        p2_meta_rating := {'mu': 24.0, 'sigma': 5.0, 'lower_bound': lower_bound_rating(24.0, 5.0)}
    ]
    rating_updates = _compute_ratings_after_contextual_game(
        full_ratings_after_game, player_ids_in_game, player_regions_in_game,
        contextual_ratings_in_game, meta_ratings_in_game
    )

    expected_p1_contextual_rating_after_game = {'mu': 28.0, 'sigma': 7.0, 'lower_bound': lower_bound_rating(28.0, 7.0)}  
    expected_p1_meta_rating_after_game = p1_meta_rating
    expected_p2_contextual_rating_after_game = {'mu': 26.0, 'sigma': 6.0, 'lower_bound': lower_bound_rating(26.0, 6.0)}
    expected_p2_meta_rating_after_game = p2_meta_rating

    expected_rating_updates = [
        [1, 'NA', p1_contextual_rating, p1_meta_rating, expected_p1_contextual_rating_after_game, expected_p1_meta_rating_after_game],
        [2, 'EU', p2_contextual_rating, p2_meta_rating, expected_p2_contextual_rating_after_game, expected_p2_meta_rating_after_game]
    ]

    assert len(rating_updates) == len(expected_rating_updates)
    for i, update in enumerate(rating_updates):
        assert update == expected_rating_updates[i]

def test_compute_ratings_after_meta_game():
    full_ratings_after_game = [model.rating(28.0, 7.0), model.rating(26.0, 6.0), model.rating(24.0, 5.0)]
    player_ids_in_game = [1, 2, 3]
    player_regions_in_game = ['NA', 'EU', 'EU']
    contextual_ratings_in_game = [
        p1_contextual_rating := {'mu': 27.0, 'sigma': 8.0, 'lower_bound': lower_bound_rating(27.0, 8.0)},
        p2_contextual_rating := {'mu': 25.0, 'sigma': 7.0, 'lower_bound': lower_bound_rating(25.0, 7.0)},
        p3_contextual_rating := {'mu': 23.0, 'sigma': 6.0, 'lower_bound': lower_bound_rating(23.0, 6.0)}
    ]
    meta_ratings_in_game = [
        p1_meta_rating := {'mu': 26.0, 'sigma': 6.0, 'lower_bound': lower_bound_rating(26.0, 6.0)},
        p2_meta_rating := {'mu': 24.0, 'sigma': 5.0, 'lower_bound': lower_bound_rating(24.0, 5.0)},
        p3_meta_rating := {'mu': 22.0, 'sigma': 4.0, 'lower_bound': lower_bound_rating(22.0, 4.0)}
    ]
    rating_updates = _compute_ratings_after_meta_game(
        full_ratings_after_game, player_ids_in_game, player_regions_in_game,
        contextual_ratings_in_game, meta_ratings_in_game
    )
    eu_region_mu_after_game = (22.0 + 19.0) / 2
    eu_region_sigma_after_game = np.sqrt((6.0**2 + 5.0**2) / 2)
    eu_meta_rating = {'mu': eu_region_mu_after_game, 'sigma': eu_region_sigma_after_game, 'lower_bound': lower_bound_rating(eu_region_mu_after_game, eu_region_sigma_after_game)}

    expected_p1_contextual_rating_after_game = p1_contextual_rating
    expected_p1_meta_rating_after_game = {'mu': 25.0, 'sigma': 7.0, 'lower_bound': lower_bound_rating(25.0, 7.0)}
    expected_p2_contextual_rating_after_game = p2_contextual_rating
    expected_p2_meta_rating_after_game = eu_meta_rating
    expected_p3_contextual_rating_after_game = p3_contextual_rating
    expected_p3_meta_rating_after_game = eu_meta_rating

    expected_rating_updates = [
        [1, 'NA', p1_contextual_rating, p1_meta_rating, expected_p1_contextual_rating_after_game, expected_p1_meta_rating_after_game],
        [2, 'EU', p2_contextual_rating, p2_meta_rating, expected_p2_contextual_rating_after_game, expected_p2_meta_rating_after_game],
        [3, 'EU', p3_contextual_rating, p3_meta_rating, expected_p3_contextual_rating_after_game, expected_p3_meta_rating_after_game]
    ]
    
    assert len(rating_updates) == len(expected_rating_updates)
    for i, update in enumerate(rating_updates):
        assert update == expected_rating_updates[i]

def test_compute_bayesian_ratings(mocker):
    data = {
        'game_id': [1, 1, 2, 2],
        'date': ['2021-01-01', '2021-01-01', '2021-01-02', '2021-01-02'],
        'player_id': [1, 2, 1, 3],
        'team_id': [1, 2, 1, 3],
        'region': ['NA', 'EU', 'NA', 'EU'],
        'win': [1, 0, 0, 1],
        'performance_score': [1, 2, 2, 1],
        'region_change': [False, False, False, True]
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['game_id', 'player_id'])

    p1_contextual_rating_before_game = {'mu': 27.0, 'sigma': 8.0, 'lower_bound': lower_bound_rating(27.0, 8.0)}
    p1_meta_rating_before_game = {'mu': 26.0, 'sigma': 6.0, 'lower_bound': lower_bound_rating(26.0, 6.0)}
    p1_contextual_rating_after_game = {'mu': 28.0, 'sigma': 7.0, 'lower_bound': lower_bound_rating(28.0, 7.0)}
    p1_meta_rating_after_game = {'mu': 26.0, 'sigma': 6.0, 'lower_bound': lower_bound_rating(26.0, 6.0)}
    p2_contextual_rating_before_game = {'mu': 25.0, 'sigma': 7.0, 'lower_bound': lower_bound_rating(25.0, 7.0)}
    p2_meta_rating_before_game = {'mu': 24.0, 'sigma': 5.0, 'lower_bound': lower_bound_rating(24.0, 5.0)}
    p2_contextual_rating_after_game = {'mu': 27.0, 'sigma': 8.0, 'lower_bound': lower_bound_rating(27.0, 8.0)}
    p2_meta_rating_after_game = {'mu': 24.0, 'sigma': 5.0, 'lower_bound': lower_bound_rating(24.0, 5.0)}

    compute_ratings_update_mock = mocker.patch(
        'pandaskill.libs.skill_rating.bayesian._compute_rating_updates_for_game', 
        return_value=[
            [1, 'NA', p1_contextual_rating_before_game, p1_meta_rating_before_game, p1_contextual_rating_after_game, p1_meta_rating_after_game],
            [2, 'EU', p2_contextual_rating_before_game, p2_meta_rating_before_game, p2_contextual_rating_after_game, p2_meta_rating_after_game]
        ]
    )

    use_ffa_setting = True
    use_meta_ratings = True
    rater_model = "openskill"
    skill_rating_updates_df = compute_bayesian_ratings(
        df, use_ffa_setting, use_meta_ratings, rater_model
    )

    assert compute_ratings_update_mock.call_count == 2

    first_call = compute_ratings_update_mock.call_args_list[0][0]
    assert first_call[0].equals(pd.Series(
        data=[pd.Timestamp("2021-01-01 00:00:00"), [1, 2], [1, 2], ["NA", "EU"], [False, False], [1, 2],  [1, 0]],
        index=['date', 'performance_score', 'player_id', 'region', 'region_change', 'team_id',  'win'],
        name=1
    ))
    assert list(first_call[1:3]) == [
        {
            i : {"mu": 25.0, "sigma": 25 / 3, "lower_bound": 0.0}
            for i in range(1, 4)
        }, 
        {
            'NA': {"mu": 25.0, "sigma": 25 / 3, "lower_bound": 0.0},
            'EU': {"mu": 25.0, "sigma": 25 / 3, "lower_bound": 0.0}
        }
    ]
    assert list(first_call[4:]) == [True, True]

    second_call = compute_ratings_update_mock.call_args_list[1][0]
    assert second_call[0].equals(pd.Series(
        data=[pd.Timestamp("2021-01-02 00:00:00"), [2, 1], [1, 3], ["NA", "EU"], [False, True], [1, 3],  [0, 1]],
        index=['date', 'performance_score', 'player_id', 'region', 'region_change', 'team_id',  'win'],
        name=2
    ))
    assert list(second_call[1:3]) == [
        {
            1: p1_contextual_rating_after_game,
            2: p2_contextual_rating_after_game, 
            3: {"mu": 25, "sigma": 25 / 3, "lower_bound": 0.0}
        }, 
        {
            'NA': p1_meta_rating_after_game,
            'EU': p2_meta_rating_after_game
        },
    ] 
    assert list(second_call[4:]) == [True, True]

    expected_p1_skill_rating_before_game_mu_sigma = combine_contextual_and_meta_ratings(
        p1_contextual_rating_before_game['mu'], p1_contextual_rating_before_game['sigma'],
        p1_meta_rating_before_game['mu'], p1_meta_rating_before_game['sigma']
    )
    expected_p1_skill_rating_before_game = lower_bound_rating(
        *expected_p1_skill_rating_before_game_mu_sigma
    )
    expected_p1_skill_rating_after_game_mu_sigma = combine_contextual_and_meta_ratings(
        p1_contextual_rating_after_game['mu'], p1_contextual_rating_after_game['sigma'],
        p1_meta_rating_after_game['mu'], p1_meta_rating_after_game['sigma']
    )
    expected_p1_skill_rating_after_game = lower_bound_rating(
        *expected_p1_skill_rating_after_game_mu_sigma
    )
    expected_p2_skill_rating_before_game_mu_sigma = combine_contextual_and_meta_ratings(
        p2_contextual_rating_before_game['mu'], p2_contextual_rating_before_game['sigma'],
        p2_meta_rating_before_game['mu'], p2_meta_rating_before_game['sigma']
    )
    expected_p2_skill_rating_before_game = lower_bound_rating(
        *expected_p2_skill_rating_before_game_mu_sigma
    )
    expected_p2_skill_rating_after_game_mu_sigma = combine_contextual_and_meta_ratings(
        p2_contextual_rating_after_game['mu'], p2_contextual_rating_after_game['sigma'],
        p2_meta_rating_after_game['mu'], p2_meta_rating_after_game['sigma']
    )
    expected_p2_skill_rating_after_game = lower_bound_rating(
        *expected_p2_skill_rating_after_game_mu_sigma
    )


    expected_df = pd.DataFrame(
        data=[
            [
                *p1_contextual_rating_before_game.values(), 
                *p1_meta_rating_before_game.values(), 
                *p1_contextual_rating_after_game.values(), 
                *p1_meta_rating_after_game.values(),
                *expected_p1_skill_rating_before_game_mu_sigma,
                expected_p1_skill_rating_before_game,
                *expected_p1_skill_rating_after_game_mu_sigma,
                expected_p1_skill_rating_after_game
            ], 
            [
                *p2_contextual_rating_before_game.values(), 
                *p2_meta_rating_before_game.values(), 
                *p2_contextual_rating_after_game.values(), 
                *p2_meta_rating_after_game.values(),
                *expected_p2_skill_rating_before_game_mu_sigma,
                expected_p2_skill_rating_before_game,
                *expected_p2_skill_rating_after_game_mu_sigma,
                expected_p2_skill_rating_after_game
            ]
        ] * 2,
        index=pd.MultiIndex.from_tuples(
            [(1, 1), (1, 2), (2, 1), (2, 2)], names=['game_id', 'player_id']
        ),
        columns=[
            "contextual_rating_before_mu", "contextual_rating_before_sigma", "contextual_rating_before", 
            "meta_rating_before_mu", "meta_rating_before_sigma", "meta_rating_before", 
            "contextual_rating_after_mu", "contextual_rating_after_sigma", "contextual_rating_after",
            "meta_rating_after_mu", "meta_rating_after_sigma", "meta_rating_after",
            "skill_rating_before_mu", "skill_rating_before_sigma", "skill_rating_before",
            "skill_rating_after_mu", "skill_rating_after_sigma", "skill_rating_after"
        ]
    )

    print("actual", skill_rating_updates_df.head())
    print("expected",expected_df.head())
    print("actual", skill_rating_updates_df.columns)
    print("expected",expected_df.columns)

    assert skill_rating_updates_df.equals(expected_df)

if __name__ == '__main__':
    pytest.main([__file__])