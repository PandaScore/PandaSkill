
import pandas as pd
from unittest.mock import patch, MagicMock

from pandaskill.libs.feature_extraction.event_features import *
from pandaskill.libs.feature_extraction.event_features import (
    _evaluate_deaths_worthlessness,
    _evaluate_deaths_worthlessness_for_game,
    _count_nb_worthless_deaths,
    _count_nb_free_kills,
    _compute_death_worth_ratios
)

def test_compute_neutral_objective_contest_features():
    stat_df = pd.DataFrame({
        "game_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "player_id": [1, 2, 3, 4, 3, 4, 1, 2],
        "team_id": [1, 1, 2, 2, 2, 2, 1, 1],
    })
    stat_df.set_index(["game_id", "player_id"], inplace=True)

    event_df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "game_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "event_type": [
            "player_kill", "drake_kill", "baron_nashor_kill", "rift_herald_kill",
            "drake_kill", "player_kill", "drake_kill", "player_kill"
        ],
        "killer_id": [
            1, 1, 3, 1,
            4, 3, 3, 3
        ],
        "killed_id": [
            3, None, None, None,
            None, 1, None, 1
        ],
        "assisting_player_ids": [
            [2], [], [], [2, 3], # 4th event is contested and won by team 1
            [1], [], [4], [] # 1st and won by team 2, 3rd not contested as only 1 team involved
        ],
    })
    event_df.set_index("id", inplace=True)


    objective_contest_winrate, objective_contest_loserate = compute_neutral_objective_contest_features(stat_df, event_df)

    expected_objective_contest_winrate = pd.Series([
        1/3, 1/3, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0 # mind player id are 3, 4, 1, 2 in game 2
    ])
    expected_objective_contest_winrate = expected_objective_contest_winrate.set_axis(stat_df.index, axis=0)    
    pd.testing.assert_series_equal(objective_contest_winrate, expected_objective_contest_winrate)

    expected_objective_contest_loserate = pd.Series([
        0.0, 0.0, 1/3, 0.0,
        0.0, 0.0, 0.5, 0.0 # mind player id are 3, 4, 1, 2 in game 2
    ])
    expected_objective_contest_loserate = expected_objective_contest_loserate.set_axis(stat_df.index, axis=0)
    pd.testing.assert_series_equal(objective_contest_loserate, expected_objective_contest_loserate)

def test_compute_kill_death_value_features():
    stat_df = pd.DataFrame({
        "game_id": (game_id_list := [1, 1, 1, 1, 2, 2, 2, 2]),
        "player_id": (player_id_list := [1, 2, 3, 4, 3, 4, 1, 2]),
        "team_id": [1, 1, 2, 2, 2, 2, 1, 1],
        "player_kills": [3, 0, 2, 0, 0, 3, 1, 1],
        "player_assists": [0, 1, 0, 0, 1, 0, 0, 1],
        "player_deaths": [2, 0, 3, 0, 2, 0, 1, 2],
        "total_kills": [5] * 4 + [5] * 4,
    })
    stat_df.set_index(["game_id", "player_id"], inplace=True)

    event_df = pd.DataFrame(
        [
            [1, 1, "player_kill", 60, 1, 3, [], 1, 2], # solo kill (worthless) 
            [2, 1, "player_kill", 120, 1, 3, [2], 1, 2], # assist doesn't change anything (worthless)
            [3, 1, "drake_kill", 151, 4, 0, [], 2, 0], # objective taken into account for death below
            [4, 1, "player_kill", 180, 1, 3, [], 1, 2], # objective in window (worth)
            [5, 1, "player_kill", 220, 3, 1, [], 2, 1], # out of window (worthless)
            [6, 1, "player_kill", 221, 3, 1, [], 2, 1], # objective taken (worth)
            [7, 1, "baron_nashor_kill", 250, 2, 0, [], 1, 0], # objective taken into account for death above

            [8, 2, "player_kill", 360, 1, 3, [2], 1, 2], # team fight (worth)
            [9, 2, "player_kill", 362, 4, 1, [3], 2, 1], # team fight (worth)
            [10, 2, "player_kill", 363, 4, 2, [], 2, 1], # team fight (worth)
            [11, 2, "player_kill", 500, 4, 2, [], 2, 1], # revenged kill next kill (worth)
            [12, 2, "player_kill", 501, 2, 3, [], 1, 2], # revenge kill, is worthless for 3 (worthless)
        ],
        columns=[
            "id", "game_id", "event_type", "timestamp", "killer_id", "killed_id", "assisting_player_ids",
            "killer_team_id", "killed_team_id"
        ]
    )

    worthless_death_ratio, free_kill_ratio, worthless_death_total_ratio, free_kill_total_ratio = (
        compute_kill_death_value_features(stat_df, event_df, window=30)
    )

    expected_worthless_death_ratio = pd.Series([1/2, 0.0, 2/3, 0.0, 1/2, 0.0, 0.0, 0.0])
    expected_worthless_death_ratio = expected_worthless_death_ratio.set_axis(stat_df.index, axis=0)
    pd.testing.assert_series_equal(worthless_death_ratio, expected_worthless_death_ratio)

    expected_free_kill_ratio = pd.Series([2/3, 1/1, 1/2, 0.0, 0.0, 0.0, 0.0, 1/2])
    expected_free_kill_ratio = expected_free_kill_ratio.set_axis(stat_df.index, axis=0)
    pd.testing.assert_series_equal(free_kill_ratio, expected_free_kill_ratio)

    expected_worthless_death_total_ratio = pd.Series([1/5, 0.0, 2/5, 0.0, 1/5, 0.0, 0.0, 0.0])
    expected_worthless_death_total_ratio = expected_worthless_death_total_ratio.set_axis(stat_df.index, axis=0)
    pd.testing.assert_series_equal(worthless_death_total_ratio, expected_worthless_death_total_ratio)

    expected_free_kill_total_ratio = pd.Series([2/5, 1/5, 1/5, 0.0, 0.0, 0.0, 0.0, 1/5])
    expected_free_kill_total_ratio = expected_free_kill_total_ratio.set_axis(stat_df.index, axis=0)
    pd.testing.assert_series_equal(free_kill_total_ratio, expected_free_kill_total_ratio)


def test__evaluate_deaths_worthlessness():  
    event_df = pd.DataFrame(
        [
            [1, 1, "player_kill"],
            [2, 1, "non_player_kill"],
            [3, 2, "non_player_kill"],
            [4, 2, "player_kill"],
        ],
        columns=["id", "game_id", "event_type"]
    )  
    mock_pool = MagicMock()
    mock_pool.imap.return_value = [
        pd.Series([False], index=[1], name='death_is_worth'),
        pd.Series([True], index=[4], name='death_is_worth')
    ]
    mock_pool.__enter__.return_value = mock_pool

    with patch('multiprocessing.Pool', return_value=mock_pool):
        with patch('multiprocessing.cpu_count', return_value=2):
            death_is_worth = _evaluate_deaths_worthlessness(event_df, window=30)

    pd.testing.assert_series_equal(
        death_is_worth, 
        pd.Series(
            [False, True],
            index=[1, 4],
            name='death_is_worth'
        )
    )

    mock_pool.imap.assert_called_once()

def test__evaluate_deaths_worthlessness_for_game():
    event_df = pd.DataFrame(
        [
            [1, 1, "player_kill", 60, 1, 3, [], 1, 2], # solo kill (worthless) 
            [2, 1, "player_kill", 120, 1, 3, [2], 1, 2], # assist doesn't change anything (worthless)
            [3, 1, "drake_kill", 151, 4, 0, [], 2, 0], # objective taken into account for death below
            [4, 1, "player_kill", 180, 1, 3, [], 1, 2], # objective in window (worth)
            [5, 1, "player_kill", 220, 3, 1, [], 2, 1], # out of window (worthless)
            [6, 1, "player_kill", 221, 3, 1, [], 2, 1], # objective taken (worth)
            [7, 1, "baron_nashor_kill", 250, 2, 0, [], 1, 0], # objective taken into account for death above
            [8, 1, "player_kill", 360, 1, 3, [2], 1, 2], # team fight (worth)
            [9, 1, "player_kill", 362, 4, 1, [3], 2, 1], # team fight (worth)
            [10, 1, "player_kill", 363, 4, 2, [], 2, 1], # team fight (worth)
            [11, 1, "player_kill", 500, 4, 2, [], 2, 1], # revenged kill next kill (worth)
            [12, 1, "player_kill", 501, 2, 3, [], 1, 2], # revenge kill, is worthless for 3 (worthless)
        ],
        columns=[
            "id", "game_id", "event_type", "timestamp", "killer_id", "killed_id", "assisting_player_ids",
            "killer_team_id", "killed_team_id"
        ]
    )
    window = 30

    event_death_is_worth = _evaluate_deaths_worthlessness_for_game((event_df, window))

    expected_event_death_is_worth = pd.Series([
        True, True, None, False, True, False, None, False, False, False, False, True
    ])

    pd.testing.assert_series_equal(event_death_is_worth, expected_event_death_is_worth)

def test__count_nb_worthless_deaths():
    event_df = pd.DataFrame({
        "game_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "killed_id": [1, 0, 3, 3, 4, 2, 3, 2],
        "death_is_worthless": [True, None, False, True, False, True, False, True]
    })

    nb_worthless_deaths = _count_nb_worthless_deaths(event_df)

    expected_nb_worthless_deaths = pd.Series(
        [1, 1, 2, 0, 0],
        index=pd.MultiIndex.from_tuples([(1,1), (1,3), (2,2), (2,3), (2,4)]),
        name="death_is_worthless"
    )
    expected_nb_worthless_deaths.index.names = ["game_id", "player_id"]

    pd.testing.assert_series_equal(nb_worthless_deaths, expected_nb_worthless_deaths)

def test__count_nb_free_kills():
    event_df = pd.DataFrame({
        "game_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "killed_id": [1, 0, 3, 3, 4, 2, 3, 2],
        "killed_team_id": [1, None, 2, 2, 2, 1, 2, 1],
        "death_is_worthless": [True, None, False, True, False, True, False, True],
        "killer_id": [4, 4, 1, 2, 1, 3, 2, 4],
        "assisting_player_ids": [
            [], [3], [2], [1, 4], # last is team kill, valuable for 1 and 2 and not 4
            [], [4], [1], []
        ]
    })

    player_id_to_team_id_mapping = {
        (1, 1): 1,
        (1, 2): 1,
        (1, 3): 2,
        (1, 4): 2,
        (2, 3): 2,
        (2, 4): 2,
        (2, 1): 1,
        (2, 2): 1,        
    }

    nb_worthless_deaths = _count_nb_free_kills(event_df, player_id_to_team_id_mapping)

    expected_nb_worthless_deaths = pd.Series(
        [
            1, 1, 1,
            0, 0, 1, 2
        ],
        index=pd.MultiIndex.from_tuples([
            (1, 1), (1, 2), (1, 4),
            (2, 1), (2, 2), (2, 3), (2, 4)
        ]),
        name="kill_is_valuable"
    )
    expected_nb_worthless_deaths.index.names = ["game_id", "player_id"]

    pd.testing.assert_series_equal(nb_worthless_deaths, expected_nb_worthless_deaths)

def test__compute_death_worth_ratios():
    stat_df = pd.DataFrame({
        "game_id": (game_id_list := [1, 1, 1, 1, 2, 2, 2, 2, 2]),
        "player_id": (player_id_list := [1, 2, 3, 4, 3, 4, 1, 2, 5]),
        "team_id": [1, 1, 2, 2, 2, 2, 1, 1, 2],
        "player_deaths": [0, 1, 2, 3, 3, 2, 1, 9, 0],
        "player_kills": [0, 1, 2, 3, 4, 5, 6, 0, 0],
        "player_assists": [0, 1, 0, 2, 0, 3, 0, 4, 0],
        "total_kills": [6] * 4 + [15] * 5,
    })
    stat_df.set_index(["game_id", "player_id"], inplace=True)

    nb_worthless_deaths = pd.Series(
        [0, 1, 2, 0, 0, 1],
        index=pd.MultiIndex.from_tuples([(1,1), (1,3), (2,2), (2,3), (2,4), (2,5)]),
        name="death_is_worthless"
    )

    nb_free_kills = pd.Series(
        [
            0, 1, 1,
            0, 0, 1, 2, 1
        ],
        index=pd.MultiIndex.from_tuples([
            (1, 1), (1, 2), (1, 4),
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)
        ]),
        name="kill_is_valuable"
    )

    (
        worthless_death_ratio, 
        free_kill_ratio, 
        worthless_death_total_kills_ratio, 
        free_kill_total_kills_ratio
    ) = _compute_death_worth_ratios(stat_df, nb_worthless_deaths, nb_free_kills)

    expected_worthless_death_ratio = pd.Series(
        [0.0, 0.0, 1/2, 0.0, 0.0, 0.0, 0.0, 2/9, 0.0],
        index=pd.MultiIndex.from_tuples([
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4), (2, 1), (2, 2), (2, 5)
        ]),
    )
    expected_worthless_death_ratio.index.names = ["game_id", "player_id"]
    pd.testing.assert_series_equal(worthless_death_ratio, expected_worthless_death_ratio)

    expected_worthless_death_total_ratio = pd.Series(
        [0.0, 0.0, 1/6, 0.0, 0.0, 0.0, 0.0, 2/15, 0.0],
        index=pd.MultiIndex.from_tuples([
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4), (2, 1), (2, 2), (2, 5)
        ]),
    )
    expected_worthless_death_total_ratio.index.names = ["game_id", "player_id"]
    pd.testing.assert_series_equal(worthless_death_total_kills_ratio, expected_worthless_death_total_ratio)

    expected_free_kill_ratio = pd.Series(
        [
            0.0, 1/(1+1), 0.0, 1/(3+2),
            1/4, 2/8, 0.0, 0.0, 0.0
        ],
        index=pd.MultiIndex.from_tuples([
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4), (2, 1), (2, 2), (2, 5)
        ]),
    )
    expected_free_kill_ratio.index.names = ["game_id", "player_id"]
    pd.testing.assert_series_equal(free_kill_ratio, expected_free_kill_ratio)

    expected_free_kill_total_ratio = pd.Series(
        [
            0.0, 1/6, 0.0, 1/6,
            1/15, 2/15, 0.0, 0.0, 0.0
        ],
        index=pd.MultiIndex.from_tuples([
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4), (2, 1), (2, 2), (2, 5)
        ]),
    )
    expected_free_kill_total_ratio.index.names = ["game_id", "player_id"]
    pd.testing.assert_series_equal(free_kill_total_kills_ratio, expected_free_kill_total_ratio)

if __name__ == "__main__":
    test_compute_kill_death_value_features()

