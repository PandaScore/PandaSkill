import pandas as pd
import numpy as np

from pandaskill.libs.feature_extraction.basic_features import *
from pandaskill.libs.feature_extraction.basic_features import (
    _compute_team_stat_from_players_stat
)

def test_compute_per_minute_feature():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0],
        "feature_2": [2.0, 5.0, 9.0],
        "game_length_in_min": [1.0, 4.0, 3.0],
    })

    feature_per_minute = compute_per_minute_feature(df, "feature_1")

    pd.testing.assert_series_equal(
        feature_per_minute,
        pd.Series([1.0, 0.5, 1.0])
    )

def test_compute_xp_per_minute():
    df = pd.DataFrame({
        "level": [1, 11, 18],
        "game_length_in_min": [1.0, 4.0, 3.0],
    })

    xppm = compute_xp_per_minute(df)

    pd.testing.assert_series_equal(
        xppm,
        pd.Series([0.0, XP_PER_LEVEL_TABLE[11] / 4, XP_PER_LEVEL_TABLE[18] / 3])
    )

def test__compute_other_team_stat():
    df = pd.DataFrame({
        "game_id": [1]*10 + [2]*10,
        "team_stat": [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5,
    })
    df = df.set_index("game_id")

    other_team_stat = compute_other_team_stat_from_team_stat(df, "team_stat")

    expected_result = pd.Series([2] * 5 + [1] * 5 + [4] * 5 + [3] * 5)
    expected_result = expected_result.set_axis(df.index, axis=0)

    pd.testing.assert_series_equal(other_team_stat, expected_result)


kda_kla_input_df = pd.DataFrame({
    "player_kills": [0, 0, 1, 0, 1, 1],
    "player_deaths": [0, 1, 1, 1, 0, 1],
    "player_assists": [0, 0, 0, 1, 1, 1]
})

def test_compute_kda():
    kda = compute_kda(kda_kla_input_df)

    pd.testing.assert_series_equal(
        kda,
        pd.Series([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    )

def test_compute_kla():
    kla = compute_kla(kda_kla_input_df)

    pd.testing.assert_series_equal(
        kla,
        pd.Series([0.0, 0.0, 0.5, 0.5, 2.0, 1.0])
    )

def test_compute_stat_per_gold():
    df = pd.DataFrame({
        "stat": [1, 2, 3],
        "gold_earned": [1, 4, 3],
    })

    stat_per_gold = compute_stat_per_gold(df, "stat")

    pd.testing.assert_series_equal(
        stat_per_gold,
        pd.Series([1.0, 0.5, 1.0])
    )

def test_compute_stat_per_total_kills():
    df = pd.DataFrame({
        "stat": [1, 2, 3],
        "total_kills": [1, 4, 3],
    })

    stat_per_total_kills = compute_stat_per_total_kills(df, "stat")

    pd.testing.assert_series_equal(
        stat_per_total_kills,
        pd.Series([1.0, 0.5, 1.0])
    )

def test_compute_stat_per_total_kills_per_gold():
    df = pd.DataFrame({
        "stat": [1, 2, 3, 0],
        "total_kills": [1, 4, 3, 1],
        "gold_earned": [1, 4, 1, 1],
    })

    stat_per_total_kills_per_gold = compute_stat_per_total_kills_per_gold(df, "stat")

    pd.testing.assert_series_equal(
        stat_per_total_kills_per_gold,
        pd.Series([1.0, 0.125, 1.0, 0.0])
    )

def test_compute_stat_per_gold_per_life():
    df = pd.DataFrame({
        "stat": [1, 2, 3, 0],
        "gold_earned": [1, 4, 3, 1],
        "player_deaths": [1, 1, 0, 1],
    })

    stat_per_gold_per_life = compute_stat_per_gold_per_life(df, "stat")

    pd.testing.assert_series_equal(
        stat_per_gold_per_life,
        pd.Series([0.5, 0.25, 1.0, 0.0])
    )

def test__compute_team_stat_from_players_stat():
    df = pd.DataFrame({
        "game_id": [1]*10 + [2]*10,
        "player_id": list(range(10)) + list(range(5,10)) + list(range(0,5)),
        "player_stat": [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5,
        "team_id": [1] * 5 + [2] * 5 + [2] * 5 + [1] * 5,
    })
    df = df.set_index(["game_id", "player_id"])

    team_stat = _compute_team_stat_from_players_stat(df, "player_stat")

    expected_result = pd.Series([5] * 5 + [10] * 5 + [15] * 5 + [20] * 5)
    expected_result = expected_result.set_axis(df.index, axis=0)
    expected_result.name = "player_stat"

    pd.testing.assert_series_equal(team_stat, expected_result)

