import logging
import pandas as pd
from typing import Tuple

def drop_unwanted_games(
    stat_df: pd.DataFrame, event_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    stat_df, dropped_games_summary = _drop_incomplete_games(stat_df)
    stat_df, specific_dropped_games = _drop_specific_games(stat_df)
    dropped_games_summary["specific"] = specific_dropped_games

    all_dropped_games = []
    for dropped_games in dropped_games_summary.values():
        all_dropped_games.extend(dropped_games)
    event_df = event_df[~event_df.game_id.isin(all_dropped_games)]

    return stat_df, event_df, dropped_games_summary

def _drop_specific_games(stat_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    game_ids_to_drop = [
        215692, # one player didn't play (afk from start) - YDN vs DD 2020
        256680, # one player didn't play (afk from start) - Mammoth vs FURY 2024
    ]

    stat_df = stat_df[
        ~stat_df.index.get_level_values("game_id").isin(game_ids_to_drop)
    ]

    return stat_df, game_ids_to_drop

def _drop_incomplete_games(stat_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    stat_df, nan_dropped_games = _drop_games_with_nans(stat_df)
    stat_df, wrong_game_length_dropped_games = _drop_rows_with_wrong_value(stat_df, "game_length", 0)
    stat_df, wrong_gold_earned_dropped_games = _drop_rows_with_wrong_value(stat_df, "gold_earned", 0)
    stat_df, wrong_total_damage_dealt_to_champions_dropped_games = _drop_rows_with_wrong_value(stat_df, "total_damage_dealt_to_champions", 0)
    stat_df, wrong_total_damage_taken_dropped_games = _drop_rows_with_wrong_value(stat_df, "total_damage_taken", 0)
    stat_df, missing_rows_dropped_games = _drop_games_with_missing_rows(stat_df)
    dropped_games_summary = {
        "nan_dropped_games": nan_dropped_games,
        "wrong_game_length_dropped_games": wrong_game_length_dropped_games,
        "wrong_gold_earned_dropped_games": wrong_gold_earned_dropped_games,
        "wrong_total_damage_dealt_to_champions_dropped_games": wrong_total_damage_dealt_to_champions_dropped_games,
        "wrong_total_damage_taken_dropped_games": wrong_total_damage_taken_dropped_games,
        "missing_rows_dropped_games": missing_rows_dropped_games
    }
    return stat_df, dropped_games_summary

def _drop_rows_with_wrong_value(
    stat_df: pd.DataFrame, feature_name: str, wrong_value: int
) -> Tuple[pd.DataFrame, list]:
    feature_wrong_value_mask = stat_df.loc[:, feature_name] == wrong_value
    feature_wrong_value_dropped_game_ids = stat_df[feature_wrong_value_mask].index.get_level_values("game_id").unique().values.astype(int).tolist()
    logging.info(f"Dropping {len(feature_wrong_value_dropped_game_ids)} games due to \
        {wrong_value} {feature_name}: {feature_wrong_value_dropped_game_ids}")
    return stat_df[~feature_wrong_value_mask], feature_wrong_value_dropped_game_ids

def _drop_games_with_nans(stat_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    non_na_mask = ~stat_df.drop("series_name", axis=1).isna().any(axis=1)
    nan_cells_dropped_game_ids = stat_df[~non_na_mask].index.get_level_values("game_id").unique().values.astype(int).tolist()
    logging.info(f"Dropping {len(nan_cells_dropped_game_ids)} games due to NaN cells: \
        {nan_cells_dropped_game_ids}")
    return stat_df[non_na_mask], nan_cells_dropped_game_ids
    
def _drop_games_with_missing_rows(stat_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    mask_with_10_rows_per_game = stat_df.reset_index("game_id").groupby("game_id").count() == 10
    mask_with_10_rows_per_game = mask_with_10_rows_per_game[mask_with_10_rows_per_game["date"]]
    complete_game_ids = mask_with_10_rows_per_game.index
    rows_to_keep_mask = stat_df.index.get_level_values("game_id").isin(complete_game_ids)
    missing_rows_dropped_game_ids = stat_df[~rows_to_keep_mask].index.get_level_values("game_id").unique().values.astype(int).tolist()
    logging.info(f"Dropping {len(missing_rows_dropped_game_ids)} games due to missing rows: \
        {missing_rows_dropped_game_ids}")
    return stat_df[rows_to_keep_mask], missing_rows_dropped_game_ids
