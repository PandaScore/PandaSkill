import ast
from pandaskill.libs.feature_extraction.basic_features import *
from pandaskill.libs.feature_extraction.event_features import *
from pandaskill.experiments.general.utils import load_data
import logging
from os.path import join
import pandas as pd

def load_raw_data(raw_data_dir: str) -> pd.DataFrame:
    stat_df = load_data()

    game_events_df = pd.read_csv(join(raw_data_dir, "game_events.csv"), index_col=0)
    game_events_df['assisting_player_ids'] = game_events_df['assisting_player_ids'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    return stat_df, game_events_df

def compute_features(stat_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    stat_df["game_length_in_min"] = stat_df["game_length"] / 60
    stat_df["total_kills"] = stat_df["team_kills"] + compute_other_team_stat_from_team_stat(stat_df, "team_kills")

    logging.info("Computing basic features...")
    stat_df["gold_per_minute"] = compute_per_minute_feature(stat_df, "gold_earned")
    stat_df["cs_per_minute"] = compute_per_minute_feature(stat_df, "total_minions_killed")
    stat_df["xp_per_minute"] = compute_xp_per_minute(stat_df) 
    stat_df["kda"] = compute_kda(stat_df)
    stat_df["kla"] = compute_kla(stat_df)
    stat_df["damage_dealt_per_total_kills"] = compute_stat_per_total_kills(stat_df, "total_damage_dealt_to_champions")
    stat_df["damage_taken_per_total_kills"] = compute_stat_per_total_kills(stat_df, "total_damage_taken")
    stat_df["damage_dealt_per_total_kills_per_gold"] = compute_stat_per_total_kills_per_gold(stat_df, "total_damage_dealt_to_champions")
    stat_df["damage_taken_per_total_kills_per_gold"] = compute_stat_per_total_kills_per_gold(stat_df, "total_damage_taken")
    stat_df["wards_placed_per_minute"] = compute_per_minute_feature(stat_df, "wards_placed")
    stat_df["largest_killing_spree_per_total_kills"] = compute_stat_per_total_kills(stat_df, "largest_killing_spree")

    logging.info("Computing event features...")
    (
        worthless_death_ratio, free_kill_ratio, worthless_death_total_kills_ratio, free_kill_total_kills_ratio
    ) = compute_kill_death_value_features(stat_df, event_df, window=30)
    stat_df["worthless_death_ratio"] = worthless_death_ratio
    stat_df["free_kill_ratio"] = free_kill_ratio
    stat_df["worthless_death_total_kills_ratio"] = worthless_death_total_kills_ratio
    stat_df["free_kill_total_kills_ratio"] = free_kill_total_kills_ratio

    (
        objective_contest_winrate, objective_contest_loserate
    ) = compute_neutral_objective_contest_features(stat_df, event_df)
    stat_df["objective_contest_winrate"] = objective_contest_winrate
    stat_df["objective_contest_loserate"] = objective_contest_loserate

    # clean up
    stat_df = stat_df.drop(columns=["game_length_in_min", "total_kills"])

    return stat_df

def drop_neutral_objective_events_with_none_killer_id(event_df: pd.DataFrame) -> pd.DataFrame:
    """For some games, the killer of neutral objective is unknown. Instead of removing the games
    altogether, we only remove the events as the stats should be correct enough without them."""
    
    good_events_mask = event_df.killer_id.notna() | \
        (~event_df.event_type.isin(["drake_kill", "rift_herald_kill", "baron_nashor_kill"]))
    
    dropped_events_dict = (event_df[~good_events_mask]
          .groupby(["game_id", "event_type"])
          .size()
          .unstack(fill_value=0)
          .to_dict(orient='index'))
    dropped_events_summary = {
        "dropped_events": dropped_events_dict
    }
    logging.info(f"Dropping {len(dropped_events_summary['dropped_events'])} neutral objective events due to missing killer_id.")
    return event_df[good_events_mask], dropped_events_summary

def clean_up_largest_killing_spree(stat_df: pd.DataFrame) -> pd.DataFrame:
    largest_killing_spree = stat_df["largest_killing_spree"]
    player_kills = stat_df["player_kills"]
    player_deaths = stat_df["player_deaths"]

    # replace 0 killing spree by the number of kills if no deaths
    mask_kills_and_zero_death = (largest_killing_spree == 0) & (player_kills > 0) & (player_deaths == 0)
    stat_df.loc[
        mask_kills_and_zero_death, "largest_killing_spree"
    ] = stat_df.loc[mask_kills_and_zero_death, "player_kills"]

    # replace 0 killing spree by the ratio of kills to deaths if deaths > 0
    mask_kills_and_some_deaths = (largest_killing_spree == 0) & (player_kills > 0) & (player_deaths > 0)
    stat_df.loc[
        mask_kills_and_some_deaths, "largest_killing_spree"
    ] = np.ceil(
        stat_df.loc[mask_kills_and_some_deaths, "player_kills"] / stat_df.loc[mask_kills_and_some_deaths, "player_deaths"]
    ).astype(int)

    return stat_df

def clean_up_largest_multi_kill(stat_df: pd.DataFrame) -> pd.DataFrame:
    # replace 0 multi kill by 1 if player_kills > 0
    mask_kills = (stat_df["largest_multi_kill"] == 0) & (stat_df["player_kills"] > 0)
    stat_df.loc[mask_kills, "largest_multi_kill"] = 1
    return stat_df

