import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple

def compute_neutral_objective_contest_features(
    stat_df: pd.DataFrame, event_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    stat_df = stat_df.copy()
    event_df = event_df.copy()

    player_id_to_team_id_mapping = stat_df.team_id.to_dict()

    event_df = _prepare_event_df_for_neutral_objective_contest(event_df)
    
    total_nb_contestable_objectives = event_df.groupby("game_id")["contestable_objective"].sum()

    event_participation_df = _break_down_contestable_objective_events_by_participation(event_df, player_id_to_team_id_mapping)
    event_participation_df = _calculate_contest_results(event_participation_df)
    win_lose_contested_events_counts_df = _count_win_and_lose_contested_events(event_participation_df)

    objective_contest_winrate = win_lose_contested_events_counts_df["win_contested_event"] / total_nb_contestable_objectives
    objective_contest_loserate = win_lose_contested_events_counts_df["lose_contested_event"] / total_nb_contestable_objectives

    objective_contest_winrate = objective_contest_winrate.reindex(stat_df.index, fill_value=0.0)
    objective_contest_loserate = objective_contest_loserate.reindex(stat_df.index, fill_value=0.0)
    
    return objective_contest_winrate, objective_contest_loserate

def _prepare_event_df_for_neutral_objective_contest(event_df: pd.DataFrame) -> pd.DataFrame:
    event_df["contestable_objective"] = event_df.loc[:, "event_type"].isin([
        "drake_kill", "rift_herald_kill", "baron_nashor_kill",
    ]) # voidgrubs have been left out of the contestable objectives (very recent, and not really contested invidivually)

    event_df["game_id_killer_id"] = list(zip(event_df["game_id"], event_df["killer_id"]))
    return event_df

def _break_down_contestable_objective_events_by_participation(
    event_df: pd.DataFrame, player_id_to_team_id_mapping: dict
) -> pd.DataFrame:
    event_participation_df = event_df[event_df["contestable_objective"]].copy()
    event_participation_df["winning_event_team_id"] = event_participation_df["game_id_killer_id"].map(player_id_to_team_id_mapping)
    event_participation_df["participant_ids"] = event_participation_df.loc[:, ["killer_id", "assisting_player_ids"]].apply(lambda row: [row['killer_id']] + row['assisting_player_ids'], axis=1)
    event_participation_df = event_participation_df.explode("participant_ids")
    event_participation_df["game_id_player_id"] = list(zip(event_participation_df["game_id"], event_participation_df["participant_ids"]))
    event_participation_df.loc[:, "participant_team_id"] = event_participation_df["game_id_player_id"].map(player_id_to_team_id_mapping)
    return event_participation_df

def _calculate_contest_results(event_participation_df: pd.DataFrame) -> pd.DataFrame:
    event_participation_df = event_participation_df[event_participation_df["participant_team_id"].notna()].copy()
    event_participation_df["contested_event"] = event_participation_df["participant_team_id"].groupby("id", as_index=True).agg(
        lambda x: len(np.unique(x).astype(int)) > 1 # a contested event is an event that has several teams in its killer+assists
    )
    event_participation_df["win_event"] = event_participation_df["participant_team_id"] == event_participation_df["winning_event_team_id"]
    event_participation_df["win_contested_event"] = event_participation_df["win_event"] & event_participation_df["contested_event"]
    event_participation_df["lose_contested_event"] = (~event_participation_df["win_event"]) & event_participation_df["contested_event"]
    return event_participation_df

def _count_win_and_lose_contested_events(participants_contestable_objectives: pd.DataFrame) -> pd.DataFrame:
    nb_contested_event_win = participants_contestable_objectives.loc[:, ["game_id", "participant_ids", "win_contested_event", "lose_contested_event"]] \
        .groupby(["game_id", "participant_ids"]).sum()
    nb_contested_event_win.index.set_levels(nb_contested_event_win.index.levels[1].astype(int), level=1)
    nb_contested_event_win.index = nb_contested_event_win.index.set_names('player_id', level='participant_ids')
    return nb_contested_event_win

def compute_kill_death_value_features(
    stat_df: pd.DataFrame, event_df: pd.DataFrame, window: int=30
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    event_df = event_df.copy()

    player_id_to_team_id_mapping = stat_df.team_id.to_dict()

    event_df = _prepare_event_df_for_death_worth_features(event_df, player_id_to_team_id_mapping)

    event_df['death_is_worthless'] = _evaluate_deaths_worthlessness(event_df, window)

    nb_worthless_deaths = _count_nb_worthless_deaths(event_df)
    nb_free_kills = _count_nb_free_kills(event_df, player_id_to_team_id_mapping)

    (
        worthless_death_ratio, 
        free_kill_ratio, 
        worthless_death_total_kills_ratio, 
        free_kill_total_kills_ratio
    ) = _compute_death_worth_ratios(stat_df, nb_worthless_deaths, nb_free_kills)

    return worthless_death_ratio, free_kill_ratio, worthless_death_total_kills_ratio, free_kill_total_kills_ratio

def _prepare_event_df_for_death_worth_features(
    event_df: pd.DataFrame, player_id_to_team_id_mapping: dict
) -> pd.DataFrame:
    event_df["game_id_killed_id"] = list(zip(event_df["game_id"], event_df["killed_id"]))
    event_df["game_id_killer_id"] = list(zip(event_df["game_id"], event_df["killer_id"]))
    event_df["killed_team_id"] = event_df["game_id_killed_id"].map(player_id_to_team_id_mapping)
    event_df["killer_team_id"] = event_df["game_id_killer_id"].map(player_id_to_team_id_mapping)
    event_df = event_df[event_df["killer_team_id"] != event_df["killed_team_id"]] # remove team kills (very rare, and can't exist outside of Renata ult)
    return event_df

def _evaluate_deaths_worthlessness(event_df: pd.DataFrame, window: int=30) -> pd.Series:
    with mp.Pool(mp.cpu_count()) as pool:
        groupby_game_id = event_df.groupby('game_id')
        results = list(tqdm(
            pool.imap(_evaluate_deaths_worthlessness_for_game, 
                    ((group, window) for _, group in groupby_game_id)),
            total=groupby_game_id.ngroups,
            desc="Evaluating death events worthlessness for games"
        ))    
    worthless_deaths_series = pd.concat(results)
    return worthless_deaths_series

def _evaluate_deaths_worthlessness_for_game(args: Tuple[pd.DataFrame, int]) -> pd.Series:
    game_df, window = args
    game_df = game_df.copy()
    game_df = game_df.sort_values('timestamp')
    timestamps = game_df['timestamp'].values
    killer_ids = game_df['killer_id'].values
    killed_ids = game_df['killed_id'].values
    assisting_player_ids = game_df['assisting_player_ids'].values
    killer_team_ids = game_df['killer_team_id'].values
    killed_team_ids = game_df['killed_team_id'].values
    event_types = game_df['event_type'].values
    
    n = len(timestamps)
    death_is_worthless = [None] * n
    
    for i in range(n):
        timestamp = timestamps[i]
        killed_id = killed_ids[i]
        killed_team_id = killed_team_ids[i]

        if event_types[i] != 'player_kill':
            continue
        
        time_diff = np.abs(timestamps - timestamp)
        in_window = time_diff < window
        
        killed_as_killer = (killer_ids[in_window] == killed_id)
        killed_as_assist = np.array([killed_id in (assist_ids or []) for assist_ids in assisting_player_ids[in_window]])
        same_team = (killer_team_ids[in_window] == killed_team_id)
        
        positive_event_participation = killed_as_killer | (killed_as_assist & same_team)
        objective_taken_by_his_team = (event_types[in_window] != 'player_kill') & same_team
        
        death_is_worthless[i] = not np.any(positive_event_participation | objective_taken_by_his_team)
    
    worthless_death_series = pd.Series(death_is_worthless, index=game_df.index)

    return worthless_death_series

def _count_nb_worthless_deaths(event_df: pd.DataFrame) -> pd.Series:
    nb_worthless_deaths = event_df[event_df.death_is_worthless.notna()] \
        .groupby(["game_id",'killed_id'])['death_is_worthless'].agg(lambda x: x.astype(int).sum())
    nb_worthless_deaths.index.rename(["game_id", "player_id"], inplace=True)
    return nb_worthless_deaths

def _count_nb_free_kills(event_df: pd.DataFrame, player_id_to_team_id_mapping: dict) -> pd.Series:
    event_df = event_df[event_df["death_is_worthless"].notna()].copy()
    event_df["participant_ids"] = event_df.apply(lambda row: [row['killer_id']] + row['assisting_player_ids'] , axis=1)
    event_per_participant_df = event_df.explode("participant_ids")
    event_per_participant_df["game_id_player_id"] = list(zip(event_per_participant_df["game_id"], event_per_participant_df["participant_ids"]))
    event_per_participant_df["participant_team_id"] = event_per_participant_df["game_id_player_id"].map(player_id_to_team_id_mapping)
    event_per_participant_df["is_not_team_kill"] = event_per_participant_df["participant_team_id"] != event_per_participant_df["killed_team_id"]
    event_per_participant_df["kill_is_valuable"] = event_per_participant_df["death_is_worthless"] & event_per_participant_df["is_not_team_kill"]
    nb_free_kills = event_per_participant_df.groupby(["game_id", "participant_ids"])["kill_is_valuable"].agg(lambda x: x.astype(int).sum())
    nb_free_kills.index.rename(["game_id", "player_id"], inplace=True)
    return nb_free_kills

def _compute_death_worth_ratios(
    stat_df: pd.DataFrame, nb_worthless_deaths: pd.Series, nb_free_kills: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    nb_worthless_deaths = nb_worthless_deaths.reindex(stat_df.index, fill_value=0.0)
    nb_free_kills = nb_free_kills.reindex(stat_df.index, fill_value=0.0)
    worthless_death_ratio = nb_worthless_deaths / stat_df["player_deaths"]
    free_kill_ratio = nb_free_kills / (stat_df["player_kills"] + stat_df["player_assists"])

    # account for (rare) misalignments between events and stats (e.g., in event a player got a kill, but player has 0 kill in his stats)
    worthless_death_ratio = worthless_death_ratio.fillna(0.0)
    worthless_death_ratio = worthless_death_ratio.replace([np.inf, -np.inf], 0.0)
    free_kill_ratio = free_kill_ratio.fillna(0.0)
    free_kill_ratio = free_kill_ratio.replace([np.inf, -np.inf], 0.0)

    # recompute the ratios with respect to total kills (taking into acount the potential misalignments)
    worthless_death_total_kills_ratio = worthless_death_ratio * stat_df["player_deaths"] / stat_df["total_kills"]
    free_kill_total_kills_ratio = free_kill_ratio * (stat_df["player_kills"] + stat_df["player_assists"]) / stat_df["total_kills"]
    return worthless_death_ratio, free_kill_ratio, worthless_death_total_kills_ratio, free_kill_total_kills_ratio
