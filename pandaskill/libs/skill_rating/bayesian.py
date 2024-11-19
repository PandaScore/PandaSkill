from functools import partial
import numpy as np
import pandas as pd
from progressbar.progressbar import ProgressBar
from openskill.models import PlackettLuce, PlackettLuceRating
from pandaskill.libs.skill_rating.trueskill import TrueSkill, TrueSkillRating
import copy 
from typing import List, Dict, TypeAlias, TypeVar

DEFAULT_MU = 25.0
DEFAULT_SIGMA = 25/3
DEFAULT_LOWER_BOUND = 0.0

RatingDictType: TypeAlias = Dict[int, Dict[str, float]]
RatingListType: TypeAlias = List[Dict[str, float]]

Rating = TypeVar("Rating", PlackettLuceRating, TrueSkillRating)
Rater = TypeVar("Rater", PlackettLuce, TrueSkill)

def compute_bayesian_ratings(
    df: pd.DataFrame, 
    use_ffa_setting: str, 
    use_meta_ratings: bool,
    rater_model: str, 
) -> pd.DataFrame:
    all_skill_ratings, all_region_ratings = _initialize_ratings(df)
    rater_model = _instantiate_rater_model(rater_model)

    data_df = pd.pivot_table(
        df.reset_index(),
        values=['date', 'player_id', 'team_id', 'region', 'win', 'performance_score', 'region_change'],
        index='game_id',
        aggfunc={
            'date': lambda x: x.iloc[0],
            'player_id': list, 
            'region': list, 
            'team_id': list, 
            'win': list, 
            'performance_score': list,
            "region_change": list
        }
    )

    data_df = data_df.sort_values(by="date")

    rating_updates_dict = {}
    for game_id, row in ProgressBar(maxval=data_df.shape[0])(data_df.iterrows()):
        rating_updates_for_game = _compute_rating_updates_for_game(
            row, all_skill_ratings, all_region_ratings, rater_model, use_ffa_setting, use_meta_ratings
        )
        
        rating_updates_dict[game_id] = rating_updates_for_game

        all_skill_ratings, all_region_ratings = _apply_rating_updates(
            rating_updates_for_game, all_skill_ratings, all_region_ratings
        )
    
    rating_updates_dict = [
        [game_id, *rating_update]
        for game_id, rating_updates in rating_updates_dict.items()
        for rating_update in rating_updates
    ]
    skill_rating_updates_df = pd.DataFrame(
        data=rating_updates_dict, 
        columns=[
            "game_id", "player_id", "region", 
            "contextual_rating_before", "meta_rating_before", 
            "contextual_rating_after", "meta_rating_after",
        ]
    )

    skill_rating_updates_df = _combine_in_dataframe_contextual_and_meta_skill_ratings(skill_rating_updates_df)
    skill_rating_updates_df = skill_rating_updates_df.drop("region", axis=1)

    return skill_rating_updates_df

def _instantiate_rater_model(model: str) -> Rater:
    if model == "openskill":
        return PlackettLuce()
    elif model == "trueskill":
        return TrueSkill()

def _initialize_ratings(
    df: pd.DataFrame
) -> tuple[RatingDictType, RatingDictType]:
    player_ids = list(df.index.get_level_values(1).unique())
    skill_ratings_dict = {
        player_id: {
            "mu": DEFAULT_MU, 
            "sigma": DEFAULT_SIGMA,
            "lower_bound": DEFAULT_LOWER_BOUND
        }
        for player_id in player_ids
    }

    regions = list(df["region"].unique())
    region_ratings_dict = {
        region: {
            "mu": DEFAULT_MU, 
            "sigma": DEFAULT_SIGMA,
            "lower_bound": DEFAULT_LOWER_BOUND
        }
        for region in regions
    }

    return skill_ratings_dict, region_ratings_dict

def _compute_rating_updates_for_game(
    data_for_game: pd.Series, 
    all_skill_ratings: RatingListType,
    all_region_ratings: RatingListType,
    model: callable,
    use_ffa_setting: bool, 
    use_meta_ratings: bool,
) -> list[list]:
    game_player_ids = data_for_game["player_id"]
    game_player_regions = data_for_game["region"]
    game_performance_scores = data_for_game["performance_score"]
    game_region_changes = data_for_game["region_change"]
    meta_game = len(set(game_player_regions)) > 1 and use_meta_ratings

    game_contextual_ratings = [all_skill_ratings[id] for id in game_player_ids]
    game_meta_ratings = [all_region_ratings[region] for region in game_player_regions]

    if use_meta_ratings:
        game_contextual_ratings = _reset_rating_sigma_when_region_change(game_contextual_ratings, game_region_changes)

    full_ratings_before_game = _compute_ratings_before_game(
        game_contextual_ratings, game_meta_ratings, model, meta_game
    )

    full_ratings_after_game = _compute_ratings_after_game(
        full_ratings_before_game, game_performance_scores, model, use_ffa_setting
    )

    rating_updates_for_game = _compute_ratings_updates(
        full_ratings_after_game, game_player_ids, game_player_regions, game_contextual_ratings, game_meta_ratings, meta_game
    )

    return rating_updates_for_game

def _reset_rating_sigma_when_region_change(
    player_contextual_ratings: RatingListType,
    game_region_changes: list[bool]
) -> RatingListType:
    updated_ratings = []
    for rating, region_change in zip(player_contextual_ratings, game_region_changes):
        if region_change:
            rating = rating.copy()
            rating["sigma"] = DEFAULT_SIGMA
            rating["lower_bound"] = lower_bound_rating(rating["mu"], rating["sigma"])
        updated_ratings.append(rating)
    return updated_ratings

def _compute_ratings_before_game(
    player_contextual_ratings: RatingListType,
    player_meta_ratings: RatingListType,
    model: Rater, 
    meta_game: bool
) -> list[Rating]:
    if meta_game: 
        full_ratings_before_game = [
            model.rating(
                player_meta_rating["mu"] + player_contextual_rating["lower_bound"], 
                player_meta_rating["sigma"]
            )
            for player_contextual_rating, player_meta_rating in zip(player_contextual_ratings, player_meta_ratings)
        ]
    else:      
        full_ratings_before_game = [
            model.rating(
                player_contextual_rating["mu"],
                player_contextual_rating["sigma"]
            )
            for player_contextual_rating in player_contextual_ratings
        ]     

    return full_ratings_before_game

def _compute_ratings_after_game(
    ratings_before_game: list[Rating], 
    game_performance_scores: list[float], 
    model: Rater, 
    use_ffa_setting: bool,
) -> list[Rating]:
    ratings_before_game_copy = copy.deepcopy(ratings_before_game)

    if use_ffa_setting:
        ffa_ratings_after_game = model.rate(
            teams=[[skill_rating] for skill_rating in ratings_before_game_copy],
            scores=game_performance_scores,
        )
        ratings_after_game = [rating[0] for rating in ffa_ratings_after_game]
    else:
        winning_team_slice = slice(0, 5)
        losing_team_slice = slice(5, 10)
        ratings_before_game_formatted = [
            ratings_before_game_copy[winning_team_slice], 
            ratings_before_game_copy[losing_team_slice]
        ]
        team_ratings_after_game = model.rate(
            ratings_before_game_formatted, scores=[1, 0]
        )
        ratings_after_game = team_ratings_after_game[0] + team_ratings_after_game[1]

    return ratings_after_game

def _compute_ratings_updates(
    ratings_after_game: list[Rating], 
    player_ids_in_game: list[int], 
    player_regions_in_game: list[str], 
    contextual_ratings_in_game: RatingListType,
    meta_ratings_in_game: RatingListType,
    meta_game: bool
) -> list[list]:
    if meta_game:
        rating_updates_for_game = _compute_ratings_after_meta_game(
            ratings_after_game, player_ids_in_game, player_regions_in_game, contextual_ratings_in_game, meta_ratings_in_game
        )
    else:
        rating_updates_for_game = _compute_ratings_after_contextual_game(
            ratings_after_game, player_ids_in_game, player_regions_in_game, contextual_ratings_in_game, meta_ratings_in_game
        )

    return rating_updates_for_game

def _compute_ratings_after_contextual_game(
    full_ratings_after_game: list[Rating], 
    player_ids_in_game: list[int], 
    player_regions_in_game: list[str], 
    contextual_ratings_in_game: RatingListType, 
    meta_ratings_in_game: RatingListType, 
) -> list[list]:
    rating_updates = []
    for player_id, region, contextual_rating_before, meta_rating_before, full_rating_after in zip(
        player_ids_in_game, 
        player_regions_in_game, 
        contextual_ratings_in_game, 
        meta_ratings_in_game, 
        full_ratings_after_game
    ):
        contextual_rating_after = {
            "mu": full_rating_after.mu,
            "sigma": full_rating_after.sigma
        }
        contextual_rating_after["lower_bound"] = lower_bound_rating(*contextual_rating_after.values())
        meta_rating_after = meta_rating_before
        rating_updates.append(
            [player_id, region, contextual_rating_before, meta_rating_before, contextual_rating_after, meta_rating_after]
        )

    return rating_updates

def _compute_ratings_after_meta_game(
    full_ratings_after_game: List[Rating],
    player_ids_in_game: List[int],
    player_regions_in_game: List[str],
    contextual_ratings_in_game: RatingListType,
    meta_ratings_in_game: RatingListType,
) -> List[List]:
    temp_df = pd.DataFrame({
        "player_id": player_ids_in_game,
        "region": player_regions_in_game,
        "contextual_rating_before": contextual_ratings_in_game,
        "meta_rating_before": meta_ratings_in_game,
        "full_rating_after": full_ratings_after_game
    })

    temp_df["meta_rating_after"] = temp_df.apply(lambda row: {
        "mu": row["full_rating_after"].mu - row["contextual_rating_before"]["lower_bound"],
        "sigma": row["full_rating_after"].sigma,
        "lower_bound": lower_bound_rating(
            row["full_rating_after"].mu - row["contextual_rating_before"]["lower_bound"],
            row["full_rating_after"].sigma
        )
    }, axis=1)

    region_meta_ratings_ill_formatted = temp_df.groupby("region")["meta_rating_after"].apply(
        lambda ratings: {
            "mu": np.mean([r["mu"] for r in ratings]),
            "sigma": np.sqrt(np.mean([r["sigma"] ** 2 for r in ratings])),
        }
    ).to_dict()

    region_meta_ratings = {}
    for (region, rating_key), rating_value in region_meta_ratings_ill_formatted.items():
        if region not in region_meta_ratings:
            region_meta_ratings[region] = {}
        region_meta_ratings[region][rating_key] = rating_value

    for region in region_meta_ratings:
        mu = region_meta_ratings[region]["mu"]
        sigma = region_meta_ratings[region]["sigma"]
        region_meta_ratings[region]["lower_bound"] = lower_bound_rating(mu, sigma)

    rating_updates = []
    for idx, row in temp_df.iterrows():
        meta_after = region_meta_ratings[row["region"]]
        rating_updates.append([
            row["player_id"],
            row["region"],
            row["contextual_rating_before"],
            row["meta_rating_before"],
            row["contextual_rating_before"], # contextual rating doesn't change in meta game
            meta_after
        ])

    return rating_updates

def _apply_rating_updates(
    rating_updates: list[list], 
    contextual_ratings_dict: RatingDictType, 
    meta_ratings_dict: RatingDictType
) -> tuple[RatingDictType, RatingDictType]:
    contextual_ratings_dict = contextual_ratings_dict.copy()
    meta_ratings_dict = meta_ratings_dict.copy()
    for player_id, region, _, _, contextual_rating_after, meta_rating_after in rating_updates:
        contextual_ratings_dict[player_id] = contextual_rating_after
        meta_ratings_dict[region] = meta_rating_after

    return contextual_ratings_dict, meta_ratings_dict

def _combine_in_dataframe_contextual_and_meta_skill_ratings(
    skill_rating_updates_df: pd.DataFrame
) -> pd.DataFrame:
    skill_rating_updates_df = skill_rating_updates_df.set_index(["game_id", "player_id"])
    for col in ["contextual_rating_before", "contextual_rating_after", "meta_rating_before", "meta_rating_after"]:
        skill_rating_updates_df[col + "_mu"] = skill_rating_updates_df[col].apply(lambda x: x["mu"])
        skill_rating_updates_df[col + "_sigma"] = skill_rating_updates_df[col].apply(lambda x: x["sigma"])
        skill_rating_updates_df[col] = skill_rating_updates_df[col].apply(lambda x: x["lower_bound"])

    combine_contextual_and_meta_ratings_from_row = lambda row, before_after: combine_contextual_and_meta_ratings(
        row[f"contextual_rating_{before_after}_mu"], 
        row[f"contextual_rating_{before_after}_sigma"], 
        row[f"meta_rating_{before_after}_mu"], 
        row[f"meta_rating_{before_after}_sigma"]
    )
    for before_after in ["before", "after"]:
        skill_rating_updates_df[f"skill_rating_{before_after}"] = skill_rating_updates_df.apply(
            partial(combine_contextual_and_meta_ratings_from_row, before_after=before_after), axis=1
        )
        skill_rating_updates_df[f"skill_rating_{before_after}_mu"] = skill_rating_updates_df[f"skill_rating_{before_after}"].apply(lambda x: x[0])
        skill_rating_updates_df[f"skill_rating_{before_after}_sigma"] = skill_rating_updates_df[f"skill_rating_{before_after}"].apply(lambda x: x[1])
        skill_rating_updates_df[f"skill_rating_{before_after}"] = skill_rating_updates_df[f"skill_rating_{before_after}"].apply(lambda x: lower_bound_rating(*x))

    skill_rating_updates_df = skill_rating_updates_df.loc[
        :, 
        [
            "region",
            "contextual_rating_before_mu", "contextual_rating_before_sigma", "contextual_rating_before",
            "meta_rating_before_mu", "meta_rating_before_sigma", "meta_rating_before",
            "contextual_rating_after_mu", "contextual_rating_after_sigma", "contextual_rating_after",
            "meta_rating_after_mu", "meta_rating_after_sigma", "meta_rating_after",
            "skill_rating_before_mu", "skill_rating_before_sigma", "skill_rating_before",
            "skill_rating_after_mu", "skill_rating_after_sigma", "skill_rating_after",
        ]    
    ]

    return skill_rating_updates_df

def combine_contextual_and_meta_ratings(
    contextual_mu: float, contextual_sigma: float, meta_mu: float, meta_sigma: float
):
    overall_mu = contextual_mu + meta_mu
    overall_sigma = float(np.sqrt(contextual_sigma**2 + meta_sigma**2))
    return overall_mu, overall_sigma
    
def lower_bound_rating(mu: float, sigma: float) -> float:
    return float(mu - 3 * sigma)