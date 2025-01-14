from datetime import datetime, timedelta
from pandaskill.experiments.data.player_region import MAIN_LEAGUE_SERIES_TOURNAMENT_WHITELIST
from pandaskill.experiments.general.utils import ROLES, ALL_REGIONS
from pandaskill.experiments.general.visualization import plot_violin_distributions
from pandaskill.libs.skill_rating.bayesian import lower_bound_rating, combine_contextual_and_meta_ratings
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import pandas as pd
import seaborn as sns
from typing import List, Optional

def visualize_ratings(
    data_with_ratings: pd.DataFrame, 
    experiment_dir: str, 
    visualization_parameters: dict
) -> None:  
    saving_dir = join(experiment_dir, "ratings_visualization")
    os.makedirs(saving_dir, exist_ok=True)

    _visualize_ratings_distributions(data_with_ratings, saving_dir, visualization_parameters["min_nb_games"], visualization_parameters["since"])

    if "meta_rating_after" in data_with_ratings.columns:
        _visualize_meta_rating_evolution(
            data_with_ratings,
            saving_dir
        )

def _visualize_ratings_distributions(
    ratings_df: pd.DataFrame, saving_dir: str, min_nb_games: int, since: str
) -> None:
    last_skill_ratings_df = ratings_df.groupby("player_id").last()
    last_skill_ratings_df["nb_games"] = ratings_df.groupby("player_id").count()["date"]
    last_skill_ratings_df = last_skill_ratings_df[last_skill_ratings_df["date"] >= since]

    last_skill_ratings_df = last_skill_ratings_df[last_skill_ratings_df["nb_games"] >= min_nb_games]


    plot_violin_distributions(
        last_skill_ratings_df,
        "role",
        "skill_rating_after",
        f"skill rating distribution by role",
        "Role",
        "Player Skill Rating",
        saving_dir,
        "skill_rating_distribution_by_role.png"
    )

    plot_violin_distributions(
        last_skill_ratings_df,
        "region",
        "skill_rating_after",
        f"skill rating distribution by region",
        "Region",
        "Player Skill Rating",
        saving_dir,
        "skill_rating_distribution_by_region.png"
    )

    if "contextual_rating_after" in last_skill_ratings_df.columns:
        plot_violin_distributions(
            last_skill_ratings_df,
            "role",
            "contextual_rating_after",
            f"Player contextual rating distribution by role",
            "Role",
            "Player Contextual Skill Rating",
            saving_dir,
            "contextual_rating_distribution_by_role.png"
        )
        plot_violin_distributions(
            last_skill_ratings_df,
            "region",
            "contextual_rating_after",
            f"Player contextual rating distribution by region",
            "Region",
            "Player Contextual Skill Rating",
            saving_dir,
            "player_contextual_rating_distribution_by_region.png"
        )


def _visualize_meta_rating_evolution(
    ratings_df: pd.DataFrame,
    experiment_dir: str,
) -> None:
    region_skill_ratings_after_series_df, nb_interregion_games_per_series = construct_skill_ratings_for_region_after_series(
        ratings_df
    )
    if region_skill_ratings_after_series_df.empty:
        return 

    _create_and_save_meta_rating_evolution(
        region_skill_ratings_after_series_df, 
        nb_interregion_games_per_series,
        "Average per region skill rating Evolution after inter-region games",
        "skill_rating_evolution_per_series_and_region_mean.png", 
        experiment_dir
    )

    sorted_df = region_skill_ratings_after_series_df.sort_values(
        by=['series_name', 'region', 'skill_rating_after'], 
        ascending=[True, True, False]
    )
    top_10_players = sorted_df.groupby(['series_name', 'region'], observed=False).head(10).reset_index()
    _create_and_save_meta_rating_evolution(
        top_10_players,
        nb_interregion_games_per_series,
        "Top 10 per region skill rating evolution after inter-region games",
        "skill_rating_evolution_per_series_and_region_best.png",
        experiment_dir
    )

def construct_skill_ratings_for_region_after_series(
    ratings_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:    
    interregion_ratings = ratings_df[
        ~(ratings_df.meta_rating_before == ratings_df.meta_rating_after)
    ]

    if interregion_ratings.empty:
        return pd.DataFrame([]), pd.Series([])

    data = []
    nb_interregion_games = []
    series_names = interregion_ratings.series_name.unique()
    for series_name in series_names:
        series_interregion_data = interregion_ratings[interregion_ratings.series_name == series_name]
        series_end_date = ratings_df[ratings_df.series_name == series_name]["date"].max()
        for region in series_interregion_data.region.unique():
            ratings_in_region_after_series = _get_skill_ratings_of_region_at_a_given_point_in_time(
                ratings_df, region, series_end_date
            )
            ratings_after_series = ratings_in_region_after_series.loc[:, ["skill_rating_after"]]
            ratings_after_series["region"] = region
            ratings_after_series["series_name"] = series_name            
            data.append(ratings_after_series)

        nb_interregion_games_for_series = len(series_interregion_data.index.get_level_values(0).unique())
        nb_interregion_games.append(nb_interregion_games_for_series)

    region_skill_ratings_after_series_df = pd.concat(data)

    nb_interregion_games_df = pd.Series(
        nb_interregion_games, index=series_names
    )

    return region_skill_ratings_after_series_df, nb_interregion_games_df


def _get_skill_ratings_of_region_at_a_given_point_in_time(
    ratings_df: pd.DataFrame,
    region: str,
    date: str
) -> pd.DataFrame:    
    """
    Return a DataFrame containing all the players ratings belonging to a given region at given date. 
    The ratings are adjusted based on the last meta rating before the date.
    Players are filtered based on their last game date, which should be at most 6 months before the given date.
    Players that have stopped playing before the 6-month date are still considered (edge case not handled).
    """

    def recompute_ratings_for_players_in_region(last_rating_for_players_in_last_regular_season, last_region_rating_mu, last_region_rating_sigma):
        last_rating_for_players_in_last_regular_season["meta_rating_after_mu"] = last_region_rating_mu
        last_rating_for_players_in_last_regular_season["meta_rating_after_sigma"] = last_region_rating_sigma
        last_rating_for_players_in_last_regular_season["meta_rating_after"] = lower_bound_rating(last_region_rating_mu, last_region_rating_sigma)
        
        new_overall_ratings = last_rating_for_players_in_last_regular_season.apply(lambda row: combine_contextual_and_meta_ratings(
            row["contextual_rating_after_mu"], row["contextual_rating_after_sigma"],
            row["meta_rating_after_mu"], row["meta_rating_after_sigma"],    
        ), axis=1).values
        new_overall_ratings = np.array([[mu, sigma] for (mu, sigma) in new_overall_ratings])
        last_rating_for_players_in_last_regular_season["skill_rating_after_mu"] = new_overall_ratings[:, 0]
        last_rating_for_players_in_last_regular_season["skill_rating_after_sigma"] = new_overall_ratings[:, 1]
        last_rating_for_players_in_last_regular_season["skill_rating_after"] = last_rating_for_players_in_last_regular_season.apply(
            lambda row: lower_bound_rating(
                row["skill_rating_after_mu"], row["skill_rating_after_sigma"]
            ),
            axis=1
        )
        return last_rating_for_players_in_last_regular_season

    six_months_before_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f") - timedelta(days=6*30)
    last_ratings_in_region = ratings_df[
        (ratings_df.date <= date)
        & (ratings_df.date >= str(six_months_before_date))
        & (ratings_df.region == region)
    ].reset_index().sort_values("date").groupby("player_id").last().copy()

    last_region_rating_mu, last_region_rating_sigma = last_ratings_in_region.tail(1).loc[:, ["meta_rating_after_mu", "meta_rating_after_sigma"]].values[0]
    last_ratings_in_region = recompute_ratings_for_players_in_region(last_ratings_in_region, last_region_rating_mu, last_region_rating_sigma)

    return last_ratings_in_region


def _create_and_save_meta_rating_evolution(
    ratings_in_region_after_series: pd.DataFrame, 
    nb_games_in_series: pd.Series,
    title: str, 
    file_name: str, 
    saving_dir: str
) -> None:
    sns.set_theme(style="white")

    fig, ax1 = plt.subplots(figsize=(18, 12))

    ax2 = ax1.twinx()

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    ax1.set_axisbelow(True)  
    ax1.grid(True, axis="y")

    ax2.grid(False)

    color_palette = sns.color_palette()
    color_dict = dict(zip(ALL_REGIONS, color_palette))

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'H', 'P', 'X']
    marker_dict = dict(zip(ALL_REGIONS, markers[:len(ALL_REGIONS)]))

    ax2.bar(
        nb_games_in_series.index, 
        nb_games_in_series.values, 
        color='lightgrey', 
        alpha=1.0
    )
    ax2.set_ylabel("Number of Inter-region Games Played")

    for region in ALL_REGIONS:
        region_data = ratings_in_region_after_series[ratings_in_region_after_series['region'] == region]
        color = color_dict[region]
        marker = marker_dict[region]

        mean_data = region_data.groupby("series_name", observed=False)["skill_rating_after"].mean().reset_index()

        sns.lineplot(
            data=mean_data, 
            x="series_name", 
            y="skill_rating_after", 
            label=region, 
            color=color,
            errorbar=None, 
            ax=ax1
        )
        
        
        ax1.scatter(
            mean_data["series_name"], 
            mean_data["skill_rating_after"], 
            color=color, 
            marker=marker,
            s=100
        )

    ax1.set_ylabel("Average Skill Rating of Players in Region After Tournament")
    ax1.set_xlabel("Tournament")

    plt.setp(ax1.get_xticklabels(), rotation=90, ha='right')

    legend_elements = [
        plt.Line2D(
            [0], [0], 
            color=color_dict[region], 
            marker=marker_dict[region], 
            label=region, 
            markersize=10, 
            linewidth=2
        )
        for region in ALL_REGIONS
    ]
    ax1.legend(handles=legend_elements, title="Region", loc='upper left')

    fig.tight_layout()

    plt.savefig(
        join(saving_dir, f"{file_name[:-4]}.pdf"), 
        format="pdf", 
        bbox_inches="tight"
    )

    plt.close()