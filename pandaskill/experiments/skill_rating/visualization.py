from datetime import datetime, timedelta
from pandaskill.experiments.data.player_region import MAIN_LEAGUE_SERIE_TOURNAMENT_WHITELIST
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
        _visualize_ratings_evolutions(
            data_with_ratings, 
            "meta", 
            saving_dir,
            "meta_rating_evolution.png",
            since=visualization_parameters["since"]
        )

def _visualize_ratings_distributions(
    ratings_df: pd.DataFrame, saving_dir: str, min_nb_games: int, since: str
) -> None:
    last_player_ratings_df = ratings_df.groupby("player_id").last()
    last_player_ratings_df["nb_games"] = ratings_df.groupby("player_id").count()["date"]
    last_player_ratings_df = last_player_ratings_df[last_player_ratings_df["date"] >= since]

    last_player_ratings_df = last_player_ratings_df[last_player_ratings_df["nb_games"] >= min_nb_games]


    plot_violin_distributions(
        last_player_ratings_df,
        "role",
        "player_rating_after",
        f"Player rating distribution by role",
        "Role",
        "Player Skill Rating",
        saving_dir,
        "player_rating_distribution_by_role.png"
    )

    plot_violin_distributions(
        last_player_ratings_df,
        "region",
        "player_rating_after",
        f"Player rating distribution by region",
        "Region",
        "Player Skill Rating",
        saving_dir,
        "player_rating_distribution_by_region.png"
    )

    if "contextual_rating_after" in last_player_ratings_df.columns:
        plot_violin_distributions(
            last_player_ratings_df,
            "role",
            "contextual_rating_after",
            f"Player contextual rating distribution by role",
            "Role",
            "Player Contextual Skill Rating",
            saving_dir,
            "contextual_rating_distribution_by_role.png"
        )
        plot_violin_distributions(
            last_player_ratings_df,
            "region",
            "contextual_rating_after",
            f"Player contextual rating distribution by region",
            "Region",
            "Player Contextual Skill Rating",
            saving_dir,
            "player_contextual_rating_distribution_by_region.png"
        )

def _visualize_ratings_evolutions(
    ratings_df: pd.DataFrame, 
    kind: str, 
    saving_dir: str, 
    file_name: str, 
    subset: Optional[List[str]] = None,
    since: Optional[str] = None
) -> None:
    if since is not None:
        ratings_df = ratings_df[ratings_df["date"] >= since]
        
    if kind == "meta":
        _visualize_meta_rating_evolution(
            ratings_df,
            saving_dir
        )
    elif kind in ["player", "team"]:
        if subset is None:
            raise ValueError("`player` or `team` rating evolution requires a `subset`")
        
        if kind == "player":
            title = f"Player rating evolution for {', '.join(subset)}"
            hue = "player_name"
            ratings_df = ratings_df.reset_index().loc[
                :, ["game_id", "date", hue, "player_rating_after"]
            ]
        elif kind == "team":
            title = f"Team rating (average of its players) evolution for {', '.join(subset)} "
            ratings_df = ratings_df.reset_index().groupby(["game_id", "team_name"]).agg(
                {
                    "player_rating_after": "mean",
                    "date": "first"
                }
            ).reset_index()
            hue = "team_name"
        
        y = "player_rating_after"
        ratings_df = ratings_df[ratings_df[hue].isin(subset)]

        _create_and_save_line_plot(
            ratings_df,
            y,
            hue,
            title,
            saving_dir,
            file_name
        )
    else:
        raise ValueError(f"`{kind}` rating evolution not recognized")
    

def _visualizes_series_ratings_evolutions(ratings_df, serie_name, saving_dir):
    ratings_df = ratings_df[ratings_df["serie_name"] == serie_name]

    serie_name_lower = serie_name.lower().replace(" ", "_")

    for role in ROLES:
        role_ratings_df = ratings_df[ratings_df["role"] == role]
        _create_and_save_line_plot(
            role_ratings_df,
            "player_rating_after",
            "player_name",
            f"Player rating evolution for {serie_name} and role {role}",
            saving_dir,
            f"{serie_name_lower}_player_rating_evolution_for_role_{role}.png"
        )

    team_ratings_df = ratings_df.reset_index().groupby(["game_id", "team_name"]).agg({
        "player_rating_after": "mean",
        "date": "first",
    }).reset_index()
    _create_and_save_line_plot(
        team_ratings_df,
        "player_rating_after",
        "team_name",
        f"Team rating evolution for {serie_name}",
        saving_dir,
        f"{serie_name_lower}_team_rating_evolution.png"
    )

def _create_and_save_line_plot(
    ratings_df: pd.DataFrame, 
    y: str, 
    hue: str, 
    title: str, 
    saving_dir: str, 
    file_name: str
):
    # sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 6))

    date = pd.to_datetime(ratings_df['date'])
    ax = sns.lineplot(x=date, y=ratings_df[y], hue=ratings_df[hue], markers=True, dashes=False)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rating")

    min_date, max_date = date.min(), date.max()
    nb_months = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month + 1

    date_display_interval = 1 if nb_months <= 12 else 3

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=date_display_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

    plt.tight_layout()
    plt.legend(loc='center left', bbox_to_anchor=(0, 0.5))

    plt.savefig(join(saving_dir, file_name), bbox_inches='tight')
    plt.close()

def _visualize_meta_rating_evolution(
    ratings_df: pd.DataFrame,
    experiment_dir: str,
) -> None:
    region_player_ratings_after_serie_df, nb_interregion_games_per_serie = construct_player_ratings_for_region_after_serie(
        ratings_df
    )
    if region_player_ratings_after_serie_df.empty:
        return 

    _create_and_save_meta_rating_evolution(
        region_player_ratings_after_serie_df, 
        nb_interregion_games_per_serie,
        "Average per region player rating Evolution after inter-region games",
        "player_rating_evolution_per_serie_and_region_mean.png", 
        experiment_dir
    )

    sorted_df = region_player_ratings_after_serie_df.sort_values(
        by=['serie_name', 'region', 'player_rating_after'], 
        ascending=[True, True, False]
    )
    top_10_players = sorted_df.groupby(['serie_name', 'region'], observed=False).head(10).reset_index()
    _create_and_save_meta_rating_evolution(
        top_10_players,
        nb_interregion_games_per_serie,
        "Top 10 per region player rating evolution after inter-region games",
        "player_rating_evolution_per_serie_and_region_best.png",
        experiment_dir
    )

def construct_player_ratings_for_region_after_serie(
    ratings_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:    
    interregion_ratings = ratings_df[
        ~(ratings_df.meta_rating_before == ratings_df.meta_rating_after)
    ]

    if interregion_ratings.empty:
        return pd.DataFrame([]), pd.Series([])

    data = []
    nb_interregion_games = []
    serie_names = interregion_ratings.serie_name.unique()
    for serie_name in serie_names:
        serie_data = interregion_ratings[interregion_ratings.serie_name == serie_name]
        date = serie_data.iloc[-1]["date"]
        for region in serie_data.region.unique():
            ratings_in_region_after_serie = _get_player_ratings_of_region_at_a_given_point_in_time(
                ratings_df, region, date
            )
            ratings_after_serie = ratings_in_region_after_serie.loc[:, ["player_rating_after"]]
            ratings_after_serie["region"] = region
            ratings_after_serie["serie_name"] = serie_name            
            data.append(ratings_after_serie)

        nb_interregion_games_for_serie = len(serie_data.index.get_level_values(0).unique())
        nb_interregion_games.append(nb_interregion_games_for_serie)

    region_player_ratings_after_serie_df = pd.concat(data)

    nb_interregion_games_df = pd.Series(
        nb_interregion_games, index=serie_names
    )

    return region_player_ratings_after_serie_df, nb_interregion_games_df


def _get_player_ratings_of_region_at_a_given_point_in_time(
    ratings_df: pd.DataFrame,
    region: str,
    date: str
) -> pd.DataFrame:    
    """
    Return a DataFrame containing all the players ratings belonging to a given region at given date. 
    The ratings are adjusted based on the last meta rating before the date.
    """
    
    region_for_player_at_date = ratings_df[ratings_df.date <= date].reset_index().groupby("player_id").last()["region"]

    player_id_in_region = region_for_player_at_date[region_for_player_at_date == region].index

    six_months_before_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f") - timedelta(days=6*30)
    ratings_from_player_in_region_bf_date = ratings_df[
        (ratings_df.reset_index().player_id.isin(player_id_in_region).values) 
        & (ratings_df.date <= date)
    ]

    last_region_rating_mu, last_region_rating_sigma = ratings_from_player_in_region_bf_date.tail(1).loc[:, ["meta_rating_after_mu", "meta_rating_after_sigma"]].values[0]

    if region != "Other":
        region_regular_season_series = list(MAIN_LEAGUE_SERIE_TOURNAMENT_WHITELIST[region].keys())
        ratings_all_regular_season_series = ratings_from_player_in_region_bf_date[ratings_from_player_in_region_bf_date.serie_name.isin(region_regular_season_series)]

        if ratings_all_regular_season_series.empty:
            # happens for series before the first regular season series (e.g., Worlds 2019)
            ratings_last_regular_season_series = ratings_from_player_in_region_bf_date[ratings_from_player_in_region_bf_date.date >= str(six_months_before_date)]
        else:
            series_whitelist = [
                ratings_all_regular_season_series[ratings_all_regular_season_series.league_id == league_id].serie_name.iloc[-1]
                for league_id in ratings_all_regular_season_series.league_id.unique()
            ] # handles Asia-Pacific region that contains several league for some years
            ratings_last_regular_season_series = ratings_all_regular_season_series[ratings_all_regular_season_series.serie_name.isin(series_whitelist)]
    else:
        
        ratings_last_regular_season_series = ratings_from_player_in_region_bf_date[ratings_from_player_in_region_bf_date.date >= str(six_months_before_date)]

    players_in_last_regular_season = ratings_last_regular_season_series.index.get_level_values(1).unique()

    ratings_from_player_in_region_bf_date = ratings_from_player_in_region_bf_date.sort_values("date")
    last_rating_for_players_in_last_regular_season = ratings_last_regular_season_series.groupby("player_id").last().loc[players_in_last_regular_season]

    last_rating_for_players_in_last_regular_season["meta_rating_after_mu"] = last_region_rating_mu
    last_rating_for_players_in_last_regular_season["meta_rating_after_sigma"] = last_region_rating_sigma
    last_rating_for_players_in_last_regular_season["meta_rating_after"] = lower_bound_rating(last_region_rating_mu, last_region_rating_sigma)

    
    new_overall_ratings = last_rating_for_players_in_last_regular_season.apply(lambda row: combine_contextual_and_meta_ratings(
        row["contextual_rating_after_mu"], row["contextual_rating_after_sigma"],
        row["meta_rating_after_mu"], row["meta_rating_after_sigma"],    
    ), axis=1).values
    new_overall_ratings = np.array([[mu, sigma] for (mu, sigma) in new_overall_ratings])
    last_rating_for_players_in_last_regular_season["player_rating_after_mu"] = new_overall_ratings[:, 0]
    last_rating_for_players_in_last_regular_season["player_rating_after_sigma"] = new_overall_ratings[:, 1]
    last_rating_for_players_in_last_regular_season["player_rating_after"] = last_rating_for_players_in_last_regular_season.apply(
        lambda row: lower_bound_rating(
            row["player_rating_after_mu"], row["player_rating_after_sigma"]
        ),
        axis=1
    )

    return last_rating_for_players_in_last_regular_season


def _create_and_save_meta_rating_evolution(
    ratings_in_region_after_serie: pd.DataFrame, 
    nb_games_in_serie: pd.Series,
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
        nb_games_in_serie.index, 
        nb_games_in_serie.values, 
        color='lightgrey', 
        alpha=1.0
    )
    ax2.set_ylabel("Number of Inter-region Games Played")

    for region in ALL_REGIONS:
        region_data = ratings_in_region_after_serie[ratings_in_region_after_serie['region'] == region]
        color = color_dict[region]
        marker = marker_dict[region]

        mean_data = region_data.groupby("serie_name", observed=False)["player_rating_after"].mean().reset_index()

        sns.lineplot(
            data=mean_data, 
            x="serie_name", 
            y="player_rating_after", 
            label=region, 
            color=color,
            errorbar=None, 
            ax=ax1
        )
        
        
        ax1.scatter(
            mean_data["serie_name"], 
            mean_data["player_rating_after"], 
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