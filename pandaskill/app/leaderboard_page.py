import streamlit as st
import datetime as dt
import numpy as np
from pandaskill.experiments.skill_rating.ranking import create_global_player_ranking
from pandaskill.experiments.general.utils import ALL_REGIONS
import matplotlib.pyplot as plt
import seaborn as sns
from pandaskill.app.misc import compute_rating_lower_bound

def display_leaderboard_page(data):
    """
    Display player or team leaderboard with filters for region, role, and date.
    """

    st.header("Leaderboard")

    (
        date, region, role, parameters, ranking_type, since, min_nb_games
    ) = _get_leaderboard_parameters(data)

    st.info(f"Leaderboard at date {date}, with at least {min_nb_games} games since {since}")

    ranking = create_global_player_ranking(data, parameters)

    average_pscore = data.groupby("player_id")["performance_score"].mean()
    ranking["pscore"] = ranking["player_id"].map(average_pscore)

    if region != "All" or role != "All":
        if region != "All":
            ranking = ranking.loc[ranking["region"] == region]
        if role != "All":
            ranking = ranking.loc[ranking["role"] == role]
        ranking["rank"] = range(1, len(ranking) + 1)

    
    if ranking_type == "Team":
        ranking = _create_team_ranking_from_player_ranking(ranking)
    else:   
        ranking = ranking.loc[:, ["rank", "player_name", "team_name", "role", "region", "nb_games", "last_game_date", "pscore", "skill_rating_mu", "skill_rating_sigma", "skill_rating"]]

    ranking_formatted = ranking.rename(columns={
        "rank": "Rank",
        "player_name": "Player",
        "team_name": "Team",
        "role": "Role",
        "region": "Region",
        "nb_games": "Nb Games",
        "last_game_date": "Last Game Date",
        "pscore": "PScore",
        "skill_rating_mu": "Skill Rating Mu",
        "skill_rating_sigma": "Skill Rating Sigma",
        "skill_rating": "Skill Rating (99.7% CI)"
    })

    ranking_formatted = ranking_formatted.set_index("Rank")
    
    st.dataframe(ranking_formatted, use_container_width=True)

    st.info("Player skill ratings are modeled as Gaussian distribution with parameters mu and sigma. They are ranked using the lower bound of the 99.7% confidence interval of the skill rating. See [here](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule) for more information.")
 
    _display_distributions(ranking)

def _get_leaderboard_parameters(data):
    date_default = data["date"].max()

    setting_columns = st.columns(5)
    with setting_columns[0]:
        ranking_type = st.selectbox("Ranking type", ["Player", "Team"], 0)
    with setting_columns[1]:
        date = st.date_input("Date", date_default, key="leaderboard_date")
    with setting_columns[2]:
        min_nb_games = st.number_input("Minimum number of games", 10, 1000, 10, 10)
    with setting_columns[3]:
        all_regions_choice = ["All"] + ALL_REGIONS
        region = st.selectbox("Region", all_regions_choice, 0)
    with setting_columns[4]:    
        all_roles_choice = ["All"] + data["role"].unique().tolist()
        role = st.selectbox("Role", all_roles_choice, 0)

    since = date - dt.timedelta(days=30*6)
    since = since.strftime("%Y-%m-%d")
    parameters = {
        "since": since,
        "min_nb_games": min_nb_games
    }
    data = data.loc[data["date"] <= dt.datetime.combine(date, dt.datetime.min.time())]

    return date, region, role, parameters, ranking_type, since, min_nb_games

def _create_team_ranking_from_player_ranking(ranking):    
    ranking = ranking.sort_values("skill_rating", ascending=False) # so that we select the top 5 players per team
    ranking = ranking.groupby("team_name").agg(
        region=("region", "first"),
        nb_games=("nb_games", "mean"),
        last_game_date=("last_game_date", "max"),
        pscore=("pscore", "mean"),
        skill_rating_mu=("skill_rating_mu", lambda x: np.mean(x[:5])),
        skill_rating_sigma=("skill_rating_sigma", lambda x: np.sqrt(np.mean(np.square(x[:5])))),
    ).reset_index()
    ranking["skill_rating"] = compute_rating_lower_bound(ranking["skill_rating_mu"], ranking["skill_rating_sigma"])
    ranking = ranking.sort_values("skill_rating", ascending=False)
    ranking["rank"] = range(1, len(ranking) + 1)     
    ranking = ranking.loc[:, ["rank", "team_name", "region", "nb_games", "last_game_date", "pscore", "skill_rating_mu", "skill_rating_sigma", "skill_rating"]]
    return ranking

def _display_distributions(ranking):
    st.header("Distributions")
    y_column = st.selectbox('Value', ["skill_rating", "pscore"])
    x_column_choices = ["region", "role"] if "role" in ranking.columns else ["region"]
    x_column = st.selectbox('Group by', x_column_choices)

    color_palette = sns.color_palette()
    df = ranking.copy()
    if x_column == "region":
        region_order_dict = {
            region: i
            for i, region in enumerate(ALL_REGIONS)
        }
        df["region_order"] = df["region"].map(region_order_dict)

        format_region = lambda region: region.replace(" ", "\n").replace("-", "\n")
        df["region"] = df["region"].apply(format_region)
        
        df = df.sort_values("region_order")
        
        all_regions_formatted = [
            format_region(region) for region in ALL_REGIONS
        ]
        color_palette = dict(zip(all_regions_formatted, color_palette))
    else:
        nb_unique_hue = len(df[x_column].unique())
        color_palette = color_palette[:nb_unique_hue]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    sns.violinplot(
        x=x_column,
        y=y_column,
        hue=x_column,
        palette=color_palette,
        data=df,
        inner="box",
        ax=ax
    )
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    plt.tight_layout()

    _, col2, _ = st.columns([1, 4, 1])

    with col2:
        st.pyplot(fig)
