import streamlit as st
import datetime as dt
import pandas as pd
import altair as alt
import numpy as np
from pandaskill.app.misc import compute_rating_lower_bound

def display_player_team_page(data):
    """
    Display skill rating evolution for a player or team. Provides the possibility to compare two players or teams.
    """

    st.header("Player / Team Evolution")

    data = data.reset_index()

    data.rename(
        columns={
            "skill_rating_after_mu": "skill_rating_mu", 
            "skill_rating_after_sigma": "skill_rating_sigma", 
            "skill_rating_after": "skill_rating",
            "performance_score": "pscore"
        }, 
        inplace=True
    )

    settings_columns = st.columns([1, 9])
    with settings_columns[0]:
        selection_type = st.selectbox("Select Type", ["Player", "Team"])

    with settings_columns[1]:
        if selection_type == "Player":
            ratings = _get_player_ratings(data)
        elif selection_type == "Team":
            ratings = _get_team_ratings(data)
        else:
            st.warning("Invalid selection type.")

    ratings = _select_ratings_in_time_window(ratings)

    _display_player_evolution(ratings)

def _get_player_ratings(data):
    all_players = data[["player_id", "player_name"]].drop_duplicates()
    player_name_to_id = all_players.set_index("player_name").to_dict()["player_id"]
    player_names = list(player_name_to_id.keys())
    selected_player_names = st.multiselect(
        "Select up to two players to compare:",
        player_names,
        default=["Faker"] if "Faker" in player_names else [],
        max_selections=5 
    )
    if len(selected_player_names) == 0:
        st.warning("Please select at least one player.")
        return
    selected_player_ids = [player_name_to_id[name] for name in selected_player_names]
    ratings = data.loc[data["player_id"].isin(selected_player_ids)].copy()
    ratings['entity_name'] = ratings['player_name']
    
    return ratings

def _get_team_ratings(data):
    all_teams = data[["team_id", "team_name"]].drop_duplicates()
    team_name_to_id = all_teams.set_index("team_name").to_dict()["team_id"]
    team_names = list(team_name_to_id.keys())
    selected_team_names = st.multiselect(
        "Select up to two teams to compare:",
        team_names,
        default=["T1"] if "T1" in team_names else [],
        max_selections=5
    )
    if len(selected_team_names) == 0:
        st.warning("Please select at least one team.")
        return
    selected_team_ids = [team_name_to_id[name] for name in selected_team_names]
    team_data = data.loc[data["team_id"].isin(selected_team_ids)].copy()
    team_data["entity_name"] = team_data["team_name"]
    ratings = team_data.groupby(["entity_name", "game_id", "date"]).agg(
        series_name=("series_name", "first"),
        tournament_name=("tournament_name", "first"),
        pscore=("pscore", "mean"),
        skill_rating_mu=("skill_rating_mu", "mean"),
        skill_rating_sigma=("skill_rating_sigma", lambda x: np.sqrt(np.mean(np.square(x)))),
    ).reset_index()

    ratings["skill_rating"] = compute_rating_lower_bound(ratings["skill_rating_mu"], ratings["skill_rating_sigma"])
    
    return ratings

def _select_ratings_in_time_window(ratings):    
    start_date, end_date = _select_date_range(ratings)

    ratings = ratings.loc[
        (ratings["date"] >= pd.Timestamp(start_date)) & 
        (ratings["date"] <= pd.Timestamp(end_date))
    ]

    ratings = ratings.sort_values(by="date")
    ratings["index"] = range(1, len(ratings) + 1)
    return ratings

def _select_date_range(skill_ratings):
    start_date = skill_ratings["date"].min().date()
    end_date = skill_ratings["date"].max().date()
    date_range = [start_date + dt.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    start_date, end_date = st.select_slider(
        'Select a date range:',
        options=date_range,
        format_func=lambda x: x.strftime('%Y-%m-%d'),
        value=(
            end_date - dt.timedelta(days=365), 
            end_date
        )
    )

    return start_date, end_date

def _display_player_evolution(skill_ratings):
    skill_ratings["skill_rating_997%"] = compute_rating_lower_bound(skill_ratings["skill_rating_mu"], skill_ratings["skill_rating_sigma"])
    secondary_y_axis = "pscore"

    skill_ratings["pscore_mean"] = skill_ratings.groupby("series_name")["pscore"].transform("mean")
    
    comparing_entities = skill_ratings['entity_name'].nunique() > 1
    show_settings_columns = st.columns([2, 8])
    with show_settings_columns[0]:
        show_gaussian_ratings = st.checkbox("Show Gaussian ratings", value=False)
    with show_settings_columns[1]:
        show_pscore = st.checkbox("Show Performance Score", value=False) if not comparing_entities else False

    if comparing_entities and show_pscore:
        st.info("Performance Score display is disabled when comparing entities.")
        show_pscore = False

    base = alt.Chart(skill_ratings).encode(
        alt.X('index:Q').title('Game number')
    )

    nearest = alt.selection_point(nearest=True, on='mouseover',
                    fields=['index'], empty=False)

    tooltip = ['date:T', 'entity_name:N', "series_name:N", "tournament_name:N", 'game_id:N',  'skill_rating:Q', 'skill_rating_mu:Q', 'skill_rating_sigma:Q']
    if show_pscore:
        tooltip.append('pscore:Q')

    if show_gaussian_ratings:
        rating_chart_area = base.mark_area().encode(
            x=alt.X('index:Q', axis=alt.Axis(title='Game number'), ),
            y=alt.Y(f"skill_rating:Q", axis=alt.Axis(title='Skill Rating'), scale=alt.Scale(
                domainMin=float(skill_ratings["skill_rating"].min()) -1,
                domainMax=float(skill_ratings["skill_rating_997%"].max() + 1)
            )),
            y2=alt.Y2(f"skill_rating_997%:Q"),
            color="entity_name:N",
            opacity=alt.value(0.4),
        ).properties(
            width=800,
            height=800
        ).interactive()
        
        metrics_chart_mean = base.mark_line(
            point=True,
            stroke="black",
        ).encode(
            x='index:Q',
            y=alt.Y(f"skill_rating_mu:Q").title(""),
            color="entity_name:N"
        ).properties(
            width=800,
            height=800
        )
        rating_chart = alt.layer(rating_chart_area, metrics_chart_mean)
    else:
        rating_chart = base.mark_line(
            point=True,
            stroke="black",
        ).encode(
            x=alt.Y('index:Q', axis=alt.Axis(title='Game number'), ),
            y=alt.Y('skill_rating:Q', axis=alt.Axis(title='Skill Rating'), scale=alt.Scale(
                domainMin=float(skill_ratings["skill_rating"].min()) -1,
                domainMax=float(skill_ratings["skill_rating"].max() + 1)
            )),
            color="entity_name:N"
        ).properties(
            width=800,
            height=800
        ).interactive()

    if show_pscore:
        metrics_chart = base.mark_point().encode(
            x='index:Q',
            y=f'{secondary_y_axis}:Q',
            color="series_name:N"
        ).properties(
            width=800,
            height=800
        ).interactive()

        metrics_chart_mean = base.mark_line().encode(
            x='index:Q',
            y=alt.Y(f"{secondary_y_axis}_mean:Q").title(""),
            opacity=alt.value(0.8),
            color="series_name:N"
        ).properties(
            width=800,
            height=800
        )

        metrics_chart_mean_area = base.mark_area().encode(
            x='index:Q',
            y=alt.Y(f"{secondary_y_axis}_5%:Q").title(f"{secondary_y_axis}"),
            y2=alt.Y2(f"{secondary_y_axis}_99.7%:Q"),
            opacity=alt.value(0.2),
            color="series_name:N"
        ).properties(
            width=800,
            height=800
        )

    selectors = base.mark_point().encode(
        x='index:Q',
        opacity=alt.value(0),
        tooltip=tooltip,
    ).add_params(
        nearest
    )
    rules = base.mark_rule(color='gray').encode(
        x='index:Q',
    ).transform_filter(
        nearest
    )

    if show_pscore:
        chart = alt.layer(rating_chart, alt.layer(metrics_chart_mean_area, metrics_chart, metrics_chart_mean), selectors, rules).resolve_scale(y='independent').configure_legend(titleColor='black', titleFontSize=14)
    else:
        chart = alt.layer(rating_chart, selectors, rules).resolve_scale(y='independent').configure_legend(titleColor='black', titleFontSize=14)

    st.altair_chart(chart, use_container_width=True)
