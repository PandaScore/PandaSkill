from pandaskill.experiments.skill_rating.visualization import construct_skill_ratings_for_region_after_serie
import streamlit as st
import pandas as pd
import altair as alt

def display_region_page(data):
    st.header("Region Evolution Page")

    data["date"] = data["date"].astype(str)
    ratings_in_region_after_serie, nb_games_in_serie = _get_region_ratings(data)

    desired_serie_order = ratings_in_region_after_serie['serie_name'].unique().tolist()

    num_tournaments = len(desired_serie_order)
    desired_chart_width = max(800, num_tournaments * 50)
    show_top10 = st.checkbox("Show Top 10 Players For Each Region", value=False)
    if show_top10:        
        ratings_in_region_after_serie = ratings_in_region_after_serie.sort_values(
            by=['serie_name', 'region', 'skill_rating_after'], 
            ascending=[True, True, False]
        )
        ratings_in_region_after_serie = ratings_in_region_after_serie.groupby(['serie_name', 'region'], observed=False).head(10).reset_index()
    
    chart = _create_meta_rating_evolution_chart(
        ratings_in_region_after_serie,
        nb_games_in_serie,
        desired_serie_order,
        title='Average Skill Rating Evolution by Region'
    )
    st.altair_chart(chart.properties(width=desired_chart_width), use_container_width=True)

@st.cache_data
def _get_region_ratings(data):
    return construct_skill_ratings_for_region_after_serie(data)

def _create_meta_rating_evolution_chart(
    ratings_in_region_after_serie: pd.DataFrame,
    nb_games_in_serie: pd.Series,
    desired_serie_order: list,
    title: str = None
) -> alt.Chart:
    mean_data = ratings_in_region_after_serie.groupby(['region', 'serie_name'])['skill_rating_after'].mean().reset_index()

    nb_games_in_serie_df = nb_games_in_serie.reset_index()
    nb_games_in_serie_df.columns = ['serie_name', 'nb_games']

    base = alt.Chart(mean_data).encode(
        x=alt.X(
            'serie_name:N',
            axis=alt.Axis(
                title='Tournament',
                labelAngle=-90,
                labelAlign='right',
                labelLimit=500,
                labelFontSize=10,
                labelSeparation=5
            ),
            sort=desired_serie_order
        ),
    )

    line_chart = base.mark_line().encode(
        y=alt.Y('skill_rating_after:Q', axis=alt.Axis(title='Average Skill Rating')),
        color=alt.Color('region:N', legend=alt.Legend(title='Region')),
    )

    points = base.mark_point(size=100).encode(
        y=alt.Y('skill_rating_after:Q', axis=None),
        color=alt.Color('region:N', legend=None),
        shape=alt.Shape('region:N', legend=None),
        tooltip=['region', 'serie_name', alt.Tooltip('skill_rating_after:Q', format='.2f')]
    )

    bar_chart = alt.Chart(nb_games_in_serie_df).mark_bar(opacity=0.3, color='lightgrey').encode(
        x=alt.X(
            'serie_name:N',
            sort=desired_serie_order,
            axis=alt.Axis(
                labels=False,
                title=None
            )
        ),
        y=alt.Y('nb_games:Q', axis=alt.Axis(title='Number of Inter-region Games')),
    )

    chart = alt.layer(
        bar_chart,
        line_chart,
        points
    ).resolve_scale(
        y='independent'
    ).resolve_legend(
        color='shared',
        shape='shared'
    ).resolve_axis(
        y='independent'
    ).properties(
        width=800,
        height=800,
        title=title
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=14,
    ).configure_legend(
        labelFontSize=12,
        titleFontSize=14
    )
    return chart


if __name__ == "__main__":
    from pandaskill.experiments.general.utils import load_data
    import os
    data = load_data(
        True, 
        os.path.join("pandaskill", "artifacts", "data", "app", "pandaskill_pscores.csv"), 
        os.path.join("pandaskill", "artifacts", "data", "app", "pandaskill_skill_ratings.csv"), 
        drop_na=True
    )

    region_ratings, _ = construct_skill_ratings_for_region_after_serie(data)

    print(region_ratings.head())