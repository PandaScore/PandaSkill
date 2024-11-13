import streamlit as st
import pandas as pd

def display_game_page(data):
    st.header("Game Analysis")

    selection_method = st.radio("Select game by:", ["Enter Game ID", "Select via League and Series"])
    if selection_method == "Enter Game ID":
        game_id_input = st.text_input("Enter Game ID:", value = "259674")
        if game_id_input:
            game_id = game_id_input.strip()
        else:
            st.warning("Please enter a Game ID.")
            return
    else:
        leagues = data['league_name'].dropna().unique()
        selected_league = st.selectbox("Select League:", leagues, index=leagues.tolist().index("LCK"))

        data_league = data[data['league_name'] == selected_league]

        series = data_league['serie_name'].dropna().unique()
        selected_serie = st.selectbox("Select Series:", series, index=len(series) - 1)
        data_serie = data_league[data_league['serie_name'] == selected_serie]

        tournaments = data_serie["tournament_name"].dropna().unique()
        selected_tournament = st.selectbox("Select Tournament:", tournaments, index=len(tournaments) - 1)
        data_tournament = data_serie[data_serie['tournament_name'] == selected_tournament]

        matches_in_serie = data_tournament['match_id'].dropna().unique()

        game_options = []
        for match_id in matches_in_serie:
            match_data = data_tournament[data_tournament['match_id'] == match_id]
            game_ids = match_data.index.get_level_values(0).dropna().unique()
            team_names = match_data['team_name'].unique()
            if len(team_names) >= 2:
                teams_vs = f"{team_names[0]} vs {team_names[1]}"
            else:
                teams_vs = " vs ".join(team_names)

            sorted_game_ids = sorted(game_ids)
            for idx, gid in enumerate(sorted_game_ids, start=1):
                label = f"{gid} - {teams_vs} - Game {idx}"
                game_options.append({'label': label, 'game_id': gid})

        if not game_options:
            st.warning("No games found in the selected series.")
            return

        game_labels = [option['label'] for option in game_options]
        selected_game_label = st.selectbox("Select Game:", game_labels)

        selected_game_option = next((option for option in game_options if option['label'] == selected_game_label), None)
        if selected_game_option:
            game_id = selected_game_option['game_id']
        else:
            st.warning("Please select a game.")
            return

    if game_id:
        game_id = int(game_id)
        game_data = data.loc[(game_id,)].copy()
        if game_data.empty:
            st.warning("No data found for the selected game.")
            return
        
        st.write

        game_data['rating_update'] = game_data['skill_rating_after'] - game_data['skill_rating_before']

        display_df = game_data[['player_name', 'team_name', 'region', 'role', 'win', 'performance_score', 'skill_rating_after', 'rating_update']]

        display_df = display_df.rename(columns={
            'player_name': 'Player',
            'team_name': 'Team',
            'region': 'Region',
            'role': 'Role',
            'win': 'Win',
            'performance_score': 'PScore',
            'skill_rating_after': 'Skill Rating',
            'rating_update': 'Rating Update'
        })

        display_df = display_df.sort_values(by='PScore', ascending=False)

        display_df = display_df.reset_index(drop=True)

        st.subheader(f"Game ID: {game_id}")

        cols = st.columns([1, 3, 1])
        with cols[0]:
            st.write(f"Date: {pd.to_datetime(str(game_data['date'].values[0])).strftime('%Y-%m-%d')}")
            st.write(f"League: {game_data['league_name'].values[0]}")
            st.write(f"Series: {game_data['serie_name'].values[0]}")
            st.write(f"Tournament: {game_data['tournament_name'].values[0]}")
            st.write(f"Match ID: {game_data['match_id'].values[0]}")

        with cols[1]:
            st.dataframe(display_df, use_container_width=True)

        with cols[2]:
            is_interregion_game = game_data['region'].nunique() > 1
            interregion_game_str = f"Yes: {' vs '.join(game_data['region'].unique())}" if is_interregion_game else "No"
            st.metric("Interregion Game", interregion_game_str)   

        detailed_stats_df = game_data[
            [
                "player_name", 
                "kla",
                "gold_per_minute",
                "xp_per_minute",    
                "cs_per_minute",
                "damage_dealt_per_total_kills",
                "damage_dealt_per_total_kills_per_gold",
                "damage_taken_per_total_kills",
                "damage_taken_per_total_kills_per_gold",
                "largest_multi_kill",
                "largest_killing_spree_per_total_kills",
                "wards_placed_per_minute",
                'objective_contest_loserate',
                'objective_contest_winrate',
                "free_kill_ratio",
                "worthless_death_ratio"
            ]
        ]
        detailed_stats_df = detailed_stats_df.set_index("player_name")

        st.dataframe(detailed_stats_df, use_container_width=True)

    else:
        st.warning("Please select a game to display.")
