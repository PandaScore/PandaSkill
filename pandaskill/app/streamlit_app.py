import streamlit as st
from pandaskill.app.leaderboard_page import display_leaderboard_page
from pandaskill.app.data import get_all_data
from pandaskill.app.region_page import display_region_page
from pandaskill.app.player_team_page import display_player_team_page
from pandaskill.app.game_page import display_game_page

st.set_page_config(layout="wide", page_icon=":panda_face:", page_title="PandaSkill")
st.title("PandaSkill")

def run():
    data = get_all_data()
    
    app = st.selectbox(
        "Choose the app",
        [
            "Leaderboard",
            "Player / Team Evolution",
            "Region Evolution",
            "Game Analysis",
        ],
        0
    )

    if data is not None:

        if app == "Leaderboard":
            display_leaderboard_page(data)
        elif app == "Game Analysis":
            display_game_page(data)
        elif app == "Player / Team Evolution":
            display_player_team_page(data)
        elif app == "Region Evolution":
            display_region_page(data)
        else:
            st.error("Unknown app")
    else:
        st.warning("Please select the experiments first")

if __name__ == "__main__":
    run()