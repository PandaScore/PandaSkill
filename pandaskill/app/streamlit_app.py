import streamlit as st
from pandaskill.app.leaderboard_page import display_leaderboard_page
from pandaskill.app.data import get_all_data
from pandaskill.app.region_page import display_region_page
from pandaskill.app.player_team_page import display_player_team_page
from pandaskill.app.game_page import display_game_page

st.set_page_config(layout="wide", page_icon=":panda_face:", page_title="PandaSkill")
st.title("PandaSkill")
st.info(
    "PandaSkill is an app showing pro League of Legend player performances and skill ratings, following the methodology described in the paper [PandaSkill - Player Performance and Skill Rating in Esports: Application to League of Legends](https://arxiv.org/abs/2109.15098) by [PandaScore](https://pandascore.co/)."
)   

def run():
    data = get_all_data()
    
    tabs = st.tabs([
            "Leaderboard",
            "Player / Team Evolution",
            "Region Evolution",
            "Game Analysis",
        ])
    
    with tabs[0]:
        display_leaderboard_page(data)
    with tabs[1]:
        display_player_team_page(data)
    with tabs[2]:
        display_region_page(data)
    with tabs[3]: 
        display_game_page(data)
        
    st.divider()
    st.markdown("This app is open-source under the [MIT License](https://github.com/PandaScore/PandaSkill/blob/main/LICENSE).")
    st.markdown("Interested in building projects with esports data? Check out the [PandaScore API](https://developers.pandascore.co/docs/introduction)!")

if __name__ == "__main__":
    run()