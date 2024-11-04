import streamlit as st
import pandas as pd

st.title("PandaSkill")

df = pd.read_csv("pandaskill/artifacts/data/data_2019-09-15_2024-09-15.csv")


df = df.loc[:,['game_id', 'player_id', 'date', 'match_id', 'tournament_id',
       'tournament_name', 'serie_id', 'serie_name', 'league_id', 'league_name', 'region',
       'team_id', 'team_name', 'team_acronym', 'player_name', 'role', 'win']]

st.dataframe(df)