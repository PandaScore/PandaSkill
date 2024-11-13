import pandas as pd
import streamlit as st
import os
from pandaskill.experiments.general.utils import load_data

@st.cache_data
def get_data_from_path(path, index_col):
    data = pd.read_csv(path, index_col=index_col)
    return data

@st.cache_data
def get_all_data():
    data = load_data(
        load_features=True,
        performance_score_path=os.path.join("pandaskill", "artifacts", "data", "app", "pandaskill_pscores.csv"),
        skill_rating_path=os.path.join("pandaskill", "artifacts", "data", "app", "pandaskill_skill_ratings.csv"),
        drop_na=True
    )

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_index()
    
    return data
