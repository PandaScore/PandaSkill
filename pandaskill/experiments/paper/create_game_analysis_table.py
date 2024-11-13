"""
This script can be used to generate the table with the performance scores and skill ratings for a given game.
"""

game_id = 36348

performance_score_experiment = "pscore"
skill_rating_experiment = "meta_ffa_openskill"

import os
from os.path import join
import pandas as pd 
import numpy as np
from pandaskill.experiments.general.utils import ARTIFACTS_DIR

data = pd.read_csv(join(ARTIFACTS_DIR, "data", "raw", "game_players_stats.csv"))

performance_scores_experiment_folder = join(ARTIFACTS_DIR, "experiments", performance_score_experiment)
performance_scores = pd.read_csv(
    join(performance_scores_experiment_folder, "performance_score", "performance_scores.csv")
)

skill_ratings = pd.read_csv(
    join(performance_scores_experiment_folder, "skill_rating", skill_rating_experiment, "skill_ratings.csv") 
)

df = pd.merge(
    pd.merge(data, performance_scores, on=['game_id', 'player_id'], how='inner'),
    skill_ratings,
    on=['game_id', 'player_id'], how='inner'
)

df = df.set_index(["game_id", "player_id"])

df_game = df.loc[game_id]

df_game["rating"] = df_game.skill_rating_after
df_game["update"] = df_game["rating"] - df_game.skill_rating_before
df_game = df_game.loc[:, ["player_name", "team_name", "role", "performance_score", "rating", "update"]]
df_game = df_game.sort_values("performance_score", ascending=False)

print(df_game)