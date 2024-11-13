import pandas as pd
from functools import partial

def compute_ewma_ratings(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.sort_values(by=["player_id", "date"])
    ratings_df = df.loc[:, ["performance_score"]].groupby("player_id", group_keys=False) \
        .apply(partial(_compute_ewma_ratings_for_player, alpha=alpha))
    return ratings_df

def _compute_ewma_ratings_for_player(
    player_game_performance_score: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    ratings_after_game = player_game_performance_score.ewm(alpha=alpha, adjust=False).mean()
    ratings_after_game = ratings_after_game["performance_score"]
    ratings_after_game.name = "skill_rating_after"
    ratings_before_game = [0.0, *ratings_after_game.values[:-1]]
    ratings_before_game = pd.Series(
        data=ratings_before_game, 
        index=player_game_performance_score.index,
        name="skill_rating_before"
    )
    skill_ratings = pd.concat([ratings_before_game, ratings_after_game], axis=1)
    return skill_ratings