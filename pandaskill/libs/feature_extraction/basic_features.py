import pandas as pd

def compute_per_minute_feature(df: pd.DataFrame, feature_name: str) -> pd.Series:
    return df[feature_name] / df["game_length_in_min"]


XP_PER_LEVEL_TABLE = {
    1:0,
    2:280,
    3:660,
    4:1140,
    5:1720,
    6:2400,
    7:3180,
    8:4060,
    9:5040,
    10:6120,
    11:7300,
    12:8580,
    13:9960,
    14:11440,
    15:13020,
    16:14700,
    17:16480,
    18:18360,
}
def compute_xp_per_minute(df: pd.DataFrame) -> pd.Series:
    """ Compute approximate xp per minute using player level as a proxy. """
    df["xp"] = df["level"].map(XP_PER_LEVEL_TABLE)
    xppm = compute_per_minute_feature(df, "xp")
    df.drop("xp", axis=1, inplace=True)
    return xppm

def compute_other_team_stat_from_team_stat(df: pd.DataFrame, team_stat_name: str) -> pd.Series:
    """Get for players of one team, the value of a team stat for the other team."""
    game_totals_game_id_mapping = (df.groupby('game_id')[team_stat_name].sum() // 5).to_dict()
    df["game_total"] = df.reset_index("game_id")["game_id"].map(game_totals_game_id_mapping).values
    other_team_stat = df['game_total'] - df[team_stat_name]
    df.drop("game_total", axis=1, inplace=True)
    return other_team_stat

def compute_kda(df: pd.DataFrame) -> pd.Series:
    """Compute the kill-death-assist ratio (KDA). Having 0 death is counted as 1 death."""
    player_deaths_zero_normalized = df["player_deaths"].copy()
    player_deaths_zero_normalized[player_deaths_zero_normalized == 0] = 1
    return (df["player_kills"] + df["player_assists"]) / player_deaths_zero_normalized

def compute_kla(df: pd.DataFrame) -> pd.Series:
    """Compute the kill-life-assist ratio (KLA). Life is the number of lives the player has played, 
    being equal to number death + 1."""
    return (df["player_kills"] + df["player_assists"]) / (df["player_deaths"] + 1)

def compute_stat_per_gold(df: pd.DataFrame, stat_name: str) -> pd.Series:
    """Normalize a stat by the gold earned by the player."""
    return df[stat_name] / df["gold_earned"]

def compute_stat_per_total_kills(df: pd.DataFrame, stat_name: str) -> pd.Series:
    """Normalize a stat by the total kills of the player."""
    return df[stat_name] / df["total_kills"]  

def compute_stat_per_total_kills_per_gold(df: pd.DataFrame, stat_name: str) -> pd.Series:
    """Normalize a stat by the total kills of the player and the gold earned."""
    return df[stat_name] / df["total_kills"] / df["gold_earned"]

def compute_stat_per_gold_per_life(df: pd.DataFrame, stat_name: str) -> pd.Series:
    """Normalize a stat by the gold earned by the player and the number of lives the player has played."""
    return df[stat_name] / df["gold_earned"] / (df["player_deaths"] + 1)

def _compute_team_stat_from_players_stat(df: pd.DataFrame, player_stat_name: str) -> pd.Series:
    """Sum the player stat for each team."""
    return df.groupby(["game_id", "team_id"])[player_stat_name].transform("sum")

if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({
        "game_id": [1]*10 + [2]*10,
        "player_id": list(range(10)) + list(range(5,10)) + list(range(0,5)),
        "player_stat": [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5,
        "team_id": [1] * 5 + [2] * 5 + [2] * 5 + [1] * 5,
    })
    df = df.set_index(["game_id", "player_id"])

    print(_compute_team_stat_from_players_stat(df, "player_stat"))