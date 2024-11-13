import logging
from os.path import join
import pandas as pd
import types
import yaml

logging.basicConfig(level=logging.INFO)

def function_representer(dumper: yaml.Dumper, data: types.FunctionType) -> yaml.ScalarNode:
    if isinstance(data, types.FunctionType):
        return dumper.represent_scalar('!python/name:' + data.__module__ + '.' + data.__name__, '')
    raise TypeError(f"Unable to serialize {data!r}")

yaml.add_multi_representer(types.FunctionType, function_representer)

def save_yaml(data: dict, dir: str, file_name: str) -> None:
    with open(join(dir, file_name), "w") as file:
        yaml.dump(data, file, default_flow_style=False)

ARTIFACTS_DIR = join("pandaskill", "artifacts")

MAIN_REGIONS = ["Korea", "China", "Europe", "North America","Asia-Pacific", "Vietnam", "Brazil", "Latin America"]
ALL_REGIONS = MAIN_REGIONS + ["Other"]
ROLES = ["Top", "Jungle", "Mid", "Bot", "Support"]

def load_data(
    load_features: bool = False,
    performance_score_path: str = None,
    skill_rating_path: str = None,
    drop_na: bool = False
) -> pd.DataFrame:
    raw_data_folder = join(ARTIFACTS_DIR, "data", "raw")
    game_metadata_df = pd.read_csv(join(raw_data_folder, "game_metadata.csv"), index_col=0)
    game_players_stats_df = pd.read_csv(join(raw_data_folder, "game_players_stats.csv"), index_col=(0,1))
    data = game_players_stats_df.join(game_metadata_df, on="game_id", how="left")

    if load_features:
        game_features_df = pd.read_csv(join(ARTIFACTS_DIR, "data", "preprocessing", "game_features.csv"), index_col=(0,1))
        data = pd.concat([data, game_features_df], axis=1)
    
    if performance_score_path:
        performance_scores_df = pd.read_csv(performance_score_path, index_col=(0,1))
        data = pd.concat([data, performance_scores_df], axis=1)

    if skill_rating_path:
        skill_rating_df = pd.read_csv(skill_rating_path, index_col=(0,1))
        data = pd.concat([data, skill_rating_df], axis=1)

    data = data.sort_values(by=["date", "win"], ascending=[True, False])
    
    if drop_na:
        data = data.dropna()
    
    return data
