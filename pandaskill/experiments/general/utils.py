import logging
from os.path import join
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

ROLES = ["top", "jungle", "mid", "bottom", "support"]

PANDASCORE_COLORS = {
    "purple": "#8F63E0",
    "green": "#16E7CF",
    "light_green": "#D1FFFA",
    "pink": "#F22462",
    "light_pink": "#FFE0E9",
    "gold": "#A49879",
    "dirty_paper": "#E7E6E1",
    "paper": "#F3F3F3",
    "greys": ["#C3C3C3", "#7C7C7C", "#3F3F3F", "#2F2F2F", "#16141A"],
}

