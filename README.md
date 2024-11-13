# Description

This repository holds the source code and data of the paper "PandaSkill - Player Performance and Skill Rating in
Esports: Application to League of Legends".

Player performances and ratings can be visualized [here](https://pandaskill.streamlit.app/).

# Installation

1. Clone the repository:

```
git clone https://github.com/PandaScore/PandaSkill.git
cd PandaSkill
```

2. Set up the environment: Create a virtual environment and install dependencies:

```
conda create -n pandaskill python=3.12.7
conda activate pandaskill
pip install -r requirements.txt
```

# Data

All the data needed to reproduce the results are located in the `pandaskill/artifacts/data/` folder. More specifically:
```
pandaskill/artifacts/data/
├── app/ # performance scores and skill ratings of PandaSkill, used in the visuzalization app
├── expert_surveys/ # files used for the expert evaluation of the models
├── preprocessing/ # features extracted from the raw data, used in the visualization app
└── raw/ # raw data used to produce the experimental results
```

In particular, the `raw` subfolder contains 3 files: 
- `game_metadata.csv`: metadata of the games
    - `game_id`: ID of the game
    - `date`: date
    - `match_id`: ID of the match (e.g., a BO5 is a match of max 5 games)
    - `tournament_id`: ID of the tournament
    - `tournament_name`: name of the tournament (e.g., Playoffs)
    - `serie_id`: ID of the serie
    - `serie_name`: name of the serie (e.g., LCK Summer 2024)
    - `league_id`: ID of the league
    - `league_name`: name of the league (e.g., LCK)

Note: every game can be incldued in a tree structure such that: `League > Serie > Tournament > Match > Game`.

- `game_players_stats.csv`:
    - `game_id`: ID of the game
    - `player_id`: ID of the player
    - `player_name`: name of the player
    - `team_id`: ID of the player's team
    - `team_name`: name of the player's team
    - `team_acronym`: acronym of the player's team
    - `role`: role of the player (e.g., Mid)
    - `win`: whether the player has won the game or not
    - `game_length`: length of the game in seconds
    - `champion_name`: name of the Champion played by the player
    - `team_kills`: total number of player kills of the player's team
    - `tower_kills`: total number of tower kills of the player's team
    - `inhibitor_kills`: total number of inhibitor kills of the player's team
    - `dragon_kills`: total number of Drake kills of the player's team
    - `herald_kills`: total number of Rift Herald kills of the player's team
    - `baron_kills`: total number of Baron Nashor kills of the player's team
    - `player_kills`: number of player kills of the player
    - `player_deaths`: number of deaths of the player
    - `player_assists`: 
    - `total_minions_killed`: 
    - `gold_earned`: 
    - `level`: 
    - `total_damage_dealt`: 
    - `total_damage_dealt_to_champions`: 
    - `total_damage_taken`: 
    - `wards_placed`: 
    - `largest_killing_spree`: 
    - `largest_multi_kill`: 
- `game_events.csv`:
    - `id`: ID of the event
    - `game_id`: ID of the game
    - `timestamp`: game timestamp in seconds
    - `event_type`: type of the event (e.g., `player_kill`) 
    - `killer_id`: ID of the killer
    - `killed_id`: ID of the killed if it exists
    - `assisting_player_ids`: list of ID of the assisting players
    - `drake_type`: type of the drake (e.g., `infernal`)


# TODO 
- provide arxiv link to paper
- provide doi of software