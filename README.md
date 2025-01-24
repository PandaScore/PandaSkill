# Table of Contents
1. [Description](#description)
2. [Installation](#installation)
3. [Data](#data)
    - [Raw Subfolder Details](#raw-subfolder-details)
4. [Reproducing Results](#reproducing-results)
5. [TODO](#todo)

# Description

This repository holds the source code and data of the paper [PandaSkill - Player Performance and Skill Rating in Esports: Application to League of Legends](https://arxiv.org/abs/2501.10049).

Player performances and ratings can be visualized [here](https://pandaskill.streamlit.app/).

# Installation

1. **Clone the repository with Git LFS:**

    Ensure you have Git LFS installed to handle large files. If Git LFS is not installed, follow these instructions to set it up.

    Then, clone the repository and pull the large files:

    ```bash
    git lfs install
    git clone https://github.com/PandaScore/PandaSkill.git
    cd PandaSkill
    ```

2. **Set up the environment:**
    Create a virtual environment and install dependencies:

    ```bash
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
<details>
  <summary>Details for the `raw` subfolder content</summary>

- `game_metadata.csv`: metadata of the games
    - `game_id`: ID of the game
    - `date`: date in format YYYY-MM-DD HH:MM:SS.ssssss
    - `match_id`: ID of the match (e.g., a BO5 is a match of max 5 games)
    - `tournament_id`: ID of the tournament
    - `tournament_name`: name of the tournament (e.g., Playoffs)
    - `series_id`: ID of the serie
    - `series_name`: name of the series (e.g., LCK Summer 2024)
    - `league_id`: ID of the league
    - `league_name`: name of the league (e.g., LCK)

Note: every game can be included in a tree structure such that: `Game ⊆ Match ⊆ Tournament ⊆ Series ⊆ League`.

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
    - `team_kills`: total number of champion kills of the player's team
    - `tower_kills`: total number of tower kills of the player's team
    - `inhibitor_kills`: total number of inhibitor kills of the player's team (destroying an inhibitor that has respawned is counted as a kill)
    - `dragon_kills`: total number of Drake kills of the player's team
    - `herald_kills`: total number of Rift Herald kills of the player's team
    - `baron_kills`: total number of Baron Nashor kills of the player's team
    - `player_kills`: player's number of champion kills
    - `player_deaths`: player's number of deaths
    - `player_assists`: player's number of assists
    - `total_minions_killed`: player's number of minions killed
    - `gold_earned`: player's total amount of gold earned
    - `level`: player's final level (max 18)
    - `total_damage_dealt`: damage dealt by the player, disregarding the target
    - `total_damage_dealt_to_champions`: player's damage dealt to Champions
    - `total_damage_taken`: player's damage taken, disregarding the source
    - `wards_placed`: player's number of wards placed
    - `largest_killing_spree`: player's largest killing spree
    - `largest_multi_kill`: player's largest multi-kill (max 5)
- `game_events.csv`:
    - `id`: ID of the event
    - `game_id`: ID of the game
    - `timestamp`: game timestamp in seconds
    - `event_type`: type of the event (e.g., `player_kill`) 
    - `killer_id`: ID of the killer
    - `killed_id`: ID of the killed if it exists
    - `assisting_player_ids`: list of ID of the assisting players
    - `drake_type`: type of the drake (e.g., `infernal`)
</details>

# Reproducing Results

To reproduce the results presented in the paper, follow these steps:

1. **Compute the features from the raw data:**
    - `python pandaskill/experiments/preprocess_data.py`
    - This is optional, as features are already precomputed in `pandaskill/artifacts/data/preprocessing`
2. **Compute the performance scores for each game:**
    - `python pandaskill/experiments/run_performance_score_experiment.py`
    - you can edit the configuration in the file itself, configurations used in the paper are provided
3. **Compute the skill ratings:**
    - `python pandaskill/experiments/run_skill_rating_experiment.py`
    - you can edit the configuration in the file itself, configurations used in the paper are provided

Results are located in `pandaskill/artifacts/experiments/`. 
