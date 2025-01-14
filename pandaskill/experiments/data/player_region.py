from datetime import datetime, timedelta
import logging
import pandas as pd

MAIN_LEAGUE_SERIES_TOURNAMENT_WHITELIST = {
    "Korea": {
        f"LCK {season} {year}": ["Regular Season"]
        for year in range(2019, 2025) 
        for season in ["Spring", "Summer"] 
    },
    "China": {
        **{
            f"LPL {season} {year}": ["Regular Season"]
            for year in range(2019, 2024) 
            for season in ["Spring", "Summer"] 
        },
        "LPL Spring 2024": ["Regular Season"],
        "LPL Summer 2024": ["Group Ascend", "Group Nirvana"]
    },
    "Europe": {
        **{
            f"LEC {season} {year}": ["Regular Season"]
            for year in range(2019, 2025)
            for season in ["Spring", "Summer"]
        },
        "LEC Winter 2023": ["Regular Season"],
        "LEC Winter 2024": ["Regular Season"],
    },
    "North America": {
        f"LCS {season} {year}": ["Regular Season"]
        for year in range(2019, 2025) 
        for season in ["Spring", "Summer"] 
    },
    "Asia-Pacific": {
        "LMS Spring 2019": ["Regular Season"],
        "LMS Summer 2019": ["Regular Season"],
        **{
            f"PCS {season} {year}": ["Regular Season"]
            for year in range(2020, 2025) 
            for season in ["Spring", "Summer"] 
        },
        **{
            f"LJL {season} {year}": ["Regular Season"]
            for year in range(2024, 2025) # only joined PCS region in 2024
            for season in ["Spring", "Summer"] 
        },

        # only joined PCS region in 2023
        "LCO Stage 1 Split 1 2023": ["Regular Season"],
        "LCO Split 2 2023": ["Regular Season"],
        "LCO Split 1 2024": ["Regular Season"],
        "LCO Split 2 2024": ["Regular Season"],
    },
    "Vietnam": {
        **{
            f"VCS {season} {year}": ["Regular Season"]
            for year in range(2019, 2021) 
            for season in ["Spring", "Summer"] 
        },
        "VCS Spring 2021": ["Regular Season"],
        "VCS Winter 2021": ["Regular Season"],
        **{
            f"VCS {season} {year}": ["Regular Season"]
            for year in range(2022, 2025) 
            for season in ["Spring", "Summer"] 
        },
    },
    "Brazil": {
        "CBLOL Summer 2019": ["Regular Season"],
        "CBLOL Winter 2019": ["Regular Season"],
        **{
            f"CBLOL {season} {year}": ["Regular Season"]
            for year in range(2020, 2025) 
            for season in ["Split 1", "Split 2"] 
        },
    },
    "Latin America": {
        f"LLA {season} {year}": (["Phase 1"] if year in [2020, 2021] else ["Regular Season"])
        for year in range(2019, 2025) 
        for season in ["Opening", "Closing"] 
    },
}

regular_season_team_names_before_data = {
    "Korea": ["KT Rolster", "SK telecom T1", "Afreeca Freecs", "Griffin", "DAMWON Gaming", "Kingzone DragonX", "Gen.G", "Hanwha Life Esports", "Liiv SANDBOX"], # exclude Jin Air Green Wings as they've demoted to Challengers Korea
    "China": ["Royal Never Give Up", "JD Gaming", "Oh My God", "EDward Gaming", "LGD Gaming", "Vici Gaming", "Invictus Gaming", "Suning", "Bilibili Gaming", "FunPlus Phoenix", "Rogue Warriors", "Team WE", "Victory Five", "Dominus Esports", "LNG Esports", "Top Esports"],
    "Europe": ["Origen", "Misfits Gaming", "G2 Esports", "Splyce", "Team Vitality", "FC Schalke 04 Esports", "Fnatic", "SK Gaming", "Excel Esports", "Rogue"],
    "North America": ["Echo Fox", "FlyQuest", "Team SoloMid", "Counter Logic Gaming", "Team Liquid", "Cloud9", "OpTic Gaming", "Golden Guardians", "Clutch Gaming", "100 Thieves"],
    "Asia-Pacific": ["Hong Kong Attitude", "J Team", "ahq e-Sports Club", "Flash Wolves", "MAD Team", "G-Rex", "Alpha Esports"],
    "Vietnam": ["EVOS Esports", "FTV Esports", "QTV Gaming", "CERBERUS Esports", "GAM Esports", "Lowkey Esport Vietnam", "Saigon Buffalo", "Team Flash"],
    "Brazil": ["KaBuM! eSports", "CNB e-Sports Club", "Vivo Keyd", "INTZ e-Sports", "ProGaming Esports", "Flamengo eSports", "Uppercut Esports", "Redemption eSports Porto Alegre"],
    "Latin America": ["Furious Gaming", "Isurus", "Kaos Latin Gamers", "Rainbow7", "Pixel Esports Club", "All Knights", "XTEN Esports", "Infinity Esports"],
}

SERIES_NAME_TO_REGION_MAPPING = {
    series_name: region 
    for region, series_dict in MAIN_LEAGUE_SERIES_TOURNAMENT_WHITELIST.items() 
    for series_name in series_dict.keys()
}

def attribute_player_in_game_to_region(df: pd.DataFrame) -> pd.DataFrame:
    regular_season_tournaments = []
    for league, series in MAIN_LEAGUE_SERIES_TOURNAMENT_WHITELIST.items():
        for series_name, tournaments in series.items():
            for tournament in tournaments:
                regular_season_tournaments.append((series_name, tournament))
    mask_df = pd.DataFrame(regular_season_tournaments, columns=['series_name', 'tournament_name'])
    regular_season_df = df.reset_index().merge(
        mask_df, on=["series_name", "tournament_name"]
    ).set_index(["game_id", "player_id"])
    
    main_regions_series_participants_df = regular_season_df.groupby(
        "series_name"
    ).agg({
        "league_name": lambda x: x.iloc[0], 
        "date": lambda x: x.iloc[0], 
        "team_id": lambda team_id_for_series_series: list(team_id_for_series_series.unique())
    })
    main_regions_series_participants_df = main_regions_series_participants_df.sort_values("date")
    
    absolute_start_date = df['date'].min()
    absolute_end_date = df['date'].max()
    main_regions_participants_lookup = _create_main_regions_participants_lookup(
        df, main_regions_series_participants_df, absolute_start_date, absolute_end_date
    )

    df['region'] = df.apply(
        lambda row: _get_current_region_from_team_name(
            row["team_id"], row["date"], main_regions_participants_lookup
        ),
        axis=1
    )

    return df

def _create_main_regions_participants_lookup(
    df: pd.DataFrame, region_rosters_df: pd.DataFrame, absolute_start_date: str, absolute_end_date: str
) -> dict:
    lookup = {}
    for _, row in region_rosters_df.iterrows():
        for team_id in row['team_id']:
            team_id = int(team_id)
            if team_id not in lookup:
                lookup[team_id] = []

            start_date = row['date']
            league_name = row['league_name']

            next_split = region_rosters_df[
                (region_rosters_df.league_name == league_name) &
                (region_rosters_df.date > start_date)
            ]

            if next_split.empty:
                end_date = absolute_end_date
            else:
                end_date = next_split.iloc[0]['date']

            lookup[int(team_id)].append({
                "start_date": start_date,
                "end_date": end_date,
                "league_name": league_name,
                "region": SERIES_NAME_TO_REGION_MAPPING[row.name]
            })
    
    for team_id in lookup:
        lookup[team_id].sort(key=lambda x: x['start_date'])

    # add lookup for teams that played in the regular season before the start of the dataset
    team_name_to_id_map = df.loc[:,["team_id", "team_name"]].set_index("team_name").drop_duplicates().to_dict()["team_id"]
    for region, team_names in regular_season_team_names_before_data.items():
        for team_name in team_names:
            if team_name not in team_name_to_id_map:
                logging.warning(f"Team {team_name} from region {region} not found in the dataset")
            else:
                team_id = team_name_to_id_map[team_name]
                if team_id not in lookup:
                    last_game_date = df[df.team_id == team_id].date.max()
                    last_game_date = datetime.strptime(last_game_date, "%Y-%m-%d %H:%M:%S.%f") + timedelta(days=1)
                    last_game_date = last_game_date.strftime("%Y-%m-%d")

                    lookup[team_id] = [
                        {
                            "start_date": absolute_start_date,
                            "end_date": last_game_date,
                            "league_name": "Unknown",
                            "region": region
                        }
                    ]
                else:
                    start_next_series = lookup[team_id][-1]["start_date"]
                    lookup[team_id].append({
                        "start_date": absolute_start_date,
                        "end_date": start_next_series,
                        "league_name": "Unknown",
                        "region": region
                    })

    return lookup

def _get_current_region_from_team_name(
    team_id: int, date: str, region_rosters_lookup: dict
):
    if team_id not in region_rosters_lookup:
        return "Other"
    
    team_data = region_rosters_lookup[team_id]

    for entry in team_data:
        if entry["start_date"] <= date and entry["end_date"] > date:
            return entry["region"]
        
    return "Other"
    
def manually_correct_team_region(df: pd.DataFrame) -> pd.DataFrame:
    df = _fix_kespa_cup_regions(df)
    df = _fix_demacia_cup_regions(df)
    df = _attribute_region_to_all_star_2020(df)
    df = _attribute_region_to_showmatches_2023(df)

    df.loc[
        (df.series_name == "Prime League 1st Division Spring 2022")
        & (df.team_name == "FC Schalke 04 Esports")
    , "region"] = "Other" # they don't participate in the LEC Spring split that starts few days later

    df.loc[
        (df.series_name == "LMF Opening 2023")
        & (df.team_name == "Globant Emerald")
    , "region"] = "Other" # they did not qualify to the LLA Opening 2023

    return df

def _fix_kespa_cup_regions(df: pd.DataFrame) -> pd.DataFrame:    
    df.loc[
        (df.series_name == "KeSPA Cup 2019")
        & (df.team_name == "T1")
    , "region"] = "Korea" # SK telecom T1 renamed to T1
    df.loc[
        (df.series_name == "KeSPA Cup 2019")
        & (df.team_name == "DRX")
    , "region"] = "Korea" # Kingzone DragonX renamed to DRX
    
    df.loc[
        (df.series_name == "KeSPA Cup 2020")
        & (df.team_name == "Nongshim Red Force")
    , "region"] = "Korea" # Team Dynamics renamed to Nongshim Red Force
    df.loc[
        (df.series_name == "KeSPA Cup 2020")
        & (df.team_name == "BRION")
    , "region"] = "Korea" # Brion Blade renamed to BRION
    
    return df

def _fix_demacia_cup_regions(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[
        (df.series_name == "Demacia Cup 2019")
        & (df.team_name == "FunPlux Phoenix")
    , "region"] = "Other" # FunPlus Phoenix plays with academy roster
    df.loc[
        (df.series_name == "Demacia Cup 2019")
        & (df.team_name == "Invictus Gaming")
    , "region"] = "Other" # Invictus Gaming plays with academy roster + new players

    df.loc[
        (df.series_name == "Demacia Cup 2020")
        & (df.team_name == "ThunderTalk Gaming")
    , "region"] = "China" # Dominus Esports renamed to ThunderTalk Gaming

    df.loc[
        (df.series_name == "Demacia Cup 2021")
        & (df.team_name == "Anyone's Legend")
    , "region"] = "China" # Rogue Warrors renamed to Anyone's Legend
    df.loc[
        (df.series_name == "Demacia Cup 2021")
        & (df.team_name == "Weibo Gaming")
    , "region"] = "China" # Suning renamed to Weibo Gaming

    return df

def _attribute_region_to_all_star_2020(df: pd.DataFrame) -> pd.DataFrame:
    team_name_region_map = {
        'All-Stars LJL': 'Other', 
        'CBLoL Allstars': 'Brazil', 
        'LCK  Queue Kings': 'Korea', 
        'LCK Allstars': 'Korea', 
        'LCK Legends': 'Korea', 
        'LCL Allstars': 'Other', 
        'LCS Allstars': 'North America', 
        'LCS Legends': 'North America', 
        'LCS Queue Kings': 'North America', 
        'LEC Allstars': 'Europe', 
        'LEC Legends': 'Europe', 
        'LEC Queue Kings': 'Europe', 
        'LLA Allstars': 'Latin America', 
        'LPL Allstars': 'China', 
        'LPL Legends': 'China', 
        'LPL Queue Kings': 'China', 
        'OPL Allstars': 'Other', 
        'PCS Allstars': 'Asia-Pacific', 
        'TCL Allstars': 'Other', 
        'VCS Allstars': 'Vietnam'
    }
    for team_name, region in team_name_region_map.items():
        df.loc[
            (df.series_name == "All-Star 2020")
            & (df.team_name == team_name)
        , "region"] = region

    return df

def _attribute_region_to_showmatches_2023(df: pd.DataFrame) -> pd.DataFrame:
    series_name_to_region_map = {
        "Season Kickoff Latin America 2023": "Latin America",
        "Season Kickoff EMEA 2023": "Europe",
        "Season Kickoff Pacific 2023": "Asia-Pacific",
        "Season Kickoff Japan 2023": "Other",
        "Season Kickoff North America 2023": "North America",
        "Season Kickoff Brazil 2023": "Brazil",
        "Season Kickoff Vietnam 2023": "Vietnam",
        "Season Kickoff Korea 2023": "Korea",
    }

    for series_name, region in series_name_to_region_map.items():
        df.loc[
            (df.series_name == series_name)
        , "region"] = region
    
    return df

def attribute_player_region_change(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df = df.sort_values(["player_id", "date"])
    previous_region = df.groupby("player_id")["region"].shift(1)    
    previous_region = previous_region.fillna(df["region"])
    df["region_change"] = df["region"] != previous_region
    df = df.set_index(["game_id", "player_id"])
    df = df.sort_values("date")
    return df