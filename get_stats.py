import json
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder, playercareerstats
from pandas import DataFrame
import os

# player_name = "Luka Doncic"
player_name = input('Enter player name:')

player = players.find_players_by_full_name(player_name)

if player:
    player_id = player[0]['id']
    career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
    career_stats_df = career_stats.get_data_frames()[0]
    seasons = career_stats_df['SEASON_ID'].unique().tolist()
else:
    print(f"Player {player_name} not found")
    exit()

player_directory = player_name.replace(' ', '_')
os.makedirs(player_directory, exist_ok=True)

for season in seasons:
    game_finder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id, season_nullable=season)
    desired_columns = ['TEAM_NAME', 'GAME_DATE', 'MATCHUP','WL', 'PTS', 'REB', 'AST', 'STL', 'BLK']
    games = games = game_finder.get_data_frames()[0][desired_columns]

    games_dict = games.to_dict(orient='records')

    # Store data in JSON file
    file_name = f"{player_directory}/{player_name.replace(' ', '_')}_{season.replace('-', '_')}_regular_season_games.json"
    with open(file_name, 'w') as json_file:
        json.dump(games_dict, json_file, indent=4)

    print(f"Game data stored in {file_name}")
