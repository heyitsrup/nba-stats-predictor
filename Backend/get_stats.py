import argparse
import json
import os
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder, playercareerstats
from pandas import DataFrame

def get_player_stats(player_name):
    player = players.find_players_by_full_name(player_name)

    if player:
        player_id = player[0]['id']
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        career_stats_df = career_stats.get_data_frames()[0]
        seasons = career_stats_df['SEASON_ID'].unique().tolist()

        player_directory = f"Player_Data/{player_name.replace(' ', '_')}"
        os.makedirs(player_directory, exist_ok=True)

        results = []
        for season in seasons:
            game_finder = leaguegamefinder.LeagueGameFinder(player_id_nullable=player_id, season_nullable=season)
            desired_columns = ['TEAM_NAME', 'GAME_DATE', 'MATCHUP','WL', 'PTS', 'REB', 'AST', 'STL', 'BLK']
            games = game_finder.get_data_frames()[0][desired_columns]

            games_dict = games.to_dict(orient='records')

            # Store data in JSON file
            file_name = f"{player_directory}/{player_name.replace(' ', '_')}_{season.replace('-', '_')}_season.json"
            with open(file_name, 'w') as json_file:
                json.dump(games_dict, json_file, indent=4)

            results.append({
                'season': season,
                'file_name': file_name,
                'games_count': len(games_dict)
            })

        return results
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get player stats')
    parser.add_argument('--player_name', required=True, help='Name of the player')
    args = parser.parse_args()

    player_name = args.player_name
    player_stats = get_player_stats(player_name)
    if player_stats:
        print(f"Player stats for {player_name} collected successfully.")
    else:
        print(f"Player {player_name} not found.")
