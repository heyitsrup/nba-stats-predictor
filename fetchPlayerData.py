from json import dump
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder, playercareerstats

def fetchPlayerData(playerName):
    player = players.find_players_by_full_name(playerName)

    if player:
        playerID = player[0]['id']
        careerStats = playercareerstats.PlayerCareerStats(player_id=playerID)
        careerStatsDF = careerStats.get_data_frames()[0]
        seasons = careerStatsDF['SEASON_ID'].unique().tolist()
    else:
        print(f"Player {playerName} not found")
        exit()

    for season in seasons:
        gameFinder = leaguegamefinder.LeagueGameFinder(player_id_nullable=playerID, season_nullable=season)
        desiredColumns = ['TEAM_NAME', 'GAME_DATE', 'MATCHUP','WL', 'PTS', 'REB', 'AST', 'STL', 'BLK']
        games = games = gameFinder.get_data_frames()[0][desiredColumns]

        gamesDict = games.to_dict(orient='records')

        # Store data in JSON file
        fileName = f"data/raw/{playerName}/{season}.json"
        with open(fileName, 'w') as JSONFile:
            dump(gamesDict, JSONFile, indent=4)

        print(f"Game data stored in {fileName}")