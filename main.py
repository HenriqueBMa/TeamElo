import csv
from datetime import datetime
from collections import defaultdict
from itertools import combinations

# ELO system constants
INITIAL_ELO = 1000
K_FACTOR = 6*32

# Store player data
player_elo = {}
player_history = defaultdict(list)

def read_groups(groups_file):
    groups = []
    with open(groups_file, 'r') as f:
        for line in f:
            players = line.strip().split()
            if players:
                groups.append(players)
    return groups

def merge_players_with_groups(players, groups):
    merged = []
    used = set()
    for group in groups:
        merged.append(group)
        used.update(group)
    for player in players:
        if player not in used:
            merged.append([player])
    return merged

def format_team(team, player_elo):
    return ', '.join(f"{p}({int(player_elo[p])})" for p in team)

def expected_score(elo1, elo2):
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))

def generate_full_balanced_teams(player_list_file):
    # Read player names
    with open(player_list_file, 'r') as f:
        players = [line.strip() for line in f if line.strip()]

    if len(players) < 2:
        print("Need at least 2 players to form teams.")
        return

    # Set Elo to 1000 if player not seen before
    for player in players:
        if player not in player_elo:
            player_elo[player] = INITIAL_ELO

    # Read groups and merge
    groups = read_groups("groups.txt")
    grouped_players = merge_players_with_groups(players, groups)

    # Generate team combinations at group level
    num_groups = len(grouped_players)
    team_size = num_groups // 2

    seen_pairs = set()
    results = []

    for team1_groups in combinations(range(num_groups), team_size):
        team1 = []
        team2 = []
        for i in range(num_groups):
            if i in team1_groups:
                team1.extend(grouped_players[i])
            else:
                team2.extend(grouped_players[i])

        # Ensure balanced size
        if abs(len(team1) - len(team2)) > 1:
            continue

        team1 = tuple(sorted(team1))
        team2 = tuple(sorted(team2))
        pair = tuple(sorted([team1, team2]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        elo_team1 = sum(player_elo[p] for p in team1)
        elo_team2 = sum(player_elo[p] for p in team2)
        diff = abs(elo_team1 - elo_team2)


        if elo_team2 > elo_team1:
            tmp = elo_team1
            elo_team1 = elo_team2
            elo_team2 = tmp
            tmp = team1
            team1 = team2
            team2 = tmp

        results.append((diff, team1, elo_team1, team2, elo_team2))

    # Sort by Elo difference
    results.sort(key=lambda x: x[0])
    #results = results[:30]

    # Print aligned results
    max_team1_str_len = max(len(format_team(team1, player_elo)) for _, team1, _, _, _ in results)
    max_team2_str_len = max(len(format_team(team2, player_elo)) for _, _, _, team2, _ in results)

    print("\n--- Full Balanced Teams ---")
    for diff, team1, elo1, team2, elo2 in results:

        team1_str = format_team(team1, player_elo)
        team2_str = format_team(team2, player_elo)
        expected_team_1_win = expected_score(elo1, elo2)
        expected_team_2_win = 1 - expected_team_1_win

        team1_changes = [int(K_FACTOR * (1 - expected_team_1_win) / len(team1)),
                         int(K_FACTOR * (0.5 - expected_team_1_win) / len(team1)),
                         int(K_FACTOR * (0 - expected_team_1_win) / len(team1))]
        team2_changes = [int(K_FACTOR * (1 - expected_team_2_win) / len(team2)),
                         int(K_FACTOR * (0.5 - expected_team_2_win) / len(team2)),
                         int(K_FACTOR * (0 - expected_team_2_win) / len(team2))]

        print(f"{team1_str:<{max_team1_str_len}} [{int(elo1)}]   vs   "
              f"{team2_str:<{max_team2_str_len}} [{int(elo2)}]   |   "
              f"Δ = {int(diff)}   |   {int(100*expected_team_1_win)}%  |  "
              f"{team1_changes}  |  {team2_changes}")

# Helper to calculate expected score
def expected_score(team_elo, opponent_elo):
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 800))


# Format team for printing
def format_team(players, elos):
    return ", ".join(f"{p} ({int(elos[p])})" for p in players)


# Update Elo ratings after a match
def update_elo(winners, losers, result, match_date):
    # Ensure everyone has an Elo
    for player in winners + losers:
        if player not in player_elo:
            player_elo[player] = INITIAL_ELO

    # Record team Elo sums
    winners_elo = sum(player_elo[p] for p in winners)
    losers_elo = sum(player_elo[p] for p in losers)

    # Calculate expected win probability
    expected_win = expected_score(winners_elo, losers_elo)

    # Change in Elo for the whole team
    team_change = K_FACTOR * (result - expected_win)

    # Distribute the change among players
    per_player_win_change = team_change / len(winners)
    per_player_lose_change = team_change / len(losers)

    # Before update, print match summary
    winners_str = format_team(winners, player_elo)
    losers_str = format_team(losers, player_elo)
    print(
        f"{winners_str} ({int(winners_elo)}) ({'+' + str(int(per_player_win_change))})  vs  {losers_str} ({int(losers_elo)}) ({'-' + str(int(per_player_lose_change))})")

    # Update winners
    for player in winners:
        player_elo[player] += per_player_win_change
        player_history[player].append((match_date, player_elo[player]))

    # Update losers
    for player in losers:
        player_elo[player] -= per_player_lose_change
        player_history[player].append((match_date, player_elo[player]))


# Main function to process matches
def process_matches(input_file, history_output_file, final_elo_output_file, map_stats_output_file):
    # Track how many times each player played on each map
    player_map_counts = defaultdict(lambda: defaultdict(int))

    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        matches = sorted(reader, key=lambda row: datetime.strptime(row['date'], "%Y-%m-%d"))

        player_cnt = defaultdict()

        print("History of past matches:")
        for row in matches:
            date = row['date']
            winners = [p.strip() for p in row['winners'].split(',')]
            losers = [p.strip() for p in row['losers'].split(',')]
            result = float(row['result'])

            if result != -1:
                update_elo(winners, losers, result, date)

        for row in matches[::-1]:

            winners = [p.strip() for p in row['winners'].split(',')]
            losers = [p.strip() for p in row['losers'].split(',')]
            match_map = row['match_map']

            # Count map plays for each player in this match
            for player in winners + losers:

                if player not in player_cnt:
                    player_cnt[player] = 0

                to_add = 1
                if player_cnt[player] < 5:
                    to_add = pow(2, 5 - player_cnt[player])

                player_map_counts[player][match_map] += to_add
                player_cnt[player] += 1

    # Write elo history
    with open(history_output_file, 'w', newline='') as csvfile:
        fieldnames = ['player', 'date', 'elo']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for player, history in player_history.items():
            for date, elo in history:
                writer.writerow({'player': player, 'date': date, 'elo': round(elo, 2)})

    # Write final elo leaderboard
    with open(final_elo_output_file, 'w', newline='') as csvfile:
        fieldnames = ['player', 'final_elo']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        # Sort players by final elo descending
        for player, elo in sorted(player_elo.items(), key=lambda x: x[1], reverse=True):
            writer.writerow({'player': player, 'final_elo': round(elo, 2)})

    # Write player map stats
    with open(map_stats_output_file, 'w', newline='') as csvfile:
        fieldnames = ['player', 'map', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for player, maps in player_map_counts.items():
            for map_name, count in maps.items():
                writer.writerow({'player': player, 'map': map_name, 'count': count})



import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (no Tk needed)
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def graph_func():
    # Load data
    df = pd.read_csv('elo_history.csv', parse_dates=['date'])
    df = df.sort_values(by='date')

    # Map each unique date to an index for x-axis
    unique_dates = df['date'].unique()
    date_index_map = {date: i for i, date in enumerate(unique_dates)}
    df['x'] = df['date'].map(date_index_map)

    # Prepare colors and markers
    colors = plt.cm.tab20.colors  # 20 distinct colors
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']

    color_cycle = itertools.cycle(colors)
    marker_cycle = itertools.cycle(markers)

    # Assign each player a color and marker
    player_styles = {}
    for player in df['player'].unique():
        player_styles[player] = (next(color_cycle), next(marker_cycle))

    # Plot
    plt.figure(figsize=(14, 6))

    for player, data in df.groupby('player'):
        color, marker = player_styles[player]
        plt.plot(data['x'], data['elo'], label=player, color=color, marker=marker)

    # X-axis labels as dates
    plt.xticks(range(len(unique_dates)), [d.strftime('%Y-%m-%d') for d in unique_dates], rotation=45)

    plt.xlabel('Match Update (by date)')
    plt.ylabel('Elo')
    plt.title('Elo Evolution per Match Update')

    # Move legend outside the plot
    plt.legend(title='Players', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig('elo_evolution_steps.png', bbox_inches='tight')


def sort_maps_by_selected_players(map_stats_file, players_file, maps_file):
    # Load selected players into a set
    with open(players_file, 'r') as f:
        selected_players = {line.strip() for line in f if line.strip()}

    # Load maps and their types from maps.csv
    map_types = {}
    with open(maps_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            map_types[row['map']] = row['type']

    # Initialize counts for all maps and types
    map_counts = {map_name: 0 for map_name in map_types.keys()}
    type_counts = {}
    for map_type in map_types.values():
        if map_type not in type_counts:
            type_counts[map_type] = 0

    # Read map stats file and accumulate counts
    with open(map_stats_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            player = row['player']
            map_name = row['map']
            count = int(row['count'])

            if player in selected_players and map_name in map_counts:
                map_counts[map_name] += count
                # Add to the corresponding type count
                map_type = map_types[map_name]
                type_counts[map_type] += count

    # Sort maps by total count (ascending)
    sorted_maps = sorted(map_counts.items(), key=lambda x: x[1])

    print("How many times these players played on each type of map:")

    # Print type summaries
    for map_type, total_count in sorted(type_counts.items()):
        print(f"{map_type}: {total_count}")
    print()

    # Print the sorted list of individual maps
    for map_name, total_count in sorted_maps:
        map_type = map_types[map_name]
        print(f"{map_name} ({map_type}): {total_count}")


if __name__ == "__main__":

    input_file = "matches.csv"  # Input matches
    history_output_file = "elo_history.csv"  # Output player ELO over time
    final_elo_output_file = "final_elo.csv"  # Output final elo ranking
    map_stats_output_file = "map_stats.csv"
    process_matches(input_file, history_output_file, final_elo_output_file, map_stats_output_file)

    player_list_file = "players.txt"

    #generate_balanced_teams(player_list_file)

    generate_full_balanced_teams(player_list_file)

    graph_func()

    print("")
    sort_maps_by_selected_players(map_stats_output_file, 'players.txt', 'maps.csv')
