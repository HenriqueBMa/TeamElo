import csv
import math
import os
from datetime import datetime
from collections import defaultdict
from itertools import combinations
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (no Tk needed)
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# KARMA system constants
INITIAL_KARMA = 1000
K_FACTOR = 64
process_only = {"FFAx", "TWOx", "1V1"}
WIN_LOSS_DIF_KARMA_WEIGHT = 40

# Store player data
player_karma = {}
player_win_loss_dif = {}
player_history = defaultdict(list)

def position_probability(team_idx, team_karmas, position):
    """
    Probability that team `team_idx` finishes at `position` (0 = 1st, 1 = 2nd, ...),
    i.e. that exactly `position` of the other teams beat it.
    Each opponent's win chance comes from expected_score, treated as independent.
    """
    dist = [1.0]  # dist[n] = probability that exactly n opponents have beaten us so far
    for j, other_karma in enumerate(team_karmas):
        if j == team_idx:
            continue
        p_lose = 1 - expected_score(team_karmas[team_idx], other_karma)
        new = [0.0] * (len(dist) + 1)
        for n, pr in enumerate(dist):
            new[n] += pr * (1 - p_lose)
            new[n + 1] += pr * p_lose
        dist = new
    return dist[position]

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

def format_team(team, inner_player_karma):
    return ', '.join(f"{p}({int(inner_player_karma[p])})" for p in team)

def format_team_with_u_bal(team, inner_player_karma):
    return ', '.join(f"{p}({int(inner_player_karma[p])}+{WIN_LOSS_DIF_KARMA_WEIGHT*player_win_loss_dif[p]})" for p in team)

def expected_score(karma1, karma2):
    dif_sensitivity = ((karma1+karma2)/1000/2)*400 # how many average players we have per team * 400
    #print(dif_sensitivity)
    return 1 / (1 + 10 ** ((karma2 - karma1) / dif_sensitivity))

def generate_full_balanced_teams(inner_player_list_file):
    # Read player names
    with open(inner_player_list_file, 'r') as f:
        players = [line.strip() for line in f if line.strip()]

    if len(players) < 2:
        print("Need at least 2 players to form teams.")
        return

    # Set Karma: use custom initial value if available, otherwise use default
    total_playing_karma = 0
    for player in players:
        if player not in player_karma:
            player_karma[player] = INITIAL_KARMA
        if player not in player_win_loss_dif:
            player_win_loss_dif[player] = 0

    player_karma_for_balancing = dict()
    for player in players:

        player_karma_for_balancing[player] = player_karma[player]+WIN_LOSS_DIF_KARMA_WEIGHT*player_win_loss_dif[player]
        if player_karma_for_balancing[player]==1000:
            player_karma_for_balancing[player] = 700
        total_playing_karma += player_karma[player]

    # Read groups and merge
    groups = read_groups("groups.txt")
    grouped_players = merge_players_with_groups(players, groups)

    # Generate team combinations at group level
    num_groups = len(grouped_players)

    results = []

    karma_dif_threshold = (total_playing_karma/1000/2)*400 # how many average players we have per team * 400
    max_team_karma = (total_playing_karma/2) + karma_dif_threshold/2
    max_team_karma = 100000
    #print(f'tpe: {total_playing_karma/2}   karma_fi: {karma_dif_threshold}    max_karma: {max_team_karma}')
    for r in range(1, num_groups-1):  # r = number of groups in Team1
        for team1_groups in combinations(range(num_groups), r):

            team1 = []
            team2 = []
            for i in range(num_groups):
                if i in team1_groups:
                    team1.extend(grouped_players[i])
                else:
                    team2.extend(grouped_players[i])

            # Compute Karma sums
            team1_karma_for_balancing = sum(player_karma_for_balancing[p] for p in team1)
            team2_karma_for_balancing = sum(player_karma_for_balancing[p] for p in team2)

            # Must meet the new rule
            if team1_karma_for_balancing < team2_karma_for_balancing:  # Must be stronger or equal
                continue

            if team1_karma_for_balancing > max_team_karma:  # Team1 cannot exceed allowed Karma
                continue

            diff = team1_karma_for_balancing - team2_karma_for_balancing
            results.append((diff, team1, team1_karma_for_balancing, team2, team2_karma_for_balancing))

    # Sort by Karma difference
    results.sort(key=lambda x: x[0])
    #results = results[:30]

    # Print aligned results
    max_team1_str_len = max(len(format_team(team1, player_karma_for_balancing)) for _, team1, _, _, _ in results)
    max_team2_str_len = max(len(format_team(team2, player_karma_for_balancing)) for _, _, _, team2, _ in results)

    print("\n--- Full Balanced Teams ---")
    cnt = 0
    for diff, team1, team1_karma_for_balancing, team2, team2_karma_for_balancing in results:
        team1_karma = sum(player_karma[p] for p in team1)
        team2_karma = sum(player_karma[p] for p in team2)

        cnt += 1
        team1_str = format_team(team1, player_karma_for_balancing)
        team2_str = format_team(team2, player_karma_for_balancing)
        expected_team_1_win = expected_score(team1_karma, team2_karma)
        expected_team_2_win = 1 - expected_team_1_win

        cur_K_FACTOR = k_factor_given_teams([team1,team2])
        team1_changes = [int(cur_K_FACTOR * (1 - expected_team_1_win) / len(team1)),
                         int(cur_K_FACTOR * (0 - expected_team_1_win) / len(team1))]
        team2_changes = [int(cur_K_FACTOR * (1 - expected_team_2_win) / len(team2)),
                         int(cur_K_FACTOR * (0 - expected_team_2_win) / len(team2))]

        print(
            f"{cnt} "
            f"{team1_str:<{max_team1_str_len}} [{int(team1_karma_for_balancing)}]   vs   "
            f"{team2_str:<{max_team2_str_len}} [{int(team2_karma_for_balancing)}]   |  "
            f"Δ = {int(diff):>4} | {int(100 * expected_team_1_win):>3}% | "
            f"{team1_changes}  |  {team2_changes}"
        )

def k_factor_given_teams(teams):
    avg_team_size = sum([len(team) for team in teams])/len(teams)
    return K_FACTOR*avg_team_size

# Update Karma ratings after a match
def update_karma_two_teams(winners, losers, result, match_date):
    # Ensure everyone has an Karma
    for player in winners + losers:
        if player not in player_karma:
            player_karma[player] = INITIAL_KARMA

    # Record team Karma sums
    winners_karma = sum(player_karma[p] for p in winners)
    losers_karma = sum(player_karma[p] for p in losers)

    # Calculate expected win probability
    expected_win = expected_score(winners_karma, losers_karma)

    # Change in Karma for the whole team
    cur_K_FACTOR = k_factor_given_teams([winners,losers])
    team_change = cur_K_FACTOR * (result - expected_win)

    # Distribute the change among players
    per_player_win_change = team_change / len(winners)
    per_player_lose_change = team_change / len(losers)

    # Before update, print match summary
    winners_str = format_team(winners, player_karma)
    losers_str = format_team(losers, player_karma)

    win_chance = expected_win
    lose_chance = 1 - expected_win
    print(f"{winners_str} ({int(winners_karma)}) ({'+' + str(int(per_player_win_change))}) [{100 * win_chance:.0f}%]  vs  "
          f"{losers_str} ({int(losers_karma)}) ({'-' + str(int(per_player_lose_change))}) [{100 * lose_chance:.0f}%]")

    #summed_change = 0.0

    # Update winners with decay based on match count
    for player in winners:
        matches_played = player_cnt.get(player, 0)
        decay_multiplier = math.exp(-matches_played / 3)

        actual_change = per_player_win_change + (1 * per_player_win_change * decay_multiplier)
        player_karma[player] += actual_change
        #summed_change += actual_change

    # Update losers with decay based on match count
    for player in losers:
        matches_played = player_cnt.get(player, 0)
        decay_multiplier = math.exp(-matches_played / 3)

        actual_change = -(per_player_lose_change + (1 * per_player_lose_change * decay_multiplier))
        player_karma[player] += actual_change
        #summed_change += actual_change

        '''
    print(summed_change)
    for player in player_karma.keys():
        player_karma[player] -= summed_change/len(player_karma)
        #print(f'{player} : {summed_change/len(player_karma)}')
        '''

    for player in winners + losers:
        player_history[player].append((match_date, player_karma[player]))
        matches_played = player_cnt.get(player, 0)
        player_cnt[player] = matches_played + 1

def update_karma_1v1(winner, loser, match_date):
    # Ensure everyone has an Karma
    for player in [winner, loser]:
        if player not in player_karma:
            player_karma[player] = INITIAL_KARMA

    # Record team Karma sums
    winners_karma = player_karma[winner]
    losers_karma = player_karma[loser]

    # Calculate expected win probability
    expected_win = expected_score(winners_karma, losers_karma)

    # Change in Karma for the whole team
    cur_K_FACTOR = k_factor_given_teams([[winner],[loser]])
    team_change = cur_K_FACTOR * (1 - expected_win)

    # Distribute the change among players
    per_player_win_change = team_change
    per_player_lose_change = team_change

    # Before update, print match summary
    winners_str = format_team([winner], player_karma)
    losers_str = format_team([loser], player_karma)

    win_chance = expected_win
    lose_chance = 1 - expected_win
    print(f"{winners_str} ({int(winners_karma)}) ({'+' + str(int(per_player_win_change))}) [{100 * win_chance:.0f}%]  vs  "
          f"{losers_str} ({int(losers_karma)}) ({'-' + str(int(per_player_lose_change))}) [{100 * lose_chance:.0f}%]")

    # Update winners with decay based on match count

    matches_played = player_cnt.get(winner, 0)
    decay_multiplier = math.exp(-matches_played / 3)

    actual_change = per_player_win_change + (1 * per_player_win_change * decay_multiplier)
    player_karma[winner] += actual_change

    # Update losers with decay based on match count

    matches_played = player_cnt.get(loser, 0)
    decay_multiplier = math.exp(-matches_played / 3)

    actual_change = -(per_player_lose_change + (1 * per_player_lose_change * decay_multiplier))
    player_karma[loser] += actual_change

    for player in [winner, loser]:
        player_history[player].append((match_date, player_karma[player]))
        matches_played = player_cnt.get(player, 0)
        player_cnt[player] = matches_played + 1

# Update Karma ratings for multi-team matches
def update_karma_multi_team(teams, match_date):
    """
    Update Karma for matches with 2+ teams.
    Teams are ordered by placement (1st, 2nd, 3rd, etc.)
    Each team is considered to have lost to all teams to its left
    and won against all teams to its right.
    """
    # Ensure all players have an Karma
    all_players = [player for team in teams for player in team]
    for player in all_players:
        if player not in player_karma:
            player_karma[player] = INITIAL_KARMA

    # Calculate team Karma sums
    team_karmas = [sum(player_karma[p] for p in team) for team in teams]

    #average_karma = sum(team_karmas)/(len(teams))
    #cur_K_FACTOR = K_FACTOR * (average_karma / 1000)
    cur_K_FACTOR = k_factor_given_teams(teams)

    # For each team, calculate total Karma change based on all matchups
    team_changes = [0.0] * len(teams)

    for i, team_i in enumerate(teams):
        # Team i lost to all teams to its left (better placements)
        for j in range(i):
            team_j = teams[j]
            total_players = len(team_i) + len(team_j)
            expected_win = expected_score(team_karmas[i], team_karmas[j])
            # Team i lost this matchup (result = 0)
            team_changes[i] += cur_K_FACTOR * (0 - expected_win)

        # Team i won against all teams to its right (worse placements)
        for j in range(i + 1, len(teams)):
            team_j = teams[j]
            total_players = len(team_i) + len(team_j)
            expected_win = expected_score(team_karmas[i], team_karmas[j])
            # Team i won this matchup (result = 1)
            team_changes[i] += cur_K_FACTOR * (1 - expected_win)

    # Print match summary
    teams_str = " vs ".join([
        f"{format_team(team, player_karma)} ({int(sum(player_karma[p] for p in team))}) "
        f"({'+' if team_changes[i] >= 0 else ''}{int(team_changes[i] / len(teams[i]))}) "
        f"[{100 * position_probability(i, team_karmas, i):.2f}%]"
        for i, team in enumerate(teams)
    ])
    print(teams_str)

    # Distribute changes among players and update with decay
    for i, team in enumerate(teams):
        per_player_change = team_changes[i] / len(team)

        for player in team:
            matches_played = player_cnt.get(player, 0)
            actual_change = per_player_change

            player_karma[player] += actual_change
            if player_karma[player] < 1:
                player_karma[player] = 1
            player_history[player].append((match_date, player_karma[player]))
            player_cnt[player] = matches_played + 1

def create_peak_leaderboard(peak_output_file="peak_karma.csv",
                            output_dir="leaderboard_peak"):
    # Highest Karma each player ever reached, and the date it happened
    peaks = {
        player: max(history, key=lambda h: h[1])
        for player, history in player_history.items() if history
    }

    # Save to a file
    with open(peak_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['rank', 'player', 'peak_karma', 'date'])
        writer.writeheader()
        for rank, (player, (date, karma)) in enumerate(
                sorted(peaks.items(), key=lambda x: x[1][1], reverse=True), start=1):
            writer.writerow({'rank': rank, 'player': player,
                             'peak_karma': round(karma, 2), 'date': date})

    # Images (same style as the main leaderboard)
    peak_values = {player: karma for player, (date, karma) in peaks.items()}
    create_leaderboard_images(peak_values, output_dir=output_dir,
                              value_label="PEAK KARMA", file_prefix="peak_part")

# Main function to process matches
def process_matches(input_file, history_output_file, final_karma_output_file):
    # Track how many times each player played on each map
    player_map_counts = defaultdict(lambda: defaultdict(int))

    with open(input_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        matches = sorted(reader, key=lambda row: datetime.strptime(row['date'], "%Y-%m-%d"))

        # Initialize player_cnt at the function level so update_karma can access it
        global player_cnt
        player_cnt = defaultdict(int)

        print("History of past matches:")
        for row in matches:
            date = row['date']
            result = float(row['result'])

            # Check if this is a multi-team match or legacy format
            teams = [
                [p.strip() for p in team.split(',')]
                for team in row['teams'].split('|')
            ]

            for team_pos in range(len(teams)):
                for player in teams[team_pos]:
                    if player not in player_win_loss_dif:
                        player_win_loss_dif[player] = 0

            if result != -1:
                if len(teams) == 2 and len(teams[0])==1 and len(teams[1])==1:
                    if "1V1" in process_only:
                        update_karma_1v1(teams[0][0], teams[1][0], date)
                elif len(teams) > 2:
                    if "FFA" in process_only:
                        update_karma_multi_team(teams, date)
                else:
                    if "TWO" in process_only:
                        update_karma_two_teams(teams[0], teams[1], result, date)
                # Legacy 2-team format

            if len(teams) == 2 and (len(teams[0]) != 1 or len(teams[1]) != 1):
                for team_pos in range(len(teams)):
                        for player in teams[team_pos]:
                            player_win_loss_dif[player] += len(teams)-1-team_pos
                            player_win_loss_dif[player] -= team_pos
                            #if player=="HellstormBishop":
                                #print(row)
            #print(player_win_loss_dif["HellstormBishop"])

        # Map Counter
        for row in matches[::-1]:
            if 'teams' in row and row['teams']:
                # Multi-team format
                teams = [
                    [p.strip() for p in team.split(',')]
                    for team in row['teams'].split('|')
                ]
                all_players = [player for team in teams for player in team]
            else:
                # Legacy format
                winners = [p.strip() for p in row['winners'].split(',')]
                losers = [p.strip() for p in row['losers'].split(',')]
                all_players = winners + losers

            # Count map plays for each player in this match
            for player in all_players:
                if player not in player_cnt:
                    player_cnt[player] = 0

                player_cnt[player] += 1

    # Write karma history
    with open(history_output_file, 'w', newline='') as csvfile:
        fieldnames = ['player', 'date', 'karma']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for player, history in player_history.items():
            for date, karma in history:
                writer.writerow({'player': player, 'date': date, 'karma': round(karma, 2)})

    # Write final karma leaderboard
    with open("final_u.csv", 'w', newline='') as csvfile:
        fieldnames = ['player', 'wins-losses']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        # Sort players by final karma descending
        for player, karma in sorted(player_win_loss_dif.items(), key=lambda x: x[1], reverse=True):
            writer.writerow({'player': player, 'wins-losses': round(karma, 2)})

    # Write final KARMA leaderboard (spaced / aligned)
    with open(final_karma_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['rank', 'player', 'karma']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        sorted_players = sorted(
            player_karma.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for rank, (player, karma) in enumerate(sorted_players, start=1):
            writer.writerow({
                'rank': f"{rank:>3}",  # right-aligned, width 3
                'player': f"{player:<20}",  # left-aligned, width 20
                'karma': f"{karma:>8.2f}"  # right-aligned, width 8, 2 decimals
            })

    create_leaderboard_images(player_karma)
    create_peak_leaderboard()

def create_leaderboard_images(player_karma, players_per_page=16, output_dir="leaderboard",
                              value_label="KARMA", file_prefix="leaderboard_part"):
    os.makedirs(output_dir, exist_ok=True)
    # 0. Make sure the leaderboard folder exists (keeps all the numbered
    #    leaderboard pages out of the main working directory, since there
    #    can be a lot of them)

    # 1. Sort the players
    sorted_players = sorted(player_karma.items(), key=lambda x: x[1], reverse=True)

    # 2. Chunk the list into groups of 25
    pages = [sorted_players[i:i + players_per_page]
             for i in range(0, len(sorted_players), players_per_page)]

    # 3. Design Settings
    padding = 40
    line_height = 35
    font_size = 24
    img_width = 600

    try:
        font = ImageFont.truetype("cour.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for page_idx, page_data in enumerate(pages):
        # Calculate height for this specific image
        img_height = (len(page_data) * line_height) + (padding * 2) + 60
        image = Image.new('RGB', (img_width, img_height), color=(30, 33, 36))
        draw = ImageDraw.Draw(image)

        # Draw Header with Page Number
        header = f"{'Rank':>5}   {'Player':<20} {value_label:>10}"
        draw.text((padding, padding), header, fill=(255, 215, 0), font=font)
        #draw.text((img_width - 150, padding - 20), f"Page {page_idx + 1}", fill=(150, 150, 150), font=font)
        draw.line([padding, padding + 35, img_width - padding, padding + 35], fill=(100, 100, 100))

        # 4. Draw Players for this page
        y_offset = padding + 60
        for i, (player, karma) in enumerate(page_data):
            # Calculate global rank
            rank = (page_idx * players_per_page) + i + 1

            row_text = f"{rank:>5}   {player:<20} {karma:>10.2f}"

            # Color Logic
            text_color = (255, 255, 255)
            if rank == 1:
                text_color = (255, 215, 0)
            elif rank == 2:
                text_color = (192, 192, 192)
            elif rank == 3:
                text_color = (205, 127, 50)

            draw.text((padding, y_offset), row_text, fill=text_color, font=font)
            y_offset += line_height

        # 5. Save numbered file into the leaderboard folder
        image.save(os.path.join(output_dir, f"{file_prefix}_{page_idx + 1}.png"))


import math

def _place_legend(n_players):
    plt.legend(
        bbox_to_anchor=(0.5, -0.18),
        loc='upper center',
        ncol=9,
        fontsize='small',
        markerscale=0.8,
        frameon=False
    )

def graph_func(min_matches=0):
    # Load data
    df = pd.read_csv('karma_history.csv', parse_dates=['date'])
    df = df.sort_values(by='date')

    # Only keep players who played more than X matches
    match_counts = df['player'].value_counts()
    eligible_players = match_counts[match_counts >= min_matches].index
    df = df[df['player'].isin(eligible_players)]

    # Map each unique date to an index for x-axis
    unique_dates = df['date'].unique()
    date_index_map = {date: i for i, date in enumerate(unique_dates)}
    df['x'] = df['date'].map(date_index_map)

    # Calculate mean Karma for each date
    mean_karma_by_date = []
    extra_karma_by_date = []
    mean_x = []

    for date in unique_dates:
        players_up_to_date = df[df['date'] <= date]['player'].unique()

        latest_karmas = []
        for player in players_up_to_date:
            player_data = df[
                (df['player'] == player) &
                (df['date'] <= date)
            ]
            if not player_data.empty:
                latest_karmas.append(player_data.iloc[-1]['karma'])

        if latest_karmas:
            mean_karma_by_date.append(
                sum(latest_karmas) / len(latest_karmas)
            )
            extra_karma_by_date.append(
                sum(latest_karmas) - 1000 * len(latest_karmas)
            )
            mean_x.append(date_index_map[date])

    # Prepare colors and markers
    colors = plt.cm.tab20.colors
    markers = [
        'o', 's', 'D', '^', 'v', '<', '>', 'p',
        '*', 'h', 'H', '+', 'x', '|', '_'
    ]

    color_cycle = itertools.cycle(colors)
    marker_cycle = itertools.cycle(markers)

    # Assign each player a color and marker
    player_styles = {}
    for player in df['player'].unique():
        player_styles[player] = (
            next(color_cycle),
            next(marker_cycle)
        )

    # Plot
    plt.figure(figsize=(14, 7))

    for player, data in df.groupby('player'):
        color, marker = player_styles[player]
        plt.plot(
            data['x'],
            data['karma'],
            label=player,
            color=color,
            marker=marker
        )

    '''
    # X-axis labels as dates
    plt.xticks(
        range(len(unique_dates)),
        [d.strftime('%Y-%m-%d') for d in unique_dates],
        rotation=45
    )
    '''
    plt.ylabel('Karma')

    title = 'Karma Evolution per Match Update'
    if ("1V1" in process_only and
            "TWO" not in process_only and
            "FFA" not in process_only):
        title = "1v1 Karma Evolution"

    if ("1V1" not in process_only and
            "TWO" not in process_only and
            "FFA" in process_only):
        title = "Diplo Karma Evolution"

    plt.title(title)

    _place_legend(df['player'].nunique())
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        'karma_evolution_steps.png',
        bbox_inches='tight',
        pad_inches=0.2
    )

def graph_last_match_players():
    # Load data
    df = pd.read_csv('karma_history.csv', parse_dates=['date'])
    df = df.sort_values(by='date')

    # Find the last match date
    last_match_date = df['date'].max()

    # Get players who played in the last match
    last_match_players = df[df['date'] == last_match_date]['player'].unique()

    # Filter dataframe to only include those players
    df = df[df['player'].isin(last_match_players)]

    # Map each unique date to an index for x-axis
    unique_dates = df['date'].unique()
    date_index_map = {date: i for i, date in enumerate(unique_dates)}
    df['x'] = df['date'].map(date_index_map)

    # Calculate mean Karma for each date
    mean_karma = df.groupby('date')['karma'].mean()
    mean_x = [date_index_map[date] for date in mean_karma.index]

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
    plt.figure(figsize=(14, 7))

    for player, data in df.groupby('player'):
        color, marker = player_styles[player]
        plt.plot(data['x'], data['karma'], label=player, color=color, marker=marker)

    # Plot mean Karma line
    # plt.plot(mean_x, mean_karma.values, label='Mean Karma', color='black', linewidth=2.5, linestyle='--', zorder=10)

    # X-axis labels as dates
    plt.xticks(range(len(unique_dates)), [d.strftime('%Y-%m-%d') for d in unique_dates], rotation=45)

    #plt.xlabel('Match Update (by date)')
    plt.ylabel('Karma')


    if "1V1" not in process_only and "TWO" in process_only and "FFA" not in process_only:
        title = "2 Teams Karma for Last Match Players"
    elif "1V1" not in process_only and "TWO" not in process_only and "FFA" in process_only:
        title = "Diplomatic Karma for Last Match Players"
    else:
        title = "1v1 Karma for Last Match Players"
    plt.title(title)

    #Move legend outside the plot
    _place_legend(df['player'].nunique())
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig('karma_evolution_last_match.png', bbox_inches='tight', pad_inches=0.3)

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

def get_potential_changes(teams):
    """
    Calculates the Karma change for each team for every possible rank.
    Returns a dictionary mapping Team -> List of changes [if 1st, if 2nd, ...]
    """
    n_teams = len(teams)
    team_karmas = [sum(player_karma.get(p, INITIAL_KARMA) for p in team) for team in teams]

    #average_karma = sum(team_karmas) / (len(teams))
    #cur_K_FACTOR = K_FACTOR * (average_karma / 1000)
    cur_K_FACTOR = k_factor_given_teams(teams)

    final_table = []
    for i, team in enumerate(teams):
        team_name = " & ".join(team)
        outcomes = []
        for rank in range(1, n_teams+1):
            total_change = 0
            opponent_rank = 1
            for j, opponent in enumerate(teams):
                if i == j: continue
                if rank == opponent_rank:
                    opponent_rank += 1
                exp = expected_score(team_karmas[i], team_karmas[j])
                # If i's rank < j's rank, i won.
                # Here we simulate i being at 'rank' and j being at all other spots
                # To simplify for the user, we assume other teams maintain their relative order
                # or we just compare i against every other team.

                # Logic: If I am 1st, I beat everyone. If 2nd, I beat everyone but 1.
                # Because we don't know who is where, we assume 'rank' is the number of
                # teams that beat us.

                # A more accurate 'Pre-match' projection:
                # Total Karma Change = (Wins * K) - (Total Expected Win * K)

                if opponent_rank < rank:
                    # Team i won this matchup (result = 1)
                    total_change += cur_K_FACTOR * (0 - exp)
                else:
                    total_change += cur_K_FACTOR * (1 - exp)
                opponent_rank += 1

            outcomes.append(int(total_change/len(teams[i])))
        final_table.append((team_name, team_karmas[i], outcomes))

    # Sort teams by Karma (Strongest first)
    final_table.sort(key=lambda x: x[1], reverse=True)

    return final_table

def generate_karma_changes_based_on_ranking(groups_file):
    # --- Execution ---

    # 1. Read teams from file
    # Assuming groups.txt format:
    # Player1 Player2
    # Player3 Player4
    teams = []
    try:
        with open("groups.txt", "r") as f:
            for line in f:
                if line.strip():
                    teams.append(line.strip().split())
    except FileNotFoundError:
        # Fallback for demonstration
        print("shitWentBad")

    # --- Calculation & Display ---
    results = get_potential_changes(teams)
    num_ranks = len(teams)

    # Find the longest label to set column width
    max_label_len = max(
        len(f"{team_name} [{int(team_karma)}]")
        for team_name, team_karma, _ in results
    )

    # Build Header
    header = f"{'Team Composition [Total Karma]':<{max_label_len}}"
    for r in range(1, num_ranks + 1):
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(r if r < 20 else (r % 10), 'th')
        header += f" | {str(r) + suffix:>5}"

    print("\nKarma Change Projections (Sorted by Team Strength):")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for team_name, team_karma, outcomes in results:
        label = f"{team_name} [{int(team_karma)}]"
        row = f"{label:<{max_label_len}}"
        for val in outcomes:
            row += f" | {val:>+5}"
        print(row)

    print("\nor\n")

    total = 0
    player_cnt_inner = 0

    team_karmas = [
        sum(player_karma.get(p, INITIAL_KARMA) for p in team)
        for team in teams
    ]

    # Sort teams by total karma, largest first
    sorted_teams = sorted(
        zip(teams, team_karmas),
        key=lambda x: x[1],
        reverse=True
    )

    for team, karma in sorted_teams:
        total += 1 / karma
        player_cnt_inner += len(team)

    for team, karma in sorted_teams:
        label = f"{' & '.join(team)} [{int(karma)}]"
        row = f"{label:<{max_label_len}}   hearts/minute: {player_cnt_inner * (1 / karma) / total}"
        print(row)

def write_match_counts(input_file, output_file="match_counts.csv"):
    counts = defaultdict(int)
    with open(input_file, newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            for team in row['teams'].split('|'):
                for player in team.split(','):
                    player = player.strip()
                    if player:
                        counts[player] += 1

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['rank', 'player', 'matches'])
        for rank, (player, count) in enumerate(
                sorted(counts.items(), key=lambda x: (-x[1], x[0])), start=1):
            writer.writerow([rank, player, count])

def validate_matches(input_file):
    problems = 0
    with open(input_file, newline='') as csvfile:
        for line_no, row in enumerate(csv.DictReader(csvfile), start=2):  # line 1 is the header
            teams = [[p.strip() for p in team.split(',')] for team in row['teams'].split('|')]
            where = f"line {line_no} ({row.get('date', '?')})"

            # Empty player (e.g. "A,,B", a trailing comma, or an empty team)
            if any(p == '' for team in teams for p in team):
                print(f"WARNING: empty player in {where}: {row['teams']}")
                problems += 1

            # Same player in more than one team
            seen = {}
            for team_idx, team in enumerate(teams):
                for p in set(team):
                    if p == '':
                        continue
                    if p in seen:
                        print(f"WARNING: '{p}' is on both team {seen[p] + 1} and team {team_idx + 1} in {where}: {row['teams']}")
                        problems += 1
                    else:
                        seen[p] = team_idx

            # Same player listed twice in the same team
            for team_idx, team in enumerate(teams):
                dupes = {p for p in team if p and team.count(p) > 1}
                for p in dupes:
                    print(f"WARNING: '{p}' appears twice in team {team_idx + 1} in {where}: {row['teams']}")
                    problems += 1

    if problems == 0:
        print("matches.csv looks clean.")
    return problems

if __name__ == "__main__":

    input_file = "matches.csv"  # Input matches
    history_output_file = "karma_history.csv"  # Output player KARMA over time
    final_karma_output_file = "final_karma.txt"  # Output final karma ranking
    validate_matches(input_file)
    process_matches(input_file, history_output_file, final_karma_output_file)

    player_list_file = "players.txt"

    generate_full_balanced_teams(player_list_file)
    #generate_karma_changes_based_on_ranking("groups.txt")

    graph_func()

    graph_last_match_players()

    write_match_counts(input_file)
    #print("")
    #sort_maps_by_selected_players(map_stats_output_file, 'players.txt', 'maps.csv')