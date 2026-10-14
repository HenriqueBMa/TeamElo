"""
Elo Simulation Environment
===========================
Generates a synthetic history of matches to stress-test the Karma/Elo engine
that already lives in main.py.

The trick of a simulation like this is that every player secretly has a fixed
"actual" skill rating that the Elo system is NEVER allowed to see. The Elo
system only ever sees match results (who beat who) and has to guess at the
truth from those results alone, exactly like in real life. At the end we
reveal the hidden ratings so you can see how close the system's Karma got to
the truth.

Pipeline per match:
    1. Every returning ("old") player has a personal "willingness" score
       between 0 and 1 -- how likely they are to want to play the next
       match. For each of them: roll against willingness to see if they
       *want* to play, then roll an extra 80% check to see if they're
       actually free to (`roll_returning_players`) -- a 20% chance they're
       busy even when willing.
    2. Decide how many brand-new players show up this match. This is a
       gaussian draw whose average and spread scale with how many old
       players ended up playing: average = old_playing/20,
       std_dev = old_playing/10 (`pick_new_player_count`).
    3. Give each brand-new player a secret hidden rating (never exposed to
       Karma) AND a fresh random willingness between 0 and 1
       (`make_new_player`).
    4. Split everyone playing into the two most balanced teams possible,
       using the same "known" Karma the real system would use
       (`balanced_split`).
    5. Decide who *actually* wins using the HIDDEN ratings, run through
       main.expected_score() to get a win probability, then flip a weighted
       coin (`decide_winner`).
    6. Feed that result into the exact same update math main.py's
       update_karma_two_teams() uses (`apply_match_result`), and log the row
       for matches.csv.
    7. Update everyone's willingness for next time (`update_willingness`):
       players who played this match get their willingness multiplied by
       1.1 or divided by 1.1 (picked randomly per player); everyone else in
       the pool -- known players who sat this one out -- get multiplied by
       1.025 or divided by 1.025 instead.

After NUM_MATCHES matches, matches.csv is handed to main.py's own
process_matches() -- the exact, already-existing code that produces
final_karma.txt / final_karma.csv / karma_history.csv / the leaderboard
images / the karma graphs -- so nothing about the reporting side is
reimplemented here. We only add one more column on top: each player's hidden
actual rating, plus the "error" between what Karma landed on and the truth.

Every number the spec called out is a named constant in SimConfig below.
"""

import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import main  # the existing Elo/Karma engine: expected_score(), process_matches(), etc.


# ============================================================================
# CONFIG -- every "magic number" from the spec lives here and only here.
# ============================================================================
@dataclass
class SimConfig:
    # --- how long the simulation runs ---
    NUM_MATCHES: int = 100

    # --- turnout mechanics ---
    # Safety cap only: balanced_split does an exact search over every possible
    # team split (2^n), so the roster actually fed into it is capped here so
    # a match can never grow large enough to make that search intractable.
    MIN_PLAYERS_PER_MATCH: int = 6
    # There's no pool to draw "old" players from before the very first match,
    # so that one match is bootstrapped with this many brand-new players
    # instead of going through the usual old_playing/20 formula.
    BOOTSTRAP_NEW_PLAYERS: int = 6

    # How a player's willingness-to-play drifts after each match: played ->
    # *1.1 or /1.1 (random per player); sat out -> *1.025 or /1.025.
    WILLINGNESS_PLAYED_FACTOR: float = pow(1.01,10)
    WILLINGNESS_IDLE_FACTOR: float = pow(1.01,1)
    # Willingness is a probability, so it's always clamped back into range.
    WILLINGNESS_MIN: float = -400
    WILLINGNESS_MAX: float = 100

    STARTING_WILLINGNESS_MIN: float = -400
    STARTING_WILLINGNESS_MAX: float = 20


    # --- hidden ("actual") skill given to brand-new players ---
    NEW_PLAYER_RATING_MEAN: float = 1000.0
    NEW_PLAYER_RATING_SIGMA: float = 0.3

    # --- what the Elo system itself believes about a brand-new player ---
    INITIAL_KNOWN_KARMA: float = main.INITIAL_KARMA   # 1000, straight from main.py

    # --- Elo mechanics, mirroring main.py's update_karma_two_teams() exactly ---
    K_FACTOR: int = main.K_FACTOR              # 32
    WIN_LOSS_DIF_KARMA_WEIGHT: float = 40.0    # the "20 * win_loss_dif" nudge used for team balancing
    DECAY_TAU: float = 3.0                     # matches main.py's math.exp(-matches_played / 3)

    # --- bookkeeping ---
    START_DATE: date = date(2024, 1, 1)
    DAYS_BETWEEN_MATCHES: int = 1
    RANDOM_SEED: Optional[int] = 42            # None = a fresh random simulation every run
    NEW_PLAYER_NAME_PREFIX: str = "SimPlayer"

    MATCHES_CSV: str = "fake_matches.csv"
    HISTORY_CSV: str = "karma_history.csv"
    FINAL_KARMA_TXT: str = "final_karma.txt"
    MAP_STATS_CSV: str = "map_stats.csv"       # unused placeholder arg required by main.process_matches
    REPORT_CSV: str = "simulation_report.csv"
    BALANCE_CSV: str = "match_balance_report.csv"


# ============================================================================
# Player state
# ============================================================================
@dataclass
class SimPlayer:
    name: str
    actual_rating: float          # HIDDEN ground truth skill -- fixed for the whole run
    known_karma: float            # what the Elo system currently believes
    willingness: float      # 0-1, how likely they are to want to play the next match
    win_loss_dif: int = 0
    matches_played: int = 0


# ============================================================================
# Step 1: which returning players show up
# ============================================================================
def roll_returning_players(cfg: SimConfig, pool: Dict[str, SimPlayer]) -> List[str]:
    """
    Every player already in the pool has a `willingness` (0-1). For each one:
    first roll against willingness to see if they *want* to play, then --
    only if they passed that -- roll an 80% check (AVAILABILITY_CHANCE) to
    see if they're actually free, simulating a 20% chance they're willing
    but busy.
    """
    playing = []
    particular_availability = random.random()*random.random()*random.random()
    for name, sp in pool.items():
        if random.random() < sp.willingness and random.random() < particular_availability:
            playing.append(name)
    return playing


# ============================================================================
# Step 2+3: how many brand-new players show up, and giving them a rating
# ============================================================================
def pick_new_player_count(cfg: SimConfig, num_old_playing: int, pool_is_empty: bool) -> int:
    """
    average = num_old_playing / 20, std_dev = num_old_playing / 10.
    There's no one in the pool yet for match 1 (num_old_playing is always 0
    then), so that first match is bootstrapped with a fixed player count
    instead of a gaussian draw around an average of zero.
    """
    if pool_is_empty:
        return cfg.BOOTSTRAP_NEW_PLAYERS

    average = num_old_playing / 25.0
    std_dev = num_old_playing / 10.0
    n = int(round(random.gauss(average, std_dev)))
    return max(0, n)


def make_new_player(cfg: SimConfig, player_id: int) -> SimPlayer:
    name = f"{cfg.NEW_PLAYER_NAME_PREFIX}{player_id}"

    mu = (
            math.log(cfg.NEW_PLAYER_RATING_MEAN)
            - (cfg.NEW_PLAYER_RATING_SIGMA ** 2) / 2
    )
    actual = random.lognormvariate(mu, cfg.NEW_PLAYER_RATING_SIGMA)

    # New players get a fresh, random likelihood of wanting to play again.
    if player_id <= cfg.BOOTSTRAP_NEW_PLAYERS:
        willingnessP = random.uniform(-10, cfg.STARTING_WILLINGNESS_MAX)
    else:
        willingnessP = random.uniform(cfg.STARTING_WILLINGNESS_MIN, cfg.STARTING_WILLINGNESS_MAX)
    willingness = pow(1.01,willingnessP)
    # The Elo system has no idea about `actual` -- it only ever sees INITIAL_KNOWN_KARMA.
    return SimPlayer(
        name=name,
        actual_rating=actual,
        known_karma=cfg.INITIAL_KNOWN_KARMA,
        willingness=willingness,
    )

def pick_roster(
    cfg: SimConfig, pool: Dict[str, SimPlayer], new_player_counter: int
) -> Tuple[List[str], int]:
    """
    Pick this match's roster.

    Each attempt rolls which "old" (pool) players return, then generates as
    many brand-new candidate players as pick_new_player_count() calls for.
    If old + new doesn't reach MIN_PLAYERS_PER_MATCH, the whole attempt is
    thrown away and retried -- and the candidate new players from a thrown-
    away attempt must NOT be kept around: only players from the attempt
    that actually becomes the final roster get a permanent name and get
    added to `pool`. Committing them earlier (on a rejected attempt) would
    leave orphaned players sitting in the pool who never played the match
    that "created" them, only to be picked up as a "returning" player in
    some later match despite never having played before.
    """
    pool_was_empty = len(pool) == 0

    while True:
        old_names = roll_returning_players(cfg, pool)

        num_new = pick_new_player_count(cfg, len(old_names), pool_was_empty)

        # Generate this attempt's candidates, but don't touch `pool` or
        # assign them a permanent name until we know this attempt is kept.
        candidates = [make_new_player(cfg, 0) for _ in range(num_new)]

        if len(old_names) + len(candidates) >= cfg.MIN_PLAYERS_PER_MATCH:
            new_names = []
            for sp in candidates:
                new_player_counter += 1
                sp.name = f"{cfg.NEW_PLAYER_NAME_PREFIX}{new_player_counter}"
                pool[sp.name] = sp
                new_names.append(sp.name)

            return old_names + new_names, new_player_counter

        # Otherwise: this attempt's candidates are simply dropped and we
        # retry -- nothing about them was ever recorded in `pool`.


# ============================================================================
# Step 7: drift everyone's willingness after the match is played
# ============================================================================
def update_willingness(cfg: SimConfig, pool: Dict[str, SimPlayer], roster: List[str]) -> None:
    """
    Participants: willingness *1.1 or /1.1 (random per player).
    Everyone else in the pool (didn't play this match): willingness *1.025
    or /1.025 (random per player). Always clamped back into [0, 1].
    """
    roster_set = set(roster)
    for name, sp in pool.items():
        factor = cfg.WILLINGNESS_PLAYED_FACTOR if name in roster_set else cfg.WILLINGNESS_IDLE_FACTOR
        sp.willingness *= factor if random.random() < 0.5 else 1.0 / factor
        sp.willingness = min(pow(1.01, cfg.WILLINGNESS_MAX), max(pow(1.01, cfg.WILLINGNESS_MIN), sp.willingness))


from bisect import bisect_left


def balanced_split(
    players: List[str],
    rating_of: Callable[[str], float],
) -> Tuple[List[str], List[str]]:
    n = len(players)

    if n < 2:
        return list(players), []

    values = [rating_of(p) for p in players]
    total = sum(values)
    target = total / 2

    # ------------------------------------------------------------------
    # Small enough: meet-in-the-middle
    # ------------------------------------------------------------------
    if n <= 20:
        mid = n // 2

        left_values = values[:mid]
        right_values = values[mid:]

        def generate_subsets(vals):
            subsets = []

            for mask in range(1 << len(vals)):
                total_value = 0.0

                for i, value in enumerate(vals):
                    if mask & (1 << i):
                        total_value += value

                subsets.append((total_value, mask))

            return subsets

        left = generate_subsets(left_values)
        right = generate_subsets(right_values)

        # Sort by rating so we can binary-search the closest complement.
        right.sort(key=lambda x: x[0])
        right_sums = [x[0] for x in right]

        best_diff = float("inf")
        best_left_mask = 0
        best_right_mask = 0

        for left_sum, left_mask in left:
            wanted = target - left_sum

            index = bisect_left(right_sums, wanted)

            # Check the closest value on either side of the insertion point.
            for j in (index - 1, index):
                if 0 <= j < len(right):
                    right_sum, right_mask = right[j]

                    diff = abs(
                        2 * (left_sum + right_sum) - total
                    )

                    if diff < best_diff:
                        best_diff = diff
                        best_left_mask = left_mask
                        best_right_mask = right_mask

        team_a = []

        for i in range(mid):
            if best_left_mask & (1 << i):
                team_a.append(players[i])

        for i in range(n - mid):
            if best_right_mask & (1 << i):
                team_a.append(players[mid + i])

        team_a_set = set(team_a)
        team_b = [p for p in players if p not in team_a_set]

        return team_a, team_b

    # ------------------------------------------------------------------
    # Large rosters: greedy + local improvement
    #
    # Exact optimization becomes increasingly expensive. Instead,
    # construct a very good balanced partition quickly.
    # ------------------------------------------------------------------
    indexed = list(enumerate(values))

    # Put strongest players first.
    indexed.sort(key=lambda x: x[1], reverse=True)

    team_a = []
    team_b = []
    sum_a = 0.0
    sum_b = 0.0

    # Greedy partition.
    for index, value in indexed:
        if sum_a <= sum_b:
            team_a.append(index)
            sum_a += value
        else:
            team_b.append(index)
            sum_b += value

    # Local improvement: repeatedly try swapping one player from each
    # team if that makes the teams more equal.
    improved = True

    while improved:
        improved = False
        current_diff = abs(sum_a - sum_b)

        best_swap = None
        best_diff = current_diff

        for i in team_a:
            for j in team_b:
                new_a = sum_a - values[i] + values[j]
                new_b = sum_b - values[j] + values[i]
                diff = abs(new_a - new_b)

                if diff < best_diff:
                    best_diff = diff
                    best_swap = (i, j)

        if best_swap is not None:
            i, j = best_swap

            team_a.remove(i)
            team_b.remove(j)

            team_a.append(j)
            team_b.append(i)

            sum_a = sum_a - values[i] + values[j]
            sum_b = sum_b - values[j] + values[i]

            improved = True

    return (
        [players[i] for i in team_a],
        [players[i] for i in team_b],
    )

# ============================================================================
# Step 5: use the HIDDEN ratings (and the existing expected_score function)
# to decide who actually wins
# ============================================================================
def decide_winner(
    team_a: List[str], team_b: List[str], pool: Dict[str, SimPlayer]
) -> Tuple[List[str], List[str], float, float, float]:
    actual_a = sum(pool[p].actual_rating for p in team_a)
    actual_b = sum(pool[p].actual_rating for p in team_b)

    p_a_wins = main.expected_score(actual_a, actual_b)  # reuse the existing function, verbatim

    if random.random() < p_a_wins:
        winners, losers = team_a, team_b
    else:
        winners, losers = team_b, team_a

    # How lopsided this match REALLY was, based on hidden actual ratings:
    # p_a_wins == 0.5 is a perfect coin flip; the further it strays toward
    # 0 or 1, the more one-sided the match actually was, regardless of what
    # the balancing step *thought* it was doing with known Karma.
    return winners, losers, p_a_wins, actual_a, actual_b


# ============================================================================
# Step 6: apply the result -- identical math to main.update_karma_two_teams,
# just working against our own SimPlayer pool instead of main.py's globals
# ============================================================================
def apply_match_result(
    cfg: SimConfig, winners: List[str], losers: List[str], pool: Dict[str, SimPlayer]
) -> None:
    winners_karma = sum(pool[p].known_karma for p in winners)
    losers_karma = sum(pool[p].known_karma for p in losers)
    expected_win = main.expected_score(winners_karma, losers_karma)

    cur_k = cfg.K_FACTOR * (len(winners) + len(losers))
    team_change = cur_k * (1 - expected_win)   # result is always 1: `winners` really did win

    per_win = team_change / len(winners)
    per_lose = team_change / len(losers)

    for p in winners:
        sp = pool[p]
        decay = math.exp(-sp.matches_played / cfg.DECAY_TAU)
        sp.known_karma += per_win + per_win * decay
        sp.win_loss_dif += 1
        sp.matches_played += 1

    for p in losers:
        sp = pool[p]
        decay = math.exp(-sp.matches_played / cfg.DECAY_TAU)
        sp.known_karma -= per_lose + per_lose * decay
        sp.win_loss_dif -= 1
        sp.matches_played += 1


# ============================================================================
# Main simulation loop
# ============================================================================
def run_simulation(cfg: SimConfig) -> Tuple[Dict[str, SimPlayer], List[dict]]:
    random.seed(cfg.RANDOM_SEED)

    pool: Dict[str, SimPlayer] = {}
    new_player_counter = 0
    rows = []
    balance_records = []

    for i in range(cfg.NUM_MATCHES):
        roster, new_player_counter = pick_roster(cfg, pool, new_player_counter)

        def balancing_rating(p: str, _pool=pool) -> float:
            sp = _pool[p]
            return sp.known_karma + cfg.WIN_LOSS_DIF_KARMA_WEIGHT * sp.win_loss_dif

        team_a, team_b = balanced_split(roster, balancing_rating)
        winners, losers, p_a_wins, actual_a, actual_b = decide_winner(team_a, team_b, pool)
        apply_match_result(cfg, winners, losers, pool)
        update_willingness(cfg, pool, roster)

        match_date = cfg.START_DATE + timedelta(days=i * cfg.DAYS_BETWEEN_MATCHES)
        rows.append(
            {
                "date": match_date.isoformat(),
                "teams": "|".join([",".join(winners), ",".join(losers)]),
                "result": 1,
            }
        )

        # Re-attach the actual-Elo sums to whichever side actually won, so we
        # can report from the winner/loser perspective rather than the
        # arbitrary team_a/team_b labels balanced_split happened to produce.
        if winners is team_a:
            winner_actual_elo, loser_actual_elo = actual_a, actual_b
            winner_win_prob = p_a_wins
        else:
            winner_actual_elo, loser_actual_elo = actual_b, actual_a
            winner_win_prob = 1 - p_a_wins

        # Balance of THIS match, judged only by the hidden actual ratings
        # (never by known Karma -- that would just measure the balancer
        # against itself). favorite_win_prob is always >= 0.5; winner_win_prob
        # is NOT -- a low value here means an upset (the underdog won).
        favorite_win_prob = max(p_a_wins, 1 - p_a_wins)
        imbalance = 2 * (favorite_win_prob - 0.5)  # 0 = coin flip, 1 = total mismatch
        balance_records.append(
            {
                "match_index": i,
                "date": match_date.isoformat(),
                "num_players": len(roster),
                "winner_size": len(winners),
                "loser_size": len(losers),
                "winner_actual_elo": round(winner_actual_elo, 1),
                "loser_actual_elo": round(loser_actual_elo, 1),
                "winner_win_prob": round(winner_win_prob, 4),
                "favorite_win_prob": favorite_win_prob,
                "imbalance": imbalance,
            }
        )

    with open(cfg.MATCHES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "teams", "result"])
        writer.writeheader()
        writer.writerows(rows)

    return pool, balance_records


# ============================================================================
# Reporting -- hands off to main.py's OWN existing code for the "official"
# final Karma / win-loss numbers (and all its usual side-effect outputs:
# karma_history.csv, final_karma.csv/.txt, leaderboard images, graphs), then
# adds the one column that only the simulation knows: the hidden actual
# rating.
# ============================================================================
def build_report(
    cfg: SimConfig, pool: Dict[str, SimPlayer], balance_records: List[dict]
) -> List[dict]:
    main.player_karma.clear()
    main.player_win_loss_dif.clear()
    main.player_history.clear()

    main.process_matches(cfg.MATCHES_CSV, cfg.HISTORY_CSV, cfg.FINAL_KARMA_TXT)

    rows = []
    for name, sp in pool.items():
        known_karma = main.player_karma.get(name, cfg.INITIAL_KNOWN_KARMA)
        win_loss = main.player_win_loss_dif.get(name, 0)
        rows.append(
            {
                "player": name,
                "final_karma": round(known_karma, 1),
                "win_loss_dif": win_loss,
                "matches_played": sp.matches_played,
                "actual_rating": round(sp.actual_rating, 1),
                "karma_error": round(known_karma - sp.actual_rating, 1),
                "feel": round(sp.willingness, 5)
            }
        )

    rows.sort(key=lambda r: r["final_karma"], reverse=True)

    with open(cfg.REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "player",
                "final_karma",
                "win_loss_dif",
                "matches_played",
                "actual_rating",
                "karma_error",
                "feel"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'Player':<20}{'Karma':>10}{'W/L':>6}{'Played':>8}{'Actual':>10}{'Error':>10}{'Feel':>10}"
    print("\n--- Simulation Report (Known Karma vs. HIDDEN Actual Rating) ---")
    print(header)
    print("-" * len(header))
    total_absolute_wl = 0
    total_absolute_error = 0
    mature_cnt = 0
    for r in rows:
        if r['matches_played']>=0:
            total_absolute_wl += abs(r['win_loss_dif'])
            total_absolute_error += abs(r['karma_error'])
            mature_cnt += 1
            print(
                f"{r['player']:<20}{r['final_karma']:>10}{r['win_loss_dif']:>6}"
                f"{r['matches_played']:>8}{r['actual_rating']:>10}{r['karma_error']:>10}{r['feel']:>10}"
            )


    print(f"avg abs W/L {total_absolute_wl/mature_cnt}")
    print(f"avg abs error {total_absolute_error/mature_cnt}")

    # --- average match balancing, judged by hidden actual ratings ---
    with open(cfg.BALANCE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "match_index",
                "date",
                "num_players",
                "winner_size",
                "loser_size",
                "winner_actual_elo",
                "loser_actual_elo",
                "winner_win_prob",
                "favorite_win_prob",
                "imbalance",
            ],
        )
        writer.writeheader()
        writer.writerows(balance_records)

    '''
    print("\n--- Per-Match Balance (winner vs. loser, judged on hidden actual Elo) ---")
    bal_header = (
        f"{'#':>5}{'Date':>12}{'Win size':>9}{'Lose size':>10}"
        f"{'Win Actual':>12}{'Lose Actual':>12}{'Win TrueProb':>13}"
    )
    print(bal_header)
    print("-" * len(bal_header))
    for b in balance_records:
        print(
            f"{b['match_index']:>5}{b['date']:>12}{b['winner_size']:>9}{b['loser_size']:>10}"
            f"{b['winner_actual_elo']:>12}{b['loser_actual_elo']:>12}{b['winner_win_prob']*100:>12.1f}%"
        )
    '''

    avg_favorite_win_prob = sum(b["favorite_win_prob"] for b in balance_records) / len(balance_records)
    avg_imbalance = sum(b["imbalance"] for b in balance_records) / len(balance_records)

    print("\n--- Average Match Balancing (measured on hidden actual ratings) ---")
    print(f"Average favorite's true win chance: {avg_favorite_win_prob * 100:.1f}%  "
          f"(50% = perfectly balanced, 100% = maximally lopsided)")
    print(f"Average imbalance score:            {avg_imbalance:.3f}  (0 = coin flip, 1 = total mismatch)")

    return rows


if __name__ == "__main__":
    cfg = SimConfig()
    pool, balance_records = run_simulation(cfg)
    build_report(cfg, pool, balance_records)