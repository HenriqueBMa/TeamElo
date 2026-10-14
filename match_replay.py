"""
match_replay.py
================
Step-by-step console replay of a simulate.py run.

Runs the exact same simulation loop as simulate.py (same function calls, same
order of random draws, so results are byte-identical for a given seed) while
recording a full per-match trace: who showed up (old vs. new), how the teams
were split by known Karma, the system's predicted win odds vs. the hidden
"actual" odds, who really won, and every player's Karma/willingness before
and after.

Import it and call things directly:

    from match_replay import build_trace, print_match
    import simulate as sim

    cfg = sim.SimConfig(NUM_MATCHES=10)     # change any parameter you like
    trace, pool, balance = build_trace(cfg)
    for m in trace:
        print_match(m, reveal=True)

...or just run it from the command line -- see `--help` for every flag:

    python match_replay.py                       # print all matches
    python match_replay.py --matches 10           # shorter run
    python match_replay.py --seed 7               # different seed
    python match_replay.py --start 10 --end 15    # only matches 10-15
    python match_replay.py --player SimPlayer3    # only matches SimPlayer3 was in
    python match_replay.py --hide-hidden          # don't reveal hidden ratings
    python match_replay.py --interactive          # pause for Enter between matches
    python match_replay.py --no-color             # plain text, no ANSI colors
"""
import argparse
import sys
from dataclasses import fields
from datetime import timedelta

import main
import simulate as sim


# ============================================================================
# Building the trace
# ============================================================================
def build_trace(cfg: "sim.SimConfig"):
    """
    Re-runs simulate.py's own run_simulation loop (calling its exact
    functions, in the exact same order, with no extra random draws inserted
    anywhere) and returns a rich per-match trace list alongside the usual
    (pool, balance_records) simulate.py normally returns.

    This relies on simulate.py's pick_roster only ever committing a
    brand-new player to `pool` once they're actually part of the final
    roster (fixed at the source in simulate.py) -- so a player's number
    always matches the match they actually debut in.
    """
    import random

    random.seed(cfg.RANDOM_SEED)

    pool = {}
    new_player_counter = 0
    rows = []
    balance_records = []
    trace = []

    for i in range(cfg.NUM_MATCHES):
        pool_before_names = set(pool.keys())

        roster, new_player_counter = sim.pick_roster(cfg, pool, new_player_counter)
        old_names = [p for p in roster if p in pool_before_names]
        new_names = [p for p in roster if p not in pool_before_names]

        pre_state = {
            p: {
                "known_karma": pool[p].known_karma,
                "win_loss_dif": pool[p].win_loss_dif,
                "matches_played": pool[p].matches_played,
                "willingness": pool[p].willingness,
                "actual_rating": pool[p].actual_rating,
            }
            for p in roster
        }

        def balancing_rating(p, _pool=pool):
            sp = _pool[p]
            return sp.known_karma + cfg.WIN_LOSS_DIF_KARMA_WEIGHT * sp.win_loss_dif

        team_a, team_b = sim.balanced_split(roster, balancing_rating)

        team_a_known = sum(pool[p].known_karma for p in team_a)
        team_b_known = sum(pool[p].known_karma for p in team_b)
        predicted_a_wins = main.expected_score(team_a_known, team_b_known)

        winners, losers, p_a_wins, actual_a, actual_b = sim.decide_winner(team_a, team_b, pool)
        sim.apply_match_result(cfg, winners, losers, pool)
        sim.update_willingness(cfg, pool, roster)

        match_date = cfg.START_DATE + timedelta(days=i * cfg.DAYS_BETWEEN_MATCHES)
        rows.append({
            "date": match_date.isoformat(),
            "teams": "|".join([",".join(winners), ",".join(losers)]),
            "result": 1,
        })

        winner_label = "A" if winners is team_a else "B"
        loser_label = "B" if winner_label == "A" else "A"
        winner_actual_elo = actual_a if winner_label == "A" else actual_b
        loser_actual_elo = actual_b if winner_label == "A" else actual_a
        winner_win_prob = p_a_wins if winner_label == "A" else 1 - p_a_wins

        favorite_win_prob = max(p_a_wins, 1 - p_a_wins)
        balance_records.append({
            "match_index": i,
            "date": match_date.isoformat(),
            "num_players": len(roster),
            "winner_size": len(winners),
            "loser_size": len(losers),
            "winner_actual_elo": round(winner_actual_elo, 1),
            "loser_actual_elo": round(loser_actual_elo, 1),
            "winner_win_prob": round(winner_win_prob, 4),
            "favorite_win_prob": favorite_win_prob,
            "imbalance": 2 * (favorite_win_prob - 0.5),
        })

        def player_rows(team):
            out = []
            for p in team:
                sp = pool[p]
                pre = pre_state[p]
                out.append({
                    "name": p,
                    "is_new": p in new_names,
                    "known_karma_pre": round(pre["known_karma"], 1),
                    "known_karma_post": round(sp.known_karma, 1),
                    "karma_delta": round(sp.known_karma - pre["known_karma"], 1),
                    "win_loss_dif_pre": pre["win_loss_dif"],
                    "win_loss_dif_post": sp.win_loss_dif,
                    "matches_played_pre": pre["matches_played"],
                    "actual_rating": round(pre["actual_rating"], 1),
                    "willingness_pre": round(pre["willingness"], 5),
                    "willingness_post": round(sp.willingness, 5),
                })
            return out

        trace.append({
            "match_index": i,
            "date": match_date.isoformat(),
            "num_old": len(old_names),
            "num_new": len(new_names),
            "roster_size": len(roster),
            "team_a": player_rows(team_a),
            "team_b": player_rows(team_b),
            "team_a_known_karma": round(team_a_known, 1),
            "team_b_known_karma": round(team_b_known, 1),
            "team_a_actual_elo": round(actual_a, 1),
            "team_b_actual_elo": round(actual_b, 1),
            "predicted_a_win_prob": round(predicted_a_wins, 4),
            "actual_a_win_prob": round(p_a_wins, 4),
            "winner_label": winner_label,
            "loser_label": loser_label,
        })

    with open(cfg.MATCHES_CSV, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["date", "teams", "result"])
        writer.writeheader()
        writer.writerows(rows)

    return trace, pool, balance_records


# ============================================================================
# Printing a single match
# ============================================================================
class C:
    """ANSI color codes -- swapped for empty strings when --no-color is set."""
    reset = bold = dim = known = hidden = win = loss = ""

    @classmethod
    def enable(cls):
        cls.reset = "\033[0m"
        cls.bold = "\033[1m"
        cls.dim = "\033[2m"
        cls.known = "\033[38;5;179m"   # amber
        cls.hidden = "\033[38;5;141m"  # violet
        cls.win = "\033[38;5;114m"     # green
        cls.loss = "\033[38;5;167m"    # red


def _fmt_delta(d):
    sign = "+" if d >= 0 else ""
    color = C.win if d >= 0 else C.loss
    return f"{color}{sign}{d:.0f}{C.reset}"


def _team_lines(team, label, won, reveal):
    tag = f"{C.win}WON{C.reset}" if won else f"{C.loss}lost{C.reset}"
    out = [f"  Team {label}  [{tag}]"]
    for p in team:
        new_tag = f" {C.dim}(new){C.reset}" if p["is_new"] else ""
        hidden_str = f"  actual={C.hidden}{p['actual_rating']:>6}{C.reset}" if reveal else ""
        out.append(
            f"    {p['name']:<16}{new_tag:<8}"
            f"karma {C.known}{p['known_karma_pre']:>7.1f}{C.reset} -> {p['known_karma_post']:>7.1f}"
            f"  ({_fmt_delta(p['karma_delta'])})"
            f"{hidden_str}"
        )
    return out


def print_match(m, reveal=True, file=sys.stdout):
    """Pretty-print one match's trace dict (as produced by build_trace)."""
    header = f"{C.bold}Match {m['match_index'] + 1}  —  {m['date']}{C.reset}"
    print(header, file=file)
    print(f"  {m['roster_size']} players ({m['num_old']} returning, {m['num_new']} new)", file=file)

    a_won = m["winner_label"] == "A"
    for line in _team_lines(m["team_a"], "A", a_won, reveal):
        print(line, file=file)
    for line in _team_lines(m["team_b"], "B", not a_won, reveal):
        print(line, file=file)

    pred_pct = round(m["predicted_a_win_prob"] * 100)
    print(f"  predicted (known Karma): Team A {pred_pct}% / Team B {100 - pred_pct}%", file=file)
    if reveal:
        act_pct = round(m["actual_a_win_prob"] * 100)
        print(
            f"  actual (hidden truth):   Team A {act_pct}% / Team B {100 - act_pct}%"
            f"  -> Team {m['winner_label']} really did win",
            file=file,
        )
    print(file=file)


def match_involves(m, player_name):
    names = {p["name"] for p in m["team_a"]} | {p["name"] for p in m["team_b"]}
    return player_name in names


# ============================================================================
# CLI
# ============================================================================
def _cfg_from_args(args) -> "sim.SimConfig":
    overrides = {}
    if args.matches is not None:
        overrides["NUM_MATCHES"] = args.matches
    if args.seed is not None:
        overrides["RANDOM_SEED"] = args.seed
    # Any other SimConfig field can be overridden the same way -- add a flag
    # below and pass it through here, or just edit `overrides` directly.
    return sim.SimConfig(**overrides)


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--matches", type=int, help="override SimConfig.NUM_MATCHES")
    parser.add_argument("--seed", type=int, help="override SimConfig.RANDOM_SEED")
    parser.add_argument("--start", type=int, default=1, help="first match number to show (1-based)")
    parser.add_argument("--end", type=int, default=None, help="last match number to show (1-based, inclusive)")
    parser.add_argument("--player", type=str, default=None, help="only show matches this player was in")
    parser.add_argument("--hide-hidden", action="store_true", help="don't reveal hidden actual ratings")
    parser.add_argument("--interactive", action="store_true", help="pause for Enter between matches")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = parser.parse_args()

    if not args.no_color and sys.stdout.isatty():
        C.enable()

    cfg = _cfg_from_args(args)
    trace, pool, balance_records = build_trace(cfg)

    start = max(1, args.start)
    end = args.end or len(trace)
    selected = trace[start - 1:end]
    if args.player:
        selected = [m for m in selected if match_involves(m, args.player)]

    print(f"{C.bold}{len(selected)} match(es) selected out of {len(trace)} simulated{C.reset}\n")

    for m in selected:
        print_match(m, reveal=not args.hide_hidden)
        if args.interactive:
            try:
                input(f"{C.dim}-- Enter for next match, Ctrl+C to quit --{C.reset}")
            except KeyboardInterrupt:
                print()
                break


if __name__ == "__main__":
    main_cli()