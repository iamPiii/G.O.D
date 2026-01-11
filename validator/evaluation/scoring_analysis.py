#!/usr/bin/env python3
"""
Analysis script for tournament scoring system.
Creates example tournament data and shows how scores are calculated and converted to weights.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentRoundResult
from core.models.tournament_models import TournamentTaskScore
from core.models.tournament_models import TournamentType
from validator.evaluation.tournament_scoring import calculate_tournament_type_scores_from_data
from validator.evaluation.tournament_scoring import tournament_scores_to_weights


def create_example_tournament_data(
    tournament_type: TournamentType, hotkeys: list[str], winner_hotkey: str | None = None
) -> TournamentResultsWithWinners:
    """
    Create example tournament data with losses for each hotkey.
    
    For TEXT/IMAGE: Winners get points based on round number * type_weight
    For ENVIRONMENT: Points based on rank within participants
    """
    # Create 3 rounds: round 1 (group), round 2 (knockout), round 3 (final)
    rounds = []
    
    # Round 1: Group stage - multiple winners advance
    round_1_tasks = []
    # Simulate 3 tasks in round 1, each with different winners
    for task_idx in range(3):
        # Winners for this task (top performers)
        task_winners = hotkeys[task_idx * 2 : (task_idx + 1) * 2]  # 2 winners per task
        task = TournamentTaskScore(
            task_id=f"task_1_{task_idx}",
            group_id=f"group_{task_idx}",
            pair_id=None,
            winner=task_winners[0] if task_winners else None,
            participant_scores=[
                {"hotkey": hk, "quality_score": 0.5 + (i * 0.1)} for i, hk in enumerate(hotkeys)
            ],
        )
        round_1_tasks.append(task)
    
    rounds.append(
        TournamentRoundResult(
            round_id="round_1",
            round_number=1,
            round_type="group",
            is_final_round=False,
            tasks=round_1_tasks,
        )
    )
    
    # Round 2: Knockout stage
    round_2_tasks = []
    # Simulate 2 tasks in round 2
    for task_idx in range(2):
        # Winners advance from round 1
        task_winner = hotkeys[task_idx * 2] if task_idx * 2 < len(hotkeys) else hotkeys[0]
        task = TournamentTaskScore(
            task_id=f"task_2_{task_idx}",
            group_id=None,
            pair_id=f"pair_{task_idx}",
            winner=task_winner,
            participant_scores=[],
        )
        round_2_tasks.append(task)
    
    rounds.append(
        TournamentRoundResult(
            round_id="round_2",
            round_number=2,
            round_type="knockout",
            is_final_round=False,
            tasks=round_2_tasks,
        )
    )
    
    # Round 3: Final round
    final_winner = winner_hotkey if winner_hotkey else hotkeys[0]
    round_3_tasks = []
    # Simulate 3 tasks in final round
    for task_idx in range(3):
        task = TournamentTaskScore(
            task_id=f"task_3_{task_idx}",
            group_id=None,
            pair_id=None,
            winner=final_winner,
            participant_scores=[],
        )
        round_3_tasks.append(task)
    
    rounds.append(
        TournamentRoundResult(
            round_id="round_3",
            round_number=3,
            round_type="knockout",
            is_final_round=True,
            tasks=round_3_tasks,
        )
    )
    
    return TournamentResultsWithWinners(
        tournament_id=f"example_{tournament_type.value}_tournament",
        rounds=rounds,
        base_winner_hotkey=None,
        winner_hotkey=final_winner,
    )


def create_environment_tournament_data(
    hotkeys: list[str], winner_hotkey: str | None = None
) -> TournamentResultsWithWinners:
    """
    Create example environment tournament data.
    Environment tournaments use quality_score ranking instead of simple winners.
    """
    rounds = []
    
    # Round 1: Group stage with quality scores
    round_1_tasks = []
    for task_idx in range(2):
        # Create participant scores with varying quality scores
        participant_scores = []
        for i, hk in enumerate(hotkeys):
            # Higher quality_score = better performance (for environment tasks)
            quality_score = 0.8 - (i * 0.05)  # Decreasing scores
            participant_scores.append({"hotkey": hk, "quality_score": quality_score})
        
        task = TournamentTaskScore(
            task_id=f"env_task_1_{task_idx}",
            group_id=f"group_{task_idx}",
            pair_id=None,
            winner=None,  # Environment tasks don't use winner field
            participant_scores=participant_scores,
        )
        round_1_tasks.append(task)
    
    rounds.append(
        TournamentRoundResult(
            round_id="env_round_1",
            round_number=1,
            round_type="group",
            is_final_round=False,
            tasks=round_1_tasks,
        )
    )
    
    # Round 2: Final round
    final_winner = winner_hotkey if winner_hotkey else hotkeys[0]
    round_2_tasks = []
    for task_idx in range(3):
        # Final round participants (top performers from round 1)
        final_participants = hotkeys[:5]  # Top 5 advance
        participant_scores = []
        for i, hk in enumerate(final_participants):
            # Winner has highest score
            if hk == final_winner:
                quality_score = 0.95
            else:
                quality_score = 0.85 - (i * 0.02)
            participant_scores.append({"hotkey": hk, "quality_score": quality_score})
        
        task = TournamentTaskScore(
            task_id=f"env_task_2_{task_idx}",
            group_id=None,
            pair_id=None,
            winner=None,
            participant_scores=participant_scores,
        )
        round_2_tasks.append(task)
    
    rounds.append(
        TournamentRoundResult(
            round_id="env_round_2",
            round_number=2,
            round_type="group",
            is_final_round=True,
            tasks=round_2_tasks,
        )
    )
    
    return TournamentResultsWithWinners(
        tournament_id="example_environment_tournament",
        rounds=rounds,
        base_winner_hotkey=None,
        winner_hotkey=final_winner,
    )


def print_scores(title: str, scores: list):
    """Print tournament scores in a formatted way."""
    print(f"\n{title}")
    print("=" * 80)
    if not scores:
        print("No scores")
        return
    
    # Sort by score descending
    sorted_scores = sorted(scores, key=lambda x: x.score, reverse=True)
    print(f"{'Rank':<6} {'Hotkey':<50} {'Score':<15}")
    print("-" * 80)
    for rank, score in enumerate(sorted_scores, 1):
        print(f"{rank:<6} {score.hotkey:<50} {score.score:<15.6f}")


def print_weights(title: str, weights: dict[str, float]):
    """Print tournament weights in a formatted way."""
    print(f"\n{title}")
    print("=" * 80)
    if not weights:
        print("No weights")
        return
    
    # Sort by weight descending
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Rank':<6} {'Hotkey':<50} {'Weight':<15} {'% of Total':<15}")
    print("-" * 80)
    total_weight = sum(weights.values())
    for rank, (hotkey, weight) in enumerate(sorted_weights, 1):
        percentage = (weight / total_weight * 100) if total_weight > 0 else 0
        print(f"{rank:<6} {hotkey:<50} {weight:<15.8f} {percentage:<15.2f}%")
    print(f"\nTotal Weight: {total_weight:.8f}")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("TOURNAMENT SCORING ANALYSIS")
    print("=" * 80)
    
    # Create 10 example hotkeys
    hotkeys = [f"5Hotkey{i:02d}{'X' * 40}" for i in range(10)]
    
    print(f"\nExample Hotkeys ({len(hotkeys)} total):")
    for i, hk in enumerate(hotkeys, 1):
        print(f"  {i:2d}. {hk}")
    
    # Analyze each tournament type
    tournament_types = [
        (TournamentType.TEXT, "TEXT"),
        (TournamentType.IMAGE, "IMAGE"),
        (TournamentType.ENVIRONMENT, "ENVIRONMENT"),
    ]
    
    for tournament_type, type_name in tournament_types:
        print("\n" + "=" * 80)
        print(f"{type_name} TOURNAMENT ANALYSIS")
        print("=" * 80)
        
        # Create tournament data
        if tournament_type == TournamentType.ENVIRONMENT:
            tournament_data = create_environment_tournament_data(hotkeys, winner_hotkey=hotkeys[0])
        else:
            tournament_data = create_example_tournament_data(tournament_type, hotkeys, winner_hotkey=hotkeys[0])
        
        # Calculate scores
        result = calculate_tournament_type_scores_from_data(tournament_type, tournament_data)
        
        print(f"\nPrevious Winner Hotkey: {result.prev_winner_hotkey}")
        print(f"Previous Winner Won Final: {result.prev_winner_won_final}")
        
        print_scores(f"{type_name} Tournament Scores (from calculate_tournament_type_scores_from_data)", result.scores)
        
        # Convert to weights
        weights = tournament_scores_to_weights(
            result.scores, result.prev_winner_hotkey, result.prev_winner_won_final
        )
        
        print_weights(f"{type_name} Tournament Weights (from tournament_scores_to_weights)", weights)
        
        # Show scoring breakdown
        print(f"\n{type_name} Tournament Scoring Breakdown:")
        print("-" * 80)
        print("Round-by-round scoring:")
        for round_result in tournament_data.rounds:
            round_num = round_result.round_number
            is_final = round_result.is_final_round
            print(f"\n  Round {round_num} ({'FINAL' if is_final else 'REGULAR'}):")
            
            if tournament_type == TournamentType.ENVIRONMENT:
                # Show participant scores for environment
                for task in round_result.tasks:
                    print(f"    Task {task.task_id}:")
                    sorted_participants = sorted(
                        task.participant_scores,
                        key=lambda x: x.get("quality_score", 0),
                        reverse=True,
                    )
                    for i, p in enumerate(sorted_participants[:5], 1):  # Show top 5
                        hk = p.get("hotkey", "unknown")
                        qs = p.get("quality_score", 0)
                        print(f"      {i}. {hk[:20]}... - quality_score: {qs:.4f}")
            else:
                # Show winners for text/image
                for task in round_result.tasks:
                    winner = task.winner
                    if winner:
                        print(f"    Task {task.task_id}: Winner = {winner[:20]}...")


if __name__ == "__main__":
    main()

