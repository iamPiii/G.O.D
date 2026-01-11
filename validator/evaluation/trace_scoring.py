#!/usr/bin/env python3
"""Trace the scoring functions to show inputs and outputs with loss numbers."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.models.tournament_models import TournamentResultsWithWinners, TournamentRoundResult, TournamentTaskScore, TournamentType, TournamentScore
from validator.evaluation.tournament_scoring import calculate_tournament_type_scores_from_data, tournament_scores_to_weights
import validator.core.constants as cts

# Create 10 example hotkeys
hotkeys = [f'5Hotkey{i:02d}{"X" * 40}' for i in range(10)]

def print_section(title):
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)

def print_input_data(tournament_data, tournament_type):
    """Show the input data going into calculate_tournament_type_scores_from_data"""
    print_section(f'INPUT TO calculate_tournament_type_scores_from_data ({tournament_type.value.upper()})')
    
    print(f'\nTournament ID: {tournament_data.tournament_id}')
    print(f'Winner hotkey: {tournament_data.winner_hotkey}')
    print(f'Base winner hotkey: {tournament_data.base_winner_hotkey}')
    print(f'\nNumber of rounds: {len(tournament_data.rounds)}')
    
    for round_result in tournament_data.rounds:
        print(f'\n  Round {round_result.round_number} ({round_result.round_type}, {"FINAL" if round_result.is_final_round else "REGULAR"}):')
        print(f'    Number of tasks: {len(round_result.tasks)}')
        
        for task_idx, task in enumerate(round_result.tasks):
            print(f'\n    Task {task_idx + 1} ({task.task_id}):')
            print(f'      Winner: {task.winner}')
            
            if task.participant_scores:
                print(f'      Participant scores (quality_score / loss):')
                sorted_participants = sorted(
                    task.participant_scores,
                    key=lambda x: x.get('quality_score', 0),
                    reverse=True
                )
                for i, p in enumerate(sorted_participants[:10], 1):  # Show all
                    hk = p.get('hotkey', 'unknown')
                    qs = p.get('quality_score', 0)
                    print(f'        {i:2d}. {hk[:25]}... - quality_score: {qs:.6f}')

def print_output_scores(result, tournament_type):
    """Show the output from calculate_tournament_type_scores_from_data"""
    print_section(f'OUTPUT FROM calculate_tournament_type_scores_from_data ({tournament_type.value.upper()})')
    
    print(f'\nPrevious winner hotkey: {result.prev_winner_hotkey}')
    print(f'Previous winner won final: {result.prev_winner_won_final}')
    print(f'\nNumber of TournamentScore objects: {len(result.scores)}')
    
    if result.scores:
        print('\nTournamentScore objects (sorted by score, descending):')
        sorted_scores = sorted(result.scores, key=lambda x: x.score, reverse=True)
        print(f'  {"Rank":<6} {"Hotkey":<30} {"Score":<15}')
        print('  ' + '-' * 50)
        for rank, score in enumerate(sorted_scores, 1):
            hotkey_short = score.hotkey[:28] + '...' if len(score.hotkey) > 28 else score.hotkey
            print(f'  {rank:<6} {hotkey_short:<30} {score.score:<15.6f}')
        
        print('\nRaw TournamentScore objects:')
        for score in sorted_scores:
            print(f'    TournamentScore(hotkey="{score.hotkey}", score={score.score})')
    else:
        print('\n(No scores - all participants were winners or excluded)')

def print_weights_input(scores, prev_winner_hotkey, prev_winner_won_final):
    """Show the input to tournament_scores_to_weights"""
    print_section('INPUT TO tournament_scores_to_weights')
    
    print(f'\nPrevious winner hotkey: {prev_winner_hotkey}')
    print(f'Previous winner won final: {prev_winner_won_final}')
    print(f'\nNumber of TournamentScore objects: {len(scores)}')
    
    if scores:
        print('\nTournamentScore objects (sorted by score, descending):')
        sorted_scores = sorted(scores, key=lambda x: x.score, reverse=True)
        print(f'  {"Rank":<6} {"Hotkey":<30} {"Score":<15}')
        print('  ' + '-' * 50)
        for rank, score in enumerate(sorted_scores, 1):
            hotkey_short = score.hotkey[:28] + '...' if len(score.hotkey) > 28 else score.hotkey
            print(f'  {rank:<6} {hotkey_short:<30} {score.score:<15.6f}')
        
        print('\nRaw TournamentScore objects:')
        for score in sorted_scores:
            print(f'    TournamentScore(hotkey="{score.hotkey}", score={score.score})')
    else:
        print('\n(Empty scores list)')

def print_weights_output(weights):
    """Show the output from tournament_scores_to_weights"""
    print_section('OUTPUT FROM tournament_scores_to_weights')
    
    if not weights:
        print('\n(Empty weights dict)')
        return
    
    print(f'\nNumber of hotkeys with weights: {len(weights)}')
    print(f'Total weight sum: {sum(weights.values()):.8f}')
    
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    print(f'\n  {"Rank":<6} {"Hotkey":<30} {"Weight":<15} {"% of Total":<15}')
    print('  ' + '-' * 65)
    total_weight = sum(weights.values())
    for rank, (hotkey, weight) in enumerate(sorted_weights, 1):
        hotkey_short = hotkey[:28] + '...' if len(hotkey) > 28 else hotkey
        percentage = (weight / total_weight * 100) if total_weight > 0 else 0
        print(f'  {rank:<6} {hotkey_short:<30} {weight:<15.8f} {percentage:<15.2f}%')
    
    print('\nRaw weights dict:')
    for hotkey, weight in sorted_weights:
        print(f'    "{hotkey}": {weight}')

def create_text_tournament(hotkeys, winner_hotkey):
    """Create a TEXT tournament with detailed loss/score data"""
    rounds = []
    
    # Round 1: Group stage - 3 tasks
    round_1_tasks = []
    for task_idx in range(3):
        # Winners for this task
        task_winner = hotkeys[task_idx * 2] if task_idx * 2 < len(hotkeys) else hotkeys[0]
        # Create participant scores (for reference, though TEXT doesn't use them for scoring)
        participant_scores = []
        for i, hk in enumerate(hotkeys):
            # Simulate test_loss values (lower is better for TEXT)
            test_loss = 0.5 + (i * 0.05)
            participant_scores.append({
                'hotkey': hk,
                'quality_score': test_loss,  # For TEXT, quality_score might be loss
                'test_loss': test_loss
            })
        
        task = TournamentTaskScore(
            task_id=f'text_task_1_{task_idx}',
            group_id=f'group_{task_idx}',
            pair_id=None,
            winner=task_winner,
            participant_scores=participant_scores,
        )
        round_1_tasks.append(task)
    
    rounds.append(TournamentRoundResult(
        round_id='text_round_1',
        round_number=1,
        round_type='group',
        is_final_round=False,
        tasks=round_1_tasks,
    ))
    
    # Round 2: Knockout stage - 2 tasks
    round_2_tasks = []
    for task_idx in range(2):
        task_winner = hotkeys[task_idx * 2] if task_idx * 2 < len(hotkeys) else hotkeys[0]
        task = TournamentTaskScore(
            task_id=f'text_task_2_{task_idx}',
            group_id=None,
            pair_id=f'pair_{task_idx}',
            winner=task_winner,
            participant_scores=[],
        )
        round_2_tasks.append(task)
    
    rounds.append(TournamentRoundResult(
        round_id='text_round_2',
        round_number=2,
        round_type='knockout',
        is_final_round=False,
        tasks=round_2_tasks,
    ))
    
    # Round 3: Final round - 3 tasks, all won by winner_hotkey
    round_3_tasks = []
    for task_idx in range(3):
        task = TournamentTaskScore(
            task_id=f'text_task_3_{task_idx}',
            group_id=None,
            pair_id=None,
            winner=winner_hotkey,
            participant_scores=[],
        )
        round_3_tasks.append(task)
    
    rounds.append(TournamentRoundResult(
        round_id='text_round_3',
        round_number=3,
        round_type='knockout',
        is_final_round=True,
        tasks=round_3_tasks,
    ))
    
    return TournamentResultsWithWinners(
        tournament_id='example_text_tournament',
        rounds=rounds,
        base_winner_hotkey=None,
        winner_hotkey=winner_hotkey,
    )

def create_environment_tournament(hotkeys, winner_hotkey):
    """Create an ENVIRONMENT tournament with detailed quality_score data - single task, no rounds"""
    # Environment tournament: single round, single task
    participant_scores = []
    for i, hk in enumerate(hotkeys):
        # Higher quality_score = better performance (for environment tasks)
        # Winner has highest score
        if hk == winner_hotkey:
            quality_score = 0.95
        else:
            quality_score = 0.85 - (i * 0.05)  # Decreasing scores
        
        participant_scores.append({
            'hotkey': hk,
            'quality_score': quality_score
        })
    
    # Sort by quality_score descending to show ranking
    participant_scores.sort(key=lambda x: x['quality_score'], reverse=True)
    
    task = TournamentTaskScore(
        task_id='env_task_1',
        group_id=None,
        pair_id=None,
        winner=None,  # Environment tasks don't use winner field
        participant_scores=participant_scores,
    )
    
    # Single round with single task
    rounds = [TournamentRoundResult(
        round_id='env_round_1',
        round_number=1,  # Still uses round_number=1 for scoring formula
        round_type='group',
        is_final_round=True,  # This is the final (and only) round
        tasks=[task],
    )]
    
    return TournamentResultsWithWinners(
        tournament_id='example_environment_tournament',
        rounds=rounds,
        base_winner_hotkey=None,
        winner_hotkey=winner_hotkey,
    )

def main():
    print('=' * 100)
    print('DETAILED TRACE OF ENVIRONMENT TOURNAMENT SCORING')
    print('=' * 100)
    
    # Analyze ENVIRONMENT tournament only
    env_tournament = create_environment_tournament(hotkeys, hotkeys[0])
    print_input_data(env_tournament, TournamentType.ENVIRONMENT)
    
    env_result = calculate_tournament_type_scores_from_data(TournamentType.ENVIRONMENT, env_tournament)
    print_output_scores(env_result, TournamentType.ENVIRONMENT)
    
    print_weights_input(env_result.scores, env_result.prev_winner_hotkey, env_result.prev_winner_won_final)
    env_weights = tournament_scores_to_weights(
        env_result.scores, env_result.prev_winner_hotkey, env_result.prev_winner_won_final
    )
    print_weights_output(env_weights)
    
    # Show scoring calculation details
    print_section('SCORING CALCULATION DETAILS')
    print('\nENVIRONMENT Tournament Scoring Formula:')
    print('  Single task, single round (round_number = 1)')
    print('  For each participant (excluding winner):')
    print('    score = round_number × TOURNAMENT_ENVIRONMENT_WEIGHT × (total_participants - rank + 1) / total_participants')
    print(f'  TOURNAMENT_ENVIRONMENT_WEIGHT = {cts.TOURNAMENT_ENVIRONMENT_WEIGHT}')
    print(f'  round_number = 1 (single round)')
    print('\n  Example calculations (10 participants, excluding winner):')
    print('    Rank 1: 1 × 0.10 × (9 - 1 + 1) / 9 = 1 × 0.10 × 1.0 = 0.10')
    print('    Rank 2: 1 × 0.10 × (9 - 2 + 1) / 9 = 1 × 0.10 × 0.8889 = 0.0889')
    print('    Rank 3: 1 × 0.10 × (9 - 3 + 1) / 9 = 1 × 0.10 × 0.7778 = 0.0778')
    print('    Rank 5: 1 × 0.10 × (9 - 5 + 1) / 9 = 1 × 0.10 × 0.5556 = 0.0556')
    print('    Rank 9: 1 × 0.10 × (9 - 9 + 1) / 9 = 1 × 0.10 × 0.1111 = 0.0111')
    
    print('\n\nWeight Conversion Formula (tournament_scores_to_weights):')
    print('  Uses exponential decay: weight = (0.3 ^ (rank - 1)) / sum(all_weights)')
    print(f'  TOURNAMENT_SIMPLE_DECAY_BASE = {cts.TOURNAMENT_SIMPLE_DECAY_BASE}')
    print('  Previous winner gets special treatment:')
    print('    - If won final: placed 1st with infinite score (~72% weight)')
    print('    - If lost but participated: placed 2nd with max_score - 0.1')
    print('    - If won by default (not in scores): placed 1st with infinite score')

if __name__ == '__main__':
    main()

