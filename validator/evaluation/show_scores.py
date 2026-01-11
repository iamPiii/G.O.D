#!/usr/bin/env python3
"""Show the scores list for each tournament type."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.models.tournament_models import TournamentResultsWithWinners, TournamentRoundResult, TournamentTaskScore, TournamentType
from validator.evaluation.tournament_scoring import calculate_tournament_type_scores_from_data

# Create 10 example hotkeys
hotkeys = [f'5Hotkey{i:02d}{"X" * 40}' for i in range(10)]

def create_text_image_tournament(tournament_type, hotkeys, winner_hotkey):
    rounds = []
    round_1_tasks = []
    for task_idx in range(3):
        task_winners = hotkeys[task_idx * 2 : (task_idx + 1) * 2]
        task = TournamentTaskScore(
            task_id=f'task_1_{task_idx}',
            group_id=f'group_{task_idx}',
            pair_id=None,
            winner=task_winners[0] if task_winners else None,
            participant_scores=[{'hotkey': hk, 'quality_score': 0.5 + (i * 0.1)} for i, hk in enumerate(hotkeys)],
        )
        round_1_tasks.append(task)
    rounds.append(TournamentRoundResult(round_id='round_1', round_number=1, round_type='group', is_final_round=False, tasks=round_1_tasks))
    
    round_2_tasks = []
    for task_idx in range(2):
        task_winner = hotkeys[task_idx * 2] if task_idx * 2 < len(hotkeys) else hotkeys[0]
        task = TournamentTaskScore(
            task_id=f'task_2_{task_idx}',
            group_id=None,
            pair_id=f'pair_{task_idx}',
            winner=task_winner,
            participant_scores=[],
        )
        round_2_tasks.append(task)
    rounds.append(TournamentRoundResult(round_id='round_2', round_number=2, round_type='knockout', is_final_round=False, tasks=round_2_tasks))
    
    round_3_tasks = []
    for task_idx in range(3):
        task = TournamentTaskScore(
            task_id=f'task_3_{task_idx}',
            group_id=None,
            pair_id=None,
            winner=winner_hotkey,
            participant_scores=[],
        )
        round_3_tasks.append(task)
    rounds.append(TournamentRoundResult(round_id='round_3', round_number=3, round_type='knockout', is_final_round=True, tasks=round_3_tasks))
    
    return TournamentResultsWithWinners(
        tournament_id=f'example_{tournament_type.value}_tournament',
        rounds=rounds,
        base_winner_hotkey=None,
        winner_hotkey=winner_hotkey,
    )

def create_environment_tournament(hotkeys, winner_hotkey):
    rounds = []
    round_1_tasks = []
    for task_idx in range(2):
        participant_scores = []
        for i, hk in enumerate(hotkeys):
            quality_score = 0.8 - (i * 0.05)
            participant_scores.append({'hotkey': hk, 'quality_score': quality_score})
        task = TournamentTaskScore(
            task_id=f'env_task_1_{task_idx}',
            group_id=f'group_{task_idx}',
            pair_id=None,
            winner=None,
            participant_scores=participant_scores,
        )
        round_1_tasks.append(task)
    rounds.append(TournamentRoundResult(round_id='env_round_1', round_number=1, round_type='group', is_final_round=False, tasks=round_1_tasks))
    
    round_2_tasks = []
    for task_idx in range(3):
        final_participants = hotkeys[:5]
        participant_scores = []
        for i, hk in enumerate(final_participants):
            if hk == winner_hotkey:
                quality_score = 0.95
            else:
                quality_score = 0.85 - (i * 0.02)
            participant_scores.append({'hotkey': hk, 'quality_score': quality_score})
        task = TournamentTaskScore(
            task_id=f'env_task_2_{task_idx}',
            group_id=None,
            pair_id=None,
            winner=None,
            participant_scores=participant_scores,
        )
        round_2_tasks.append(task)
    rounds.append(TournamentRoundResult(round_id='env_round_2', round_number=2, round_type='group', is_final_round=True, tasks=round_2_tasks))
    
    return TournamentResultsWithWinners(
        tournament_id='example_environment_tournament',
        rounds=rounds,
        base_winner_hotkey=None,
        winner_hotkey=winner_hotkey,
    )

print('=' * 100)
print('SCORES LIST FOR EACH TOURNAMENT TYPE')
print('=' * 100)

for tournament_type, type_name in [(TournamentType.TEXT, 'TEXT'), (TournamentType.IMAGE, 'IMAGE'), (TournamentType.ENVIRONMENT, 'ENVIRONMENT')]:
    print(f'\n{"=" * 100}')
    print(f'{type_name} TOURNAMENT SCORES')
    print('=' * 100)
    
    if tournament_type == TournamentType.ENVIRONMENT:
        tournament_data = create_environment_tournament(hotkeys, hotkeys[0])
    else:
        tournament_data = create_text_image_tournament(tournament_type, hotkeys, hotkeys[0])
    
    result = calculate_tournament_type_scores_from_data(tournament_type, tournament_data)
    
    print(f'\nPrevious winner hotkey: {result.prev_winner_hotkey}')
    print(f'Previous winner won final: {result.prev_winner_won_final}')
    print(f'\nNumber of scores: {len(result.scores)}')
    print('\nScores list (sorted by score, descending):')
    print('-' * 100)
    print(f'{"Rank":<6} {"Hotkey":<50} {"Score":<15}')
    print('-' * 100)
    
    sorted_scores = sorted(result.scores, key=lambda x: x.score, reverse=True)
    for rank, score in enumerate(sorted_scores, 1):
        print(f'{rank:<6} {score.hotkey:<50} {score.score:<15.6f}')
    
    if not result.scores:
        print('(No scores - winner was excluded from scoring)')
    
    # Show raw score objects
    print('\nRaw TournamentScore objects:')
    print('-' * 100)
    for score in sorted_scores:
        print(f'  TournamentScore(hotkey="{score.hotkey}", score={score.score})')

