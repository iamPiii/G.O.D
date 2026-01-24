"""
Reward functions for environment rollouts.
"""


def environment_reward_fn(
    prompts,
    completions,
    completion_ids,
    episode_rewards=None,
    is_action_valid=None,
    invalid_action_penalty_coef=0.0,
    **kwargs,
):
    """
    Return episode rewards directly (computed during rollout).

    For per-turn samples, all turns in the same episode share the same reward.
    """
    if episode_rewards is None:
        episode_rewards = kwargs.get("env_rewards")

    if episode_rewards is None:
        raise ValueError("episode_rewards must be provided for environment tasks")

    if len(episode_rewards) != len(completions):
        raise ValueError(
            f"episode_rewards length ({len(episode_rewards)}) does not match completions length ({len(completions)})."
        )

    rewards = [float(r) for r in episode_rewards]

    if is_action_valid is None:
        is_action_valid = kwargs.get("is_action_valid")
    if isinstance(invalid_action_penalty_coef, list):
        invalid_action_penalty_coef = invalid_action_penalty_coef[0] if invalid_action_penalty_coef else 0.0
    if invalid_action_penalty_coef in (None, ""):
        invalid_action_penalty_coef = 0.0

    if is_action_valid is not None and float(invalid_action_penalty_coef) > 0.0:
        if len(is_action_valid) != len(rewards):
            raise ValueError(
                f"is_action_valid length ({len(is_action_valid)}) does not match rewards length ({len(rewards)})."
            )
        penalties = []
        for flags in is_action_valid:
            if isinstance(flags, (list, tuple)):
                invalid_count = sum(1 for v in flags if not v)
                penalties.append(float(invalid_action_penalty_coef) * invalid_count)
            else:
                penalties.append(float(invalid_action_penalty_coef) * (0.0 if flags else 1.0))
        rewards = [r - p for r, p in zip(rewards, penalties)]

    return rewards