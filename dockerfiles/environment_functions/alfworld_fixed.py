"""
Improved AlfWorld Rollout Function

1. BPE Mismatch Prevention (Priority 1): Token-space operations, no decode/re-encode
2. Constant Context Length (Priority 3): Process only current observation, not full history
3. Robust Action Masking (Priority 2): Structured episode data with guaranteed alignment
4. Invalid Action Penalty (Priority 4): Track and report invalid actions for penalty

- Store episode state as token IDs, not strings
- Rebuild prompt each turn (system + current obs only)
- Derive masks from actual token structure
- Track invalid actions for loss computation
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EpisodeTurn:
    """
    Structured storage for one turn of the episode.
    Guarantees alignment between token IDs and masks at construction time.
    """
    token_ids: List[int]  # Either action tokens or observation tokens
    logprobs: List[float]  # Log probabilities (0.0 for non-generated tokens)
    is_action: bool  # True for action tokens, False for observation tokens
    is_valid: Optional[bool] = None  # For actions: was it valid? None for observations

    def __post_init__(self):
        """Validate alignment at construction"""
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError(
                f"Token IDs ({len(self.token_ids)}) and logprobs ({len(self.logprobs)}) "
                f"length mismatch in EpisodeTurn"
            )

    @property
    def action_mask(self) -> List[int]:
        """Action mask: 1 for action tokens, 0 for observation tokens"""
        return [1 if self.is_action else 0] * len(self.token_ids)

    @property
    def length(self) -> int:
        return len(self.token_ids)


def build_episode_data(initial_prompt_ids: List[int], turns: List[EpisodeTurn]) -> dict:
    """
    Construct episode data with guaranteed alignment.

    Args:
        initial_prompt_ids: The prompt token IDs from turn 0
        turns: List of turns (actions + observations)

    Returns:
        Dict with aligned prompt_ids, completion_ids, action_mask, logprobs, is_action_valid
    """
    completion_ids = []
    action_mask = []
    logprobs = []
    is_action_valid = []

    for turn in turns:
        completion_ids.extend(turn.token_ids)
        action_mask.extend(turn.action_mask)
        logprobs.extend(turn.logprobs)

        # Track validity for actions only
        if turn.is_action and turn.is_valid is not None:
            is_action_valid.append(turn.is_valid)

    # Validate alignment
    assert len(completion_ids) == len(action_mask) == len(logprobs), \
        f"Alignment error: completion_ids={len(completion_ids)}, " \
        f"action_mask={len(action_mask)}, logprobs={len(logprobs)}"

    return {
        "prompt_ids": initial_prompt_ids,
        "completion_ids": completion_ids,
        "action_mask": action_mask,
        "logprobs": logprobs,
        "is_action_valid": is_action_valid,  # List of bools for each action
    }


def extract_action_from_response(response_text: str, eos_token: str = None) -> str:
    """
    Extract action from model response using verl-agent format.

    Expected format:
        <think>reasoning process</think>
        <action>action text</action>

    Args:
        response_text: Raw response from model
        eos_token: Tokenizer EOS token to strip

    Returns:
        Extracted action text, or full response if no tags found
    """
    action = response_text.strip()

    # Remove EOS tokens
    if eos_token and action.endswith(eos_token):
        action = action[: -len(eos_token)].rstrip()
    elif action.endswith("</s>"):
        action = action[:-5].rstrip()

    # Extract action from <action> tags (verl-agent format)
    action_lower = action.lower()
    if "<action>" in action_lower:
        start_tag = "<action>"
        end_tag = "</action>"

        start_idx = action_lower.index(start_tag) + len(start_tag)
        # Look for closing tag
        if end_tag in action_lower[start_idx:]:
            end_idx = action_lower.index(end_tag, start_idx)
            action = action[start_idx:end_idx].strip()
        else:
            # No closing tag, take everything after opening tag
            action = action[start_idx:].strip()

    # Fallback: support old ReAct format for backward compatibility
    elif "Action:" in action:
        action = action.split("Action:")[-1].strip()

    return action


def _init_alfworld_environment(rollout_fn):
    """
    Initialize the AlfWorld environment once per process/rank.

    Args:
        rollout_fn: Function object used to store static attributes

    Returns:
        Tuple of (env_id, base_url)
    """
    import os
    import requests

    if not getattr(rollout_fn, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]

        if not server_list:
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]

        rollout_fn.base_url = base_url

        # Create environment (POST /create) - ONLY ONCE
        try:
            print(f"Initializing AlfWorld environment on rank {rank} at {base_url}...")
            create_res = requests.post(f"{base_url}/create", timeout=300)
            create_res.raise_for_status()
            rollout_fn.env_id = create_res.json()["id"]
            rollout_fn.initialized = True
            print(f"Environment initialized. ID: {rollout_fn.env_id}")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    return rollout_fn.env_id, rollout_fn.base_url


def alfworld_rollout_per_turn(prompts: list[str], trainer, max_turns: int = 30, history_length: int = 2) -> dict[str, list]:
    """
    verl-agent style rollout: each turn is a separate sample.

    Each turn returns:
      - prompt_ids = [System + current observation]
      - completion_ids = [Single action tokens]
      - logprobs = per-token logprobs for the action
      - uid = group id for GRPO baseline
      - traj_uid = episode id (all turns share)
      - episode_rewards = broadcast episode reward to every turn
    """
    from trl.experimental.openenv import generate_rollout_completions
    import random
    import requests
    import uuid

    # --- Constants ---
    MAX_PROMPT_LEN = 24576
    DATA_LEN = 2500
    TIMEOUT = 2400

    # --- Static Initialization (Once per Rank) ---
    env_id, env_endpoint = _init_alfworld_environment(alfworld_rollout_per_turn)

    # --- Rollout Output ---
    all_prompt_ids: list[list[int]] = []
    all_completion_ids: list[list[int]] = []
    all_logprobs: list[list[float]] = []
    all_action_masks: list[list[int]] = []
    all_uids: list[str] = []
    all_traj_uids: list[str] = []
    all_episode_rewards: list[float] = []
    all_episode_turns: list[int] = []
    all_episode_truncated: list[bool] = []

    tokenizer = trainer.processing_class

    # Verl-agent prompt template
    ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

    ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

    # --- Batch Loop ---
    if trainer.model.training:
        num_generations = getattr(trainer, "num_generations", 1)
    else:
        num_generations = getattr(trainer, "num_generations_eval", getattr(trainer, "num_generations", 1))
    num_generations = max(1, int(num_generations))

    if len(prompts) % num_generations != 0:
        print(f"WARNING: Prompts count {len(prompts)} not divisible by num_generations {num_generations}")

    current_group_game_id = None
    current_group_index = None

    for i, prompt in enumerate(prompts):
        group_index = i // num_generations
        if group_index != current_group_index:
            current_group_index = group_index
            current_group_game_id = random.randint(0, DATA_LEN - 1)
        game_id = current_group_game_id

        uid = f"group_{group_index}"
        traj_uid = str(uuid.uuid4())

        # Episode state
        turn_samples = []
        invalid_count = 0
        done = False
        solved = False
        turn_number = 0
        episode_truncated = False
        memory = []  # History tracking: list of {"text_obs": str, "action": str}
        task_description = None

        # --- Reset Environment ---
        payload = {"id": env_id, "game": game_id, "world_type": "Text"}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()

            current_observation = reset_data["observation"]
            current_available_actions = reset_data["available_actions"]
            
            # Extract task description from initial observation (verl-agent style)
            task_marker = "Your task is to: "
            if task_marker in current_observation:
                task_description = current_observation[current_observation.find(task_marker) + len(task_marker):].strip()
            else:
                task_description = "complete the task"
            
            # Format admissible actions (verl-agent style: newline-separated, quoted, exclude 'help')
            reformatted_admissible_actions = "\n ".join(
                f"'{s}'" for s in current_available_actions if s != 'help'
            )
            
            # First turn uses ALFWORLD_TEMPLATE_NO_HIS
            formatted_prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=current_observation,
                admissible_actions=reformatted_admissible_actions,
            )
            pre_text_obs = current_observation  # Track for memory
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue

        # --- Interaction Loop (Per-Turn Samples) ---
        while not done and (turn_number < max_turns):
            current_prompt_messages = [{"role": "user", "content": formatted_prompt}]

            try:
                rollout_outputs = generate_rollout_completions(
                    trainer,
                    prompts=[current_prompt_messages],
                    as_chat=True,
                )[0]
            except Exception as e:
                print(f"Generation failed at turn {turn_number}: {e}")
                episode_truncated = True
                break

            prompt_ids = rollout_outputs.get("prompt_ids") or []
            completion_ids = rollout_outputs.get("completion_ids") or []
            logprobs = rollout_outputs.get("logprobs")
            if logprobs is None:
                logprobs = [0.0] * len(completion_ids)

            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}")
                episode_truncated = True
                break

            completion_text_raw = rollout_outputs.get("text")
            if completion_text_raw is None:
                completion_text_raw = tokenizer.decode(completion_ids, skip_special_tokens=False)

            action_to_send = extract_action_from_response(completion_text_raw, tokenizer.eos_token)

            # --- Step Environment ---
            step_reward = 0.0
            step_done = False
            step_state = ""
            step_info = {}

            try:
                step_payload = {"id": env_id, "action": action_to_send}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()

                step_state = step_data["observation"]
                step_reward = step_data["reward"]
                step_done = step_data["done"]
                current_available_actions = step_data["available_actions"]
                step_info = step_data.get("info", {})

                # Store in memory BEFORE building next prompt (verl-agent stores pre_text_obs)
                memory.append({"text_obs": pre_text_obs, "action": action_to_send})
                pre_text_obs = step_state  # Update for next iteration

                # Format admissible actions (verl-agent style)
                reformatted_admissible_actions = "\n ".join(
                    f"'{s}'" for s in current_available_actions if s != 'help'
                )

                # Build next prompt: use ALFWORLD_TEMPLATE if history_length > 0
                if history_length > 0 and len(memory) > 0:
                    # Fetch recent history (SimpleMemory.fetch() style)
                    recent = memory[-history_length:]
                    valid_len = len(recent)
                    start_idx = len(memory) - valid_len
                    
                    # Format history as: [Observation N: 'obs', Action N: 'act']
                    history_lines = []
                    for j, rec in enumerate(recent):
                        step_num = start_idx + j + 1
                        history_lines.append(
                            f"[Observation {step_num}: '{rec['text_obs']}', Action {step_num}: '{rec['action']}']"
                        )
                    action_history = "\n".join(history_lines)
                    
                    formatted_prompt = ALFWORLD_TEMPLATE.format(
                        task_description=task_description,
                        step_count=len(memory),
                        history_length=valid_len,
                        action_history=action_history,
                        current_step=len(memory) + 1,
                        current_observation=step_state,
                        admissible_actions=reformatted_admissible_actions,
                    )
                else:
                    formatted_prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                        current_observation=step_state,
                        admissible_actions=reformatted_admissible_actions,
                    )
            except Exception as e:
                print(f"Step failed at turn {turn_number}: {e}")
                formatted_prompt = "Invalid Action.\n\n" + formatted_prompt
                step_reward = 0.0
                step_done = False

            # Track invalid actions
            if "Nothing happens" in step_state:
                invalid_count += 1

            turn_samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "completion_ids": completion_ids,
                    "logprobs": logprobs,
                    "action_mask": [1] * len(completion_ids),
                }
            )

            if step_done and step_reward > 0:
                solved = True
            done = step_done
            turn_number += 1

        if not done and turn_number >= max_turns:
            episode_truncated = True

        if not turn_samples:
            continue

        # Calculate episode reward with invalid action penalty
        episode_reward = (10.0 if solved else 0.0) - 0.1 * float(invalid_count)

        # Broadcast episode-level metadata to each turn
        for sample in turn_samples:
            all_prompt_ids.append(sample["prompt_ids"])
            all_completion_ids.append(sample["completion_ids"])
            all_logprobs.append(sample["logprobs"])
            all_action_masks.append(sample["action_mask"])
            all_uids.append(uid)
            all_traj_uids.append(traj_uid)
            all_episode_rewards.append(episode_reward)
            all_episode_turns.append(turn_number)
            all_episode_truncated.append(episode_truncated)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "action_mask": all_action_masks,
        "uid": all_uids,
        "traj_uid": all_traj_uids,
        "episode_rewards": all_episode_rewards,
        "env_rewards": all_episode_rewards,
        "episode_turns": all_episode_turns,
        "episode_truncated": all_episode_truncated,
    }


def alfworld_rollout_reward_func(completions, **kwargs):
    """Reward function for AlfWorld (returns env rewards)"""
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)