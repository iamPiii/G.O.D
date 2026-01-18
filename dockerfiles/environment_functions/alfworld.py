"""
This is an example of a working Rollout Function implementation for the Alfworld Environment Task.
The design of a rollout function has a huge impact on the quality of the trained model on that task.
For Environment Tasks miners are expected to implement their own rollout function.
You can always expect the environment server url for that task to be available in the env variable 'ENVIRONMENT_SERVER_URL'.
For most (if not all) tasks the environment server can be expected to have a standardized interface with /reset, /step, and /observe endpoints.
While this example is for Alfworld the design should work for all standardized environment tasks.
This implementation stores a full episode rollout and returns an action mask so the trainer can
optimize only on action tokens while still conditioning on observations.
Read more about rollout functions here: https://huggingface.co/docs/trl/main/en/openenv
"""

# NOTE: Keeping the original TRL helper for rollouts; server mode is configured via GRPOConfig.
# def _generate_rollout_completions(prompts: list, trainer, *, as_chat: bool | None = None) -> list[dict[str, list]]:
#     from trl.experimental.openenv import generate_rollout_completions as colocate_generate
#     return colocate_generate(trainer, prompts=prompts, as_chat=as_chat)


def alfworld_rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json

    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
    MAX_PROMPT_LEN = 24576      # Max prompt tokens before ending episode early

    # --- 1. Static Initialization (Once per Rank) ---
    # We check if the function has already established a connection for this worker
    if not getattr(alfworld_rollout_first_prompt_and_completion, "initialized", False):
        # Get local rank
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Get env server for that local rank
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]

        # Determine endpoint
        if not server_list:
            # Fallback (though likely fatal for the task)
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]

        # Store endpoint on the function to avoid re-parsing
        alfworld_rollout_first_prompt_and_completion.base_url = base_url

        # Create environment (POST /create) - ONLY ONCE
        try:
            print(f"Initializing AlfWorld environment on rank {rank} at {base_url}...")
            create_res = requests.post(f"{base_url}/create", timeout=300)
            create_res.raise_for_status()
            # Store env_id on the function
            alfworld_rollout_first_prompt_and_completion.env_id = create_res.json()["id"]
            alfworld_rollout_first_prompt_and_completion.initialized = True
            print(f"Environment initialized. ID: {alfworld_rollout_first_prompt_and_completion.env_id}")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    # Retrieve static variables
    env_id = alfworld_rollout_first_prompt_and_completion.env_id
    env_endpoint = alfworld_rollout_first_prompt_and_completion.base_url

    # --- 2. Rollout Setup ---
    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []
    all_episode_action_masks: list[list[int]] = []

    tokenizer = trainer.processing_class
    DATA_LEN = 2500
    TIMEOUT = 2400

    # Hardcoded System Prompt (ReAct)
    conversation_start = [
        {
            "from": "human",
            "value": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
        },
        {
            "from": "gpt",
            "value": "OK. I'll follow your instructions and try my best to solve the task.",
        }
    ]

    # --- 3. Batch Loop ---
    # We use a random game_id for the batch, or you could sample per item if preferred
    game_id = random.randint(0, DATA_LEN - 1)

    for i, prompt in enumerate(prompts):
        # Track episode data: collect text and logprobs during rollout, tokenize once at end
        turn_completions: list[dict] = []  # List of {text, logprobs} for each assistant turn
        invalid_count = 0
        done = False
        solved = False
        turn_number = 0
        episode_prompt_ids: list[int] = []

        # --- Reset Environment (POST /reset) ---
        # Reuse existing env_id, just change the game
        payload = {"id": env_id, "game": game_id, "world_type": "Text"}

        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()

            # Construct Initial Observation
            current_observation = reset_data["observation"]
            current_available_actions = reset_data["available_actions"]
            formatted_observation = f"{current_observation}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue

        # --- Build Conversation History ---
        messages = []
        for message in conversation_start:
            if message["from"] == "human":
                messages.append({"role": "user", "content": message["value"]})
            elif message["from"] == "gpt":
                messages.append({"role": "assistant", "content": message["value"]})

        messages.append({"role": "user", "content": formatted_observation})

        # --- Interaction Loop ---
        while not done and (turn_number < max_turns):
            # Generate Rollout Completion
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Check if prompt exceeds max length - end episode early to prevent context overflow
            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}, ending episode early")
                done = True
                break

            # Save prompt_ids only from first turn (initial prompt)
            if turn_number == 0:
                episode_prompt_ids = prompt_ids

            # Store completion text and logprobs for this turn (will tokenize at end)
            turn_completions.append({
                "text": completion_text,
                "logprobs": logprobs,
                "completion_ids": completion_ids,
            })

            messages.append({"role": "assistant", "content": completion_text})

            # --- Parse Action ---
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            # Parse ReAct format
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            # --- Step Environment (POST /step) ---
            step_reward = 0.0
            step_done = False
            step_state = ""

            try:
                step_payload = {"id": env_id, "action": action_to_send}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()

                # Extract response data
                step_state = step_data["observation"]
                step_reward = step_data["reward"]
                step_done = step_data["done"]
                current_available_actions = step_data["available_actions"]

                # Format next observation
                formatted_observation = f"{step_state}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"

            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = "Invalid Action.\n\n" + formatted_observation
                step_reward = 0.0
                step_done = False

            # Update Loop State
            if step_done and step_reward > 0:
                solved = True

            if "Nothing happens" in step_state:
                invalid_count += 1

            done = step_done

            if not done:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1

        # --- Post-episode: Build completion sequence with action mask using text-based alignment ---
        # This avoids BPE mismatch by tokenizing the full conversation once at the end
        # and using text positions to determine which tokens are assistant responses

        episode_completion_ids: list[int] = []
        episode_logprobs: list[float] = []
        episode_action_mask: list[int] = []

        if turn_completions:
            # Get the full conversation text using the chat template (without generation prompt)
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Get the initial prompt text (conversation_start + first observation)
            initial_messages = messages[:3]  # system user, system assistant, first user observation
            prompt_text = tokenizer.apply_chat_template(initial_messages, tokenize=False, add_generation_prompt=True)

            # The completion text is everything after the prompt
            completion_text_full = full_text[len(prompt_text):]

            # Tokenize the completion portion only
            if completion_text_full:
                completion_token_ids = tokenizer.encode(completion_text_full, add_special_tokens=False)

                # Build action mask by finding assistant response boundaries in the text
                # We track character positions of each assistant turn's content
                assistant_ranges: list[tuple[int, int]] = []  # (start_char, end_char) relative to completion_text_full

                current_pos = 0
                for turn_idx, turn_data in enumerate(turn_completions):
                    turn_text = turn_data["text"]
                    # Find this turn's text in the completion
                    turn_start = completion_text_full.find(turn_text, current_pos)
                    if turn_start != -1:
                        turn_end = turn_start + len(turn_text)
                        assistant_ranges.append((turn_start, turn_end))
                        current_pos = turn_end

                # Use offset mapping to map character positions to token positions
                # Tokenize with return_offsets_mapping to get character spans for each token
                encoding = tokenizer(
                    completion_text_full,
                    add_special_tokens=False,
                    return_offsets_mapping=True
                )
                offset_mapping = encoding.get("offset_mapping", [])

                # Build action mask: 1 for tokens within assistant ranges, 0 otherwise
                action_mask = []
                for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                    is_action = False
                    for range_start, range_end in assistant_ranges:
                        # Token is part of action if it overlaps with an assistant range
                        if char_start < range_end and char_end > range_start:
                            is_action = True
                            break
                    action_mask.append(1 if is_action else 0)

                episode_completion_ids = completion_token_ids
                episode_action_mask = action_mask

                # For logprobs: we have per-turn logprobs, need to map them to tokens
                # Since exact token alignment is tricky, we'll use a simplified approach:
                # Assign logprobs based on which turn each token belongs to
                episode_logprobs = [0.0] * len(completion_token_ids)

                # Map tokens to turns and assign logprobs
                turn_token_starts: list[int] = []
                for turn_idx, turn_data in enumerate(turn_completions):
                    if turn_idx < len(assistant_ranges):
                        range_start, _ = assistant_ranges[turn_idx]
                        # Find first token that starts at or after range_start
                        for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
                            if char_start >= range_start:
                                turn_token_starts.append(tok_idx)
                                break
                        else:
                            turn_token_starts.append(len(offset_mapping))

                # Assign logprobs from each turn
                for turn_idx, turn_data in enumerate(turn_completions):
                    turn_logprobs = turn_data["logprobs"]
                    if turn_idx < len(turn_token_starts):
                        start_tok = turn_token_starts[turn_idx]
                        end_tok = turn_token_starts[turn_idx + 1] if turn_idx + 1 < len(turn_token_starts) else len(episode_logprobs)

                        # Assign logprobs to tokens in this turn
                        for i, tok_idx in enumerate(range(start_tok, end_tok)):
                            if tok_idx < len(episode_logprobs) and i < len(turn_logprobs):
                                episode_logprobs[tok_idx] = turn_logprobs[i]

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids)}), truncating")
            episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
            episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
            episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

        train_reward = (10.0 if solved else 0.0) - 0.1 * float(invalid_count)
        all_episode_prompt_ids.append(episode_prompt_ids)
        all_episode_completion_ids.append(episode_completion_ids)
        all_episode_logprobs.append(episode_logprobs)
        all_episode_rewards.append(train_reward)
        all_episode_action_masks.append(episode_action_mask)

    return {
        "prompt_ids": all_episode_prompt_ids,
        "completion_ids": all_episode_completion_ids,
        "logprobs": all_episode_logprobs,
        "env_rewards": all_episode_rewards,
        "action_mask": all_episode_action_masks
    }

def alfworld_rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)