#!/usr/bin/env python3
"""
Script to sample generations from a trained alfworld model to check for collapse.
This will run a few episodes and print out the model's generations.
"""

import os
import sys
import requests
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def sample_alfworld_episode(model, tokenizer, env_endpoint, game_id, max_turns=10):
    """Run a single alfworld episode and return the generations."""

    # Create environment
    try:
        create_res = requests.post(f"{env_endpoint}/create", timeout=300)
        create_res.raise_for_status()
        env_id = create_res.json()["id"]
        print(f"Environment initialized. ID: {env_id}")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return []

    # Reset environment
    try:
        payload = {"id": env_id, "game": game_id, "world_type": "Text"}
        reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=2400)
        reset_res.raise_for_status()
        reset_data = reset_res.json()

        current_observation = reset_data["observation"]
        current_available_actions = reset_data["available_actions"]
        formatted_observation = f"{current_observation}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        return []

    # System prompt
    conversation_start = [
        {
            "role": "user",
            "content": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
        },
        {
            "role": "assistant",
            "content": "OK. I'll follow your instructions and try my best to solve the task.",
        }
    ]

    messages = conversation_start.copy()
    messages.append({"role": "user", "content": formatted_observation})

    generations = []
    done = False
    turn = 0

    print(f"\n{'='*80}")
    print(f"EPISODE - Game {game_id}")
    print(f"{'='*80}")
    print(f"Initial Observation: {current_observation[:200]}...")
    print(f"{'='*80}\n")

    while not done and turn < max_turns:
        # Format conversation for model
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # Decode only the new tokens
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion_text = full_text[len(prompt):].strip()

        print(f"Turn {turn + 1}:")
        print(f"  Generation: {completion_text}")

        generations.append({
            "turn": turn,
            "generation": completion_text,
            "observation": current_observation if turn == 0 else step_state
        })

        messages.append({"role": "assistant", "content": completion_text})

        # Parse action
        action_to_send = completion_text
        if action_to_send.endswith("</s>"):
            action_to_send = action_to_send[:-5]

        if "Action:" in action_to_send:
            action_to_send = action_to_send.split("Action:")[-1].strip()

        print(f"  Action sent: {action_to_send}")

        # Step environment
        try:
            step_payload = {"id": env_id, "action": action_to_send}
            step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=2400)
            step_res.raise_for_status()
            step_data = step_res.json()

            step_state = step_data["observation"]
            step_reward = step_data["reward"]
            step_done = step_data["done"]
            current_available_actions = step_data["available_actions"]

            formatted_observation = f"{step_state}\nAVAILABLE ACTIONS: {','.join(current_available_actions)}"

            print(f"  Reward: {step_reward}, Done: {step_done}")
            print(f"  Observation: {step_state[:100]}...")

        except Exception as e:
            print(f"  Step failed: {e}")
            formatted_observation = "Invalid Action.\n\n" + formatted_observation
            step_reward = 0.0
            step_done = False
            step_state = "Invalid Action"

        done = step_done

        if not done:
            messages.append({"role": "user", "content": formatted_observation})

        turn += 1
        print()

    print(f"{'='*80}")
    print(f"Episode finished after {turn} turns")
    print(f"{'='*80}\n")

    return generations


def check_for_collapse(generations_list):
    """Analyze generations for signs of collapse."""
    print(f"\n{'='*80}")
    print("COLLAPSE ANALYSIS")
    print(f"{'='*80}\n")

    all_texts = []
    for episode_gens in generations_list:
        for gen in episode_gens:
            all_texts.append(gen["generation"])

    if not all_texts:
        print("No generations to analyze")
        return

    # Check for repetition
    unique_generations = set(all_texts)
    print(f"Total generations: {len(all_texts)}")
    print(f"Unique generations: {len(unique_generations)}")
    print(f"Repetition rate: {1 - len(unique_generations)/len(all_texts):.2%}")

    # Check average length
    avg_len = sum(len(t) for t in all_texts) / len(all_texts)
    print(f"Average generation length: {avg_len:.1f} characters")

    # Check for very short generations (potential collapse)
    very_short = sum(1 for t in all_texts if len(t) < 10)
    print(f"Very short generations (<10 chars): {very_short}/{len(all_texts)} ({very_short/len(all_texts):.1%})")

    # Sample some unique generations
    print(f"\nSample unique generations:")
    for i, text in enumerate(list(unique_generations)[:10], 1):
        print(f"{i}. {text[:150]}{'...' if len(text) > 150 else ''}")


def main():
    # Configuration
    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    base_model = "/cache/models/Qwen--Qwen2.5-3B-Instruct"
    adapter_path = "/outputs/1/environment_test"
    env_endpoint = os.environ.get("ENV_SERVER_URL", "http://localhost:8000")
    num_episodes = 3
    max_turns_per_episode = 5

    print("Loading model...")
    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # Load adapter
    model = PeftModel.from_pretrained(base_model_obj, adapter_path)
    model.eval()

    print("Model loaded successfully!\n")

    # Sample episodes
    all_generations = []
    for i in range(num_episodes):
        game_id = random.randint(0, 2499)
        print(f"\n{'#'*80}")
        print(f"SAMPLING EPISODE {i+1}/{num_episodes}")
        print(f"{'#'*80}")

        generations = sample_alfworld_episode(
            model, tokenizer, env_endpoint, game_id, max_turns=max_turns_per_episode
        )
        all_generations.append(generations)

    # Analyze for collapse
    check_for_collapse(all_generations)


if __name__ == "__main__":
    main()
