import random

import numpy as np
import torch

from bot_cleaner_basic_coverage.algos.ddpg import DDPG
from bot_cleaner_basic_coverage.environments.environment import (
    ContinuousVacuumCleanerEnv,
)


def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create environment
    env = ContinuousVacuumCleanerEnv(size=5.0, max_steps=2000)
    agent = DDPG(env)

    # Training hyperparameters
    max_episodes = 1000
    update_interval = 500
    replay_buffer_size = 10000
    batch_size = 64
    best_coverage = 0

    # Curriculum learning size increments
    size_increments = [(200, 7.0), (400, 8.0), (600, 9.0), (800, 10.0)]

    # Replay buffer
    replay_buffer = []

    # Rendering flag
    render_enabled = True

    try:
        for episode in range(max_episodes):
            # Curriculum learning: increase environment size
            current_size = 5.0
            for threshold, size in size_increments:
                if episode >= threshold:
                    current_size = size

            # Update environment parameters
            env.size = current_size
            env.cell_size = env.size / env.resolution
            env.max_steps = int(2000 * (current_size / 5.0))

            # Reset environment
            obs = env.reset()
            done = False
            total_reward = 0
            episode_steps = 0

            while not done:
                # Render environment if enabled
                if render_enabled:
                    env.render()

                # Preprocess observations
                coverage = obs["coverage"]
                position = obs["position"]

                # Select action
                action = agent.act(coverage, position)

                # Take action
                next_obs, reward, done, info = env.step(action)

                # Preprocess next observations
                next_coverage = next_obs["coverage"]
                next_position = next_obs["position"]

                # Store experience in replay buffer
                experience = {
                    "coverage": coverage,
                    "position": position,
                    "action": action,
                    "reward": reward,
                    "next_coverage": next_coverage,
                    "next_position": next_position,
                    "done": done,
                }
                replay_buffer.append(experience)

                # Remove old experiences if buffer is full
                if len(replay_buffer) > replay_buffer_size:
                    replay_buffer.pop(0)

                # Update agent from replay buffer
                if (
                    len(replay_buffer) >= batch_size
                    and episode_steps % update_interval == 0
                ):
                    batch = random.sample(replay_buffer, batch_size)
                    agent.update(batch)

                # Update tracking variables
                total_reward += reward
                obs = next_obs
                episode_steps += 1

            # Track and save best model
            current_cov = info["coverage_percentage"]
            if current_cov > best_coverage:
                best_coverage = current_cov
                torch.save(
                    agent.actor.state_dict(),
                    f"bot_cleaner_basic_coverage/models/ddpg/best_model_{best_coverage:.2f}.pth",
                )

            # Print episode summary
            print(
                f"Ep {episode} | Coverage: {current_cov:.2%} | "
                f"Size: {current_size:.1f} | Reward: {total_reward:.1f}"
            )

    except KeyboardInterrupt:
        print("Training stopped by user")

    # Save final model and close environment
    torch.save(
        agent.actor.state_dict(),
        "bot_cleaner_basic_coverage/models/ddpg/final_model.pth",
    )
    env.close()


if __name__ == "__main__":
    main()
