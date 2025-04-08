import os
import random

import numpy as np
import torch

from bot_cleaner_basic_coverage.algos.ddpg import DDPG, ReplayBuffer
from bot_cleaner_basic_coverage.environments.environment import (
    ContinuousVacuumCleanerEnv,
)


def train_ddpg():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create environment
    env = ContinuousVacuumCleanerEnv(size=10.0, max_steps=2000)

    # Initialize DDPG agent
    agent = DDPG(env)

    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)

    # Training hyperparameters
    max_episodes = 1000
    batch_size = 64
    warmup_steps = 1000
    update_interval = 1
    best_coverage = 0

    # Curriculum learning size increments
    size_increments = [(200, 7.0), (400, 8.0), (600, 9.0), (800, 10.0)]

    # Rendering flag
    render_enabled = True

    # Create directory for saving models if it doesn't exist
    os.makedirs("bot_cleaner_basic_coverage/models/ddpg", exist_ok=True)

    # Training loop
    try:
        # Fill replay buffer with random transitions
        obs = env.reset()
        for _ in range(warmup_steps):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)

            coverage = obs["coverage"].reshape(1, 50, 50)
            position = obs["position"]
            next_coverage = next_obs["coverage"].reshape(1, 50, 50)
            next_position = next_obs["position"]

            replay_buffer.push(
                coverage,
                position,
                action,
                reward,
                next_coverage,
                next_position,
                float(done),
            )

            obs = next_obs if not done else env.reset()

        print(f"Warmup completed, buffer size: {len(replay_buffer)}")

        total_steps = 0
        for episode in range(max_episodes):
            # Curriculum learning: increase environment size
            current_size = 10.0
            for threshold, size in size_increments:
                if episode >= threshold:
                    current_size = size

            # Update environment parameters
            env.size = current_size
            env.cell_size = env.size / env.resolution
            env.max_steps = int(2000 * (current_size / 10.0))

            # Reset environment and noise
            obs = env.reset()
            agent.noise.reset()
            done = False
            total_reward = 0
            episode_steps = 0

            while not done:
                # Render environment if enabled
                if render_enabled:
                    env.render()

                # Preprocess observations
                coverage = obs["coverage"].reshape(1, 50, 50)
                position = obs["position"]

                # Select action
                action = agent.select_action(coverage, position)

                # Take action
                next_obs, reward, done, info = env.step(action)

                # Preprocess next observations
                next_coverage = next_obs["coverage"].reshape(1, 50, 50)
                next_position = next_obs["position"]

                # Store experience in replay buffer
                replay_buffer.push(
                    coverage,
                    position,
                    action,
                    reward,
                    next_coverage,
                    next_position,
                    float(done),
                )

                # Update agent
                if (
                    total_steps % update_interval == 0
                    and len(replay_buffer) > batch_size
                ):
                    batch = replay_buffer.sample(batch_size)
                    agent.update(batch)

                # Update tracking variables
                total_reward += reward
                obs = next_obs
                episode_steps += 1
                total_steps += 1

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
                f"Size: {current_size:.1f} | Reward: {total_reward:.1f} | "
                f"Steps: {episode_steps}"
            )

    except KeyboardInterrupt:
        print("Training stopped by user")

    # Save final model
    torch.save(
        agent.actor.state_dict(),
        "bot_cleaner_basic_coverage/models/ddpg/final_model.pth",
    )

    # Also save critic for potential further training
    torch.save(
        agent.critic.state_dict(),
        "bot_cleaner_basic_coverage/models/ddpg/final_critic.pth",
    )

    env.close()


if __name__ == "__main__":
    train_ddpg()
