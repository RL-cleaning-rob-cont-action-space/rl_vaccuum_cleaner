"""Training script for Proximal Policy Optimization (PPO) vacuum cleaner agent."""
import numpy as np
import torch
from bot_cleaner_dynamic_environment.environments.environment import ContinuousVacuumCleanerEnv, SimpleWallEnv
from bot_cleaner_dynamic_environment.algos.ppo import PPO
import os

def preprocess_observation(obs: dict) -> tuple:
    """Process observation for PPO network input."""
    coverage = obs['coverage'].reshape(1, 1, 50, 50)
    position = obs['position'].reshape(1, -1)
    return coverage, position

def train():
    # Environment setup
    env = ContinuousVacuumCleanerEnv(size=5.0, coverage_radius=0.5)
    agent = PPO(env)

    # Training parameters
    max_episodes = 1000
    update_interval = 500
    print_interval = 10
    save_interval = 100
    size_increments = [(200, 7.0), (400, 8.0), (600, 9.0), (800, 10.0)]
    render = True
    best_coverage = 0

    try:
        # Training loop
        for episode in range(max_episodes):
            # Update environment size based on current episode
            current_size = 5.0
            for threshold, size in size_increments:
                if episode >= threshold:
                    current_size = size
            env.size = current_size
            env.cell_size = env.size / env.resolution
            env.max_steps = int(2000 * (current_size/5.0))

            obs = env.reset()
            done = False
            total_reward = 0
            rollout = []

            while not done:
                # Environment interaction
                coverage, position = preprocess_observation(obs)
                action, log_prob, value = agent.act(coverage, position)
                next_obs, reward, done, info = env.step(action)

                # Store experience
                rollout.append({
                    'coverage': coverage.squeeze(),
                    'position': position.squeeze(),
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'log_prob': log_prob,
                    'value': value
                })

                # Print step information
                print(f"Step: {info['steps']}, Reward: {reward:.3f}, "
                      f"Coverage: {info['coverage_percentage']:.2%}, "
                      f"Action: {action}")

                total_reward += reward
                obs = next_obs

                # Render if enabled
                if render:
                    env.render()

                # Update if needed
                if len(rollout) >= update_interval:
                    loss_stats = agent.update(rollout)
                    print(f"Policy Loss: {loss_stats.get('policy_loss', 0):.3f}, "
                          f"Value Loss: {loss_stats.get('value_loss', 0):.3f}")
                    rollout = []

            # Final update for episode
            if rollout:
                loss_stats = agent.update(rollout)

            # Logging
            if episode % print_interval == 0:
                print(f"Episode {episode:4d} | "
                      f"Total Reward: {total_reward:7.1f} | "
                      f"Size: {current_size:.1f} | "
                      f"Coverage: {info['coverage_percentage']:.2%}")

            # Save best model
            current_cov = info['coverage_percentage']
            if current_cov > best_coverage:
                best_coverage = current_cov
                torch.save(agent.policy.state_dict(), 
                          f"bot_cleaner_dynamic_environment/models/ppo/best_model_{best_coverage:.2f}.pth")
                print(f"New best coverage: {best_coverage:.2%}")

            # Periodic save
            if episode % save_interval == 0:
                torch.save(agent.policy.state_dict(),
                          f"bot_cleaner_dynamic_environment/models/ppo/ppo_cleaner_ep{episode}.pth")

    except KeyboardInterrupt:
        print("Training interrupted! Saving final model...")

    # Final save and cleanup
    torch.save(agent.policy.state_dict(), 
              "bot_cleaner_dynamic_environment/models/ppo/ppo_cleaner_final.pth")
    env.close()

if __name__ == "__main__":
    train()
