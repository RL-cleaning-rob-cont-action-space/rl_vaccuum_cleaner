"""Training script for Deep Deterministic Policy Gradient (DDPG) vacuum cleaner agent."""
import numpy as np
import torch
from bot_cleaner_dynamic_environment.environments.environment import ContinuousVacuumCleanerEnv, SimpleWallEnv
from bot_cleaner_dynamic_environment.algos.ddpg import DDPG
from collections import deque
import os

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, coverage, position, action, reward, next_coverage, next_position, done):
        self.buffer.append((coverage, position, action, reward, next_coverage, next_position, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        return map(np.stack, zip(*samples))
    
    def __len__(self):
        return len(self.buffer)

def preprocess_observation(obs: dict) -> tuple:
    """Process observation for DDPG network input."""
    coverage = obs['coverage'].reshape(1, 1, 50, 50)
    position = obs['position'].reshape(1, -1)
    return coverage, position

def train():
    # Create directories for model saving
    os.makedirs("bot_cleaner_dynamic_environment/models/ddpg", exist_ok=True)
    
    # Environment setup
    env = ContinuousVacuumCleanerEnv(size=5.0, coverage_radius=0.5)
    
    # Check action space
    print(f"Action space: {env.action_space}")
    action_dim = env.action_space.shape[0]
    
    agent = DDPG(env)
    replay_buffer = ReplayBuffer()

    # Training parameters
    max_episodes = 1000
    batch_size = 256
    min_buffer_size = 5000
    update_interval = 50
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
            steps = 0

            while not done:
                # Environment interaction
                coverage, position = preprocess_observation(obs)
                action = agent.get_action(coverage, position)
                
                # Fix action shape - ensure it's a 1D array with 2 elements
                action = action.squeeze()  # Remove batch dimension
                if len(action.shape) > 1:
                    action = action[0]  # Take first action if multiple
                
                # Clip actions to environment bounds
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                # Debug print for first few steps
                if steps < 5:
                    print(f"Action shape: {action.shape}, Action values: {action}")
                    
                next_obs, reward, done, info = env.step(action)

                # Store experience
                replay_buffer.push(
                    coverage.squeeze(),
                    position.squeeze(),
                    action,
                    reward,
                    next_obs['coverage'].reshape(1, 50, 50),
                    next_obs['position'],
                    done
                )

                total_reward += reward
                obs = next_obs
                steps += 1

                # Render if enabled
                if render:
                    env.render()

                # Update if enough samples
                if len(replay_buffer) > min_buffer_size and steps % update_interval == 0:
                    experiences = replay_buffer.sample(batch_size)
                    critic_loss, actor_loss = agent.update(experiences)
                    print(f"Critic Loss: {critic_loss:.3f}, Actor Loss: {actor_loss:.3f}")

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
                agent.save(
                    f"bot_cleaner_dynamic_environment/models/ddpg/best_model_{best_coverage:.2f}.pth"
                )
                print(f"New best coverage: {best_coverage:.2%}")

            # Periodic save
            if episode % save_interval == 0:
                agent.save(
                    f"bot_cleaner_dynamic_environment/models/ddpg/ddpg_cleaner_ep{episode}.pth"
                )

    except KeyboardInterrupt:
        print("Training interrupted! Saving final model...")

    finally:
        # Final save and cleanup
        agent.save("bot_cleaner_dynamic_environment/models/ddpg/ddpg_cleaner_final.pth")
        env.close()

if __name__ == "__main__":
    train()
