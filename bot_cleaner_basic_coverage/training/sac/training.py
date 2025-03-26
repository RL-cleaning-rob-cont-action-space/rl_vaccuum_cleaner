"""Training script for Soft Actor-Critic (SAC) vacuum cleaner agent."""
import numpy as np
import torch
from bot_cleaner_basic_coverage.environments.environment import ContinuousVacuumCleanerEnv
from bot_cleaner_basic_coverage.algos.sac import SAC, ReplayBuffer  # Import from your SAC implementation

def preprocess_observation(obs: dict) -> np.ndarray:
    """Flatten environment observation into state vector."""
    return np.concatenate([
        obs['coverage'].flatten(),  # Flatten 50x50 coverage map
        obs['position']             # Add position coordinates (x, y, theta)
    ])

def train():
    # Environment setup
    env = ContinuousVacuumCleanerEnv(size=10.0, coverage_radius=0.5)
    state_dim = 50*50 + 3  # Coverage map size + position coordinates
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)

    # Agent and buffer initialization
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_range=action_range,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    buffer = ReplayBuffer(capacity=100000)

    # Training parameters
    max_episodes = 1000
    batch_size = 256
    print_interval = 10
    render = True  # Set to False for faster training

    try:
        # Training loop
        for episode in range(max_episodes):
            obs = env.reset()
            state = preprocess_observation(obs)
            total_reward = 0
            done = False

            while not done:
                # Environment interaction
                action = agent.get_action(state)
                next_obs, reward, done, _ = env.step(action)
                next_state = preprocess_observation(next_obs)

                # Store experience
                buffer.push(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                # Render if enabled
                if render:
                    env.render()

                # Update agent
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    agent.update_parameters(batch)

            # Logging
            if episode % print_interval == 0:
                print(f"Episode {episode:4d} | "
                    f"Total Reward: {total_reward:7.1f} | "
                    f"Alpha: {agent.alpha.item():.3f}")

            # Save model periodically
            if episode % 100 == 0:
                torch.save(agent.actor.state_dict(),
                        f"bot_cleaner_basic_coverage/models/sac/sac_cleaner_actor_ep{episode}.pth")

    except KeyboardInterrupt:
        print("Training interrupted! Saving final model...")

    # Final save and cleanup
    torch.save(agent.actor.state_dict(), "bot_cleaner_basic_coverage/models/sac/sac_cleaner_actor_final.pth")
    env.close()

if __name__ == "__main__":
    train()
