from bot_cleaner_static_obstacles.environments.environment import GridMazeVacuumCleanerEnv
from bot_cleaner_static_obstacles.algos.ddpg import DDPG
import torch
import numpy as np
import time
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        # Separate state components
        coverages = np.stack([s['coverage'] for s in states])
        walls = np.stack([s['walls'] for s in walls])
        positions = np.stack([s['position'] for s in positions])
        
        next_coverages = np.stack([s['coverage'] for s in next_states])
        next_walls = np.stack([s['walls'] for s in next_states])
        next_positions = np.stack([s['position'] for s in next_positions])
        
        return (coverages, walls, positions, np.array(actions), 
                np.array(rewards), next_coverages, next_walls, 
                next_positions, np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

def main():
    # Initialize environment and agent
    env = GridMazeVacuumCleanerEnv(size_x=25, size_y=15, max_steps=1000)
    agent = DDPG(env)
    replay_buffer = ReplayBuffer()

    # Training parameters
    max_episodes = 1000
    batch_size = 64
    min_buffer_size = 1000
    best_coverage = 0
    render_every = 20
    save_interval = 50

    # Training statistics
    ep_rewards = []
    ep_coverages = []
    ep_lengths = []
    
    try:
        for episode in range(1, max_episodes + 1):
            obs = env.reset()
            agent.noise.reset()  # Reset exploration noise
            done = False
            total_reward = 0
            start_time = time.time()
            
            render_episode = (episode % render_every == 0) or (episode == 1)
            
            while not done:
                if render_episode:
                    env.render()
                    time.sleep(0.02)
                
                # Select action
                action = agent.act(obs)
                
                # Take step in environment
                next_obs, reward, done, info = env.step(action)
                
                # Store experience
                replay_buffer.push(obs, action, reward, next_obs, done)
                
                total_reward += reward
                obs = next_obs
                
                # Update networks if enough samples
                if len(replay_buffer) > min_buffer_size:
                    batch = replay_buffer.sample(batch_size)
                    critic_loss, actor_loss = agent.update(batch)
            
            # Calculate episode statistics
            ep_time = time.time() - start_time
            coverage = info['coverage_percentage']
            steps = info['steps']
            
            # Save statistics
            ep_rewards.append(total_reward)
            ep_coverages.append(coverage)
            ep_lengths.append(steps)
            
            # Save best model
            if coverage > best_coverage:
                best_coverage = coverage
                agent.save(f"bot_cleaner_static_obstacles/models/ddpg/best_model_{best_coverage:.2f}.pth")
                print(f"New best model saved with coverage: {best_coverage:.2%}")
            
            # Periodic model saving
            if episode % save_interval == 0:
                agent.save(f"bot_cleaner_static_obstacles/models/ddpg/model_ep_{episode}.pth")
            
            # Print episode summary
            print(f"Episode {episode:4d} | "
                  f"Coverage: {coverage:.2%} | "
                  f"Steps: {steps:4d}/{env.max_steps} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Time: {ep_time:.2f}s")
            
            # Print moving averages every 50 episodes
            if episode % 50 == 0:
                avg_reward = np.mean(ep_rewards[-50:])
                avg_coverage = np.mean(ep_coverages[-50:])
                avg_length = np.mean(ep_lengths[-50:])
                print(f"\nLast 50 episodes average:")
                print(f"Coverage: {avg_coverage:.2%} | "
                      f"Steps: {avg_length:.1f} | "
                      f"Reward: {avg_reward:.2f}\n")
    
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    
    finally:
        # Save final model and close environment
        agent.save("bot_cleaner_static_obstacles/models/ddpg/final_model.pth")
        env.close()
        
        # Print final statistics
        if len(ep_rewards) > 0:
            print("\nTraining completed.")
            print(f"Final coverage: {ep_coverages[-1]:.2%}")
            print(f"Average reward: {np.mean(ep_rewards):.2f}")
            print(f"Average coverage: {np.mean(ep_coverages):.2%}")
            print(f"Average episode length: {np.mean(ep_lengths):.1f} steps")

if __name__ == "__main__":
    main()
    