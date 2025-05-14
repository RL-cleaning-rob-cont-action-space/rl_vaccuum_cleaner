import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class EnhancedSACNetworks(nn.Module):
    def __init__(self, grid_size_x=10, grid_size_y=10):
        super().__init__()
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

        # Process coverage grid
        self.coverage_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU(),
        )

        # Process walls grid
        self.walls_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU(),
        )

        # Process dirt grid
        self.dirt_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU(),
        )

        # Process position and orientation
        self.position_net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(256 + 256 + 256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )


class Actor(nn.Module):
    def __init__(self, shared_model, action_dim=2):
        super().__init__()
        self.shared = shared_model
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        
        # Initialize output layer with small weights
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        nn.init.xavier_uniform_(self.log_std.weight, gain=0.01)
        
    def forward(self, coverage, walls, dirt, position):
        # Reshape inputs if needed
        if len(coverage.shape) == 3:
            coverage = coverage.unsqueeze(1)  # Add channel dimension
        if len(walls.shape) == 3:
            walls = walls.unsqueeze(1)
        if len(dirt.shape) == 3:
            dirt = dirt.unsqueeze(1)

        # Process coverage grid
        cov_features = self.shared.coverage_net(coverage)

        # Process walls grid
        wall_features = self.shared.walls_net(walls)

        # Process dirt grid
        dirt_features = self.shared.dirt_net(dirt)

        # Process position
        pos_features = self.shared.position_net(position)

        # Combine features
        combined = torch.cat(
            [cov_features, wall_features, dirt_features, pos_features], dim=-1
        )
        shared_out = self.shared.shared_net(combined)
        
        # Get mean and std
        mean = self.mean(shared_out)
        log_std = self.log_std(shared_out)
        log_std = torch.clamp(log_std, -20, 2)  # Limit std range for stability
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, coverage, walls, dirt, position):
        mean, std = self.forward(coverage, walls, dirt, position)
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        
        # Apply tanh squashing
        action = torch.tanh(x_t)
        
        # Calculate log probability, corrected for tanh squashing
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, shared_model):
        super().__init__()
        self.shared = shared_model
        self.q = nn.Linear(256, 1)
        
    def forward(self, coverage, walls, dirt, position):
        # Reshape inputs if needed
        if len(coverage.shape) == 3:
            coverage = coverage.unsqueeze(1)  # Add channel dimension
        if len(walls.shape) == 3:
            walls = walls.unsqueeze(1)
        if len(dirt.shape) == 3:
            dirt = dirt.unsqueeze(1)

        # Process coverage grid
        cov_features = self.shared.coverage_net(coverage)

        # Process walls grid
        wall_features = self.shared.walls_net(walls)

        # Process dirt grid
        dirt_features = self.shared.dirt_net(dirt)

        # Process position
        pos_features = self.shared.position_net(position)

        # Combine features
        combined = torch.cat(
            [cov_features, wall_features, dirt_features, pos_features], dim=-1
        )
        shared_out = self.shared.shared_net(combined)
        
        # Get Q-value
        q_value = self.q(shared_out)
        
        return q_value


class EnhancedSAC:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize shared features network
        self.shared = EnhancedSACNetworks(
            grid_size_x=env.size_x, grid_size_y=env.size_y
        ).to(self.device)
        
        # Initialize actor and critic networks
        self.actor = Actor(self.shared).to(self.device)
        self.critic1 = Critic(self.shared).to(self.device)
        self.critic2 = Critic(self.shared).to(self.device)
        
        # Initialize target critic networks
        self.target_critic1 = Critic(self.shared).to(self.device)
        self.target_critic2 = Critic(self.shared).to(self.device)
        
        # Copy weights to target networks
        self.update_target_networks(tau=1.0)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # Fix duplicate parameter issue by creating separate parameter lists
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())
        
        # Ensure shared parameters are only counted once
        shared_params_ids = set(id(p) for p in self.shared.parameters())
        critic_unique_params = []
        
        # Only add parameters that aren't part of the shared network
        for p in critic1_params + critic2_params:
            if id(p) not in shared_params_ids:
                critic_unique_params.append(p)
                
        self.critic_optimizer = optim.Adam(critic_unique_params, lr=3e-4)
        
        # Initialize shared parameters optimizer separately
        self.shared_optimizer = optim.Adam(self.shared.parameters(), lr=3e-4)
        
        # Entropy coefficient (alpha) - learnable parameter
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Set target entropy to negative of action dimension
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).to(self.device)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.replay_buffer_size = 1000000
        self.replay_buffer = []
        self.min_buffer_size = 200  # Reduced from 1000 for quicker testing
        self.gradient_steps = 1
        
    def get_action(self, obs, evaluate=False):
        # Convert observations to tensors with correct shapes
        coverage = torch.FloatTensor(obs["coverage"]).view(
            1, self.env.size_y, self.env.size_x
        ).to(self.device)
        walls = torch.FloatTensor(obs["walls"]).view(
            1, self.env.size_y, self.env.size_x
        ).to(self.device)
        dirt = torch.FloatTensor(obs["dirt"]).view(
            1, self.env.size_y, self.env.size_x
        ).to(self.device)
        position = torch.FloatTensor(obs["position"]).view(1, -1).to(self.device)

        with torch.no_grad():
            if evaluate:
                # Deterministic action for evaluation
                mean, _ = self.actor(coverage, walls, dirt, position)
                action = torch.tanh(mean)
            else:
                # Stochastic action for training
                action, _, _ = self.actor.sample(coverage, walls, dirt, position)
                
            action = action.cpu().numpy()[0]
            
            # Clip action to environment bounds
            action = np.clip(
                action, self.env.action_space.low, self.env.action_space.high
            )
            
        return action
    
    def add_to_replay_buffer(self, obs, action, reward, next_obs, done):
        self.replay_buffer.append({
            "coverage": obs["coverage"],
            "walls": obs["walls"],
            "dirt": obs["dirt"],
            "position": obs["position"],
            "action": action,
            "reward": reward,
            "next_coverage": next_obs["coverage"],
            "next_walls": next_obs["walls"],
            "next_dirt": next_obs["dirt"],
            "next_position": next_obs["position"],
            "done": done
        })
        
        # Keep buffer size limited
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)
    
    def sample_batch(self):
        # Randomly sample a batch from replay buffer
        indices = np.random.randint(0, len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Convert to tensors
        coverages = torch.FloatTensor(np.array([t["coverage"] for t in batch])).view(
            -1, self.env.size_y, self.env.size_x
        ).to(self.device)
        walls = torch.FloatTensor(np.array([t["walls"] for t in batch])).view(
            -1, self.env.size_y, self.env.size_x
        ).to(self.device)
        dirt = torch.FloatTensor(np.array([t["dirt"] for t in batch])).view(
            -1, self.env.size_y, self.env.size_x
        ).to(self.device)
        positions = torch.FloatTensor(np.array([t["position"] for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t["action"] for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t["reward"] for t in batch])).view(-1, 1).to(self.device)
        next_coverages = torch.FloatTensor(np.array([t["next_coverage"] for t in batch])).view(
            -1, self.env.size_y, self.env.size_x
        ).to(self.device)
        next_walls = torch.FloatTensor(np.array([t["next_walls"] for t in batch])).view(
            -1, self.env.size_y, self.env.size_x
        ).to(self.device)
        next_dirt = torch.FloatTensor(np.array([t["next_dirt"] for t in batch])).view(
            -1, self.env.size_y, self.env.size_x
        ).to(self.device)
        next_positions = torch.FloatTensor(np.array([t["next_position"] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t["done"] for t in batch])).view(-1, 1).to(self.device)
        
        return (coverages, walls, dirt, positions, actions, rewards, 
                next_coverages, next_walls, next_dirt, next_positions, dones)
    
    def update(self):
        if len(self.replay_buffer) < self.min_buffer_size:
            return
        
        try:
            for _ in range(self.gradient_steps):
                # Sample a batch from replay buffer
                (coverages, walls, dirt, positions, actions, rewards, 
                 next_coverages, next_walls, next_dirt, next_positions, dones) = self.sample_batch()
                
                # Current alpha value
                alpha = self.log_alpha.exp()
                
                # ========== Update Critic ==========
                with torch.no_grad():
                    # Sample next actions and get log probs from the actor
                    next_actions, next_log_probs, _ = self.actor.sample(
                        next_coverages, next_walls, next_dirt, next_positions
                    )
                    
                    # Get target Q values from target critics
                    target_q1 = self.target_critic1(next_coverages, next_walls, next_dirt, next_positions)
                    target_q2 = self.target_critic2(next_coverages, next_walls, next_dirt, next_positions)
                    
                    # Take minimum Q value for stability
                    target_q = torch.min(target_q1, target_q2)
                    
                    # Calculate target with entropy regularization
                    target_q = rewards + (1 - dones) * self.gamma * (target_q - alpha * next_log_probs)
                
                # Get current Q values from critics
                current_q1 = self.critic1(coverages, walls, dirt, positions)
                current_q2 = self.critic2(coverages, walls, dirt, positions)
                
                # Calculate critic loss (MSE)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                
                # Update critics
                self.shared_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.shared_optimizer.step()
                
                # ========== Update Actor ==========
                # Get actions and log probs from the actor
                new_actions, log_probs, _ = self.actor.sample(coverages, walls, dirt, positions)
                
                # Get Q values for new actions
                q1 = self.critic1(coverages, walls, dirt, positions)
                q2 = self.critic2(coverages, walls, dirt, positions)
                q = torch.min(q1, q2)
                
                # Calculate actor loss (maximizing Q value and entropy)
                actor_loss = (alpha * log_probs - q).mean()
                
                # Update actor
                self.shared_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.shared_optimizer.step()
                
                # ========== Update Alpha ==========
                # Optimize entropy coefficient
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                # ========== Update Target Networks ==========
                self.update_target_networks(self.tau)
        except Exception as e:
            print(f"Error during update: {e}")
            import traceback
            traceback.print_exc()
    
    def update_target_networks(self, tau):
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, path):
        torch.save({
            'shared_state_dict': self.shared.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.shared.load_state_dict(checkpoint['shared_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.log_alpha = checkpoint['log_alpha'] 