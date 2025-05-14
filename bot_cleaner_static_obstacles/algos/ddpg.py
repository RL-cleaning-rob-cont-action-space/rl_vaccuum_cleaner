import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, grid_size_x=25, grid_size_y=15):
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
            nn.ReLU()
        )
        
        # Process walls grid
        self.walls_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU()
        )
        
        # Process position
        self.position_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(256 + 256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Action output (2 values: linear and angular velocity)
        self.action_head = nn.Linear(256, 2)
        
    def forward(self, coverage, walls, position):
        # Reshape inputs if needed
        if len(coverage.shape) == 3:
            coverage = coverage.unsqueeze(1)
        if len(walls.shape) == 3:
            walls = walls.unsqueeze(1)
            
        # Process inputs
        cov_features = self.coverage_net(coverage)
        wall_features = self.walls_net(walls)
        pos_features = self.position_net(position)
        
        # Combine features
        combined = torch.cat([cov_features, wall_features, pos_features], dim=-1)
        shared_out = self.shared_net(combined)
        
        # Output actions in [-1, 1] range
        actions = torch.tanh(self.action_head(shared_out))
        return actions

class Critic(nn.Module):
    def __init__(self, grid_size_x=25, grid_size_y=15):
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
            nn.ReLU()
        )
        
        # Process walls grid
        self.walls_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * grid_size_x * grid_size_y, 256),
            nn.ReLU()
        )
        
        # Process position
        self.position_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Process action
        self.action_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(256 + 256 + 128 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Value output
        self.value_head = nn.Linear(256, 1)
        
    def forward(self, coverage, walls, position, action):
        # Reshape inputs if needed
        if len(coverage.shape) == 3:
            coverage = coverage.unsqueeze(1)
        if len(walls.shape) == 3:
            walls = walls.unsqueeze(1)
            
        # Process inputs
        cov_features = self.coverage_net(coverage)
        wall_features = self.walls_net(walls)
        pos_features = self.position_net(position)
        action_features = self.action_net(action)
        
        # Combine features
        combined = torch.cat([cov_features, wall_features, pos_features, action_features], dim=-1)
        shared_out = self.shared_net(combined)
        
        # Output Q-value
        value = self.value_head(shared_out)
        return value

class DDPG:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = Actor(grid_size_x=env.size_x, grid_size_y=env.size_y).to(self.device)
        self.actor_target = Actor(grid_size_x=env.size_x, grid_size_y=env.size_y).to(self.device)
        self.critic = Critic(grid_size_x=env.size_x, grid_size_y=env.size_y).to(self.device)
        self.critic_target = Critic(grid_size_x=env.size_x, grid_size_y=env.size_y).to(self.device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.001
        self.noise = OUNoise(env.action_space.shape[0])
        
    def act(self, obs):
        coverage = torch.FloatTensor(obs["coverage"]).view(1, self.env.size_y, self.env.size_x).to(self.device)
        walls = torch.FloatTensor(obs["walls"]).view(1, self.env.size_y, self.env.size_x).to(self.device)
        position = torch.FloatTensor(obs["position"]).view(1, -1).to(self.device)
        
        with torch.no_grad():
            action = self.actor(coverage, walls, position)
            action = action.cpu().numpy()[0]
            
        # Add noise for exploration
        action += self.noise.sample()
        
        # Clip action to environment bounds
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        return action

    def update(self, batch):
        # Unpack batch
        coverages, walls, positions, actions, rewards, next_coverages, next_walls, next_positions, dones = batch
        
        # Convert to tensors
        coverages = torch.FloatTensor(coverages).to(self.device)
        walls = torch.FloatTensor(walls).to(self.device)
        positions = torch.FloatTensor(positions).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_coverages = torch.FloatTensor(next_coverages).to(self.device)
        next_walls = torch.FloatTensor(next_walls).to(self.device)
        next_positions = torch.FloatTensor(next_positions).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_coverages, next_walls, next_positions)
            target_q = self.critic_target(next_coverages, next_walls, next_positions, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        current_q = self.critic(coverages, walls, positions, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(coverages, walls, positions, 
                                self.actor(coverages, walls, positions)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update_targets()
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update_targets(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
