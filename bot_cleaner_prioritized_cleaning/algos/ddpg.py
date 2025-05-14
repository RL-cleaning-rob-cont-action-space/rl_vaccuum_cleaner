import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class EnhancedFeatureNetwork(nn.Module):
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

        # Shared network for combining features
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
        self.action_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # DDPG uses tanh for bounded actions
        )
        
        # Initialize output layer with small weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        
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
        
        # Get actions
        actions = self.action_net(shared_out)
        
        return actions


class Critic(nn.Module):
    def __init__(self, shared_model, action_dim=2):
        super().__init__()
        self.shared = shared_model
        
        # Additional layers to integrate action
        self.action_layer = nn.Linear(action_dim, 128)
        self.q_layer1 = nn.Linear(256 + 128, 256)
        self.q_layer2 = nn.Linear(256, 128)
        self.q_layer3 = nn.Linear(128, 1)
        
        # Initialize output layer with small weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, coverage, walls, dirt, position, action):
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
        
        # Process action
        action_out = F.relu(self.action_layer(action))
        
        # Combine state and action features
        x = torch.cat([shared_out, action_out], dim=1)
        x = F.relu(self.q_layer1(x))
        x = F.relu(self.q_layer2(x))
        q_value = self.q_layer3(x)
        
        return q_value


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration"""
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


class EnhancedDDPG:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize shared feature network
        self.shared = EnhancedFeatureNetwork(
            grid_size_x=env.size_x, grid_size_y=env.size_y
        ).to(self.device)
        
        # Initialize actor and critic networks
        self.actor = Actor(self.shared).to(self.device)
        self.critic = Critic(self.shared).to(self.device)
        
        # Initialize target networks
        self.target_actor = Actor(EnhancedFeatureNetwork(
            grid_size_x=env.size_x, grid_size_y=env.size_y
        ).to(self.device)).to(self.device)
        self.target_critic = Critic(self.target_actor.shared).to(self.device)
        
        # Copy weights to target networks
        self.update_target_networks(tau=1.0)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # Fix duplicate parameter issue
        critic_params = list(self.critic.parameters())
        shared_params_ids = set(id(p) for p in self.shared.parameters())
        critic_unique_params = []
        
        # Only add parameters that aren't part of the shared network
        for p in critic_params:
            if id(p) not in shared_params_ids:
                critic_unique_params.append(p)
        
        self.critic_optimizer = optim.Adam(critic_unique_params, lr=1e-3)
        self.shared_optimizer = optim.Adam(self.shared.parameters(), lr=1e-4)
        
        # Initialize noise process for exploration
        self.noise = OUNoise(env.action_space.shape[0])
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.001
        self.batch_size = 64
        self.replay_buffer_size = 1000000
        self.replay_buffer = []
        self.min_buffer_size = 200  # For quicker testing
        self.gradient_steps = 1
        self.noise_scale = 0.5  # Scale for exploration noise
        
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
            # Get deterministic action from actor
            action = self.actor(coverage, walls, dirt, position)
            action = action.cpu().numpy()[0]
            
            # Add noise if not evaluating
            if not evaluate:
                noise = self.noise.sample() * self.noise_scale
                action = action + noise
            
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
                
                # ========== Update Critic ==========
                with torch.no_grad():
                    # Get next actions from target actor
                    next_actions = self.target_actor(next_coverages, next_walls, next_dirt, next_positions)
                    
                    # Get target Q values
                    target_q = self.target_critic(
                        next_coverages, next_walls, next_dirt, next_positions, next_actions
                    )
                    
                    # Calculate target with Bellman equation
                    target_q = rewards + (1 - dones) * self.gamma * target_q
                
                # Get current Q values
                current_q = self.critic(coverages, walls, dirt, positions, actions)
                
                # Calculate critic loss (MSE)
                critic_loss = F.mse_loss(current_q, target_q)
                
                # Update critic
                self.shared_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.shared_optimizer.step()
                
                # ========== Update Actor ==========
                # Get actions from actor
                pred_actions = self.actor(coverages, walls, dirt, positions)
                
                # Calculate actor loss (negative Q value)
                actor_loss = -self.critic(coverages, walls, dirt, positions, pred_actions).mean()
                
                # Update actor
                self.shared_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.shared_optimizer.step()
                
                # ========== Update Target Networks ==========
                self.update_target_networks(self.tau)
        except Exception as e:
            print(f"Error during update: {e}")
            import traceback
            traceback.print_exc()
    
    def update_target_networks(self, tau):
        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'shared_state_dict': self.shared.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.shared.load_state_dict(checkpoint['shared_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.eval()
        self.critic.eval() 