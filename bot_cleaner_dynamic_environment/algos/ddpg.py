import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple, Tuple

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # CNN for processing coverage map
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        ).to(device)
        
        # MLP for processing position
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ).to(device)
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        ).to(device)
        
        # Action outputs with mean and log_std like SAC
        self.mean = nn.Linear(256, 2).to(device)
        self.log_std = nn.Linear(256, 2).to(device)
        
    def forward(self, coverage, position) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = coverage.size(0)
        
        # Process coverage map
        cov_features = self.conv_layers(coverage).view(batch_size, -1)
        
        # Process position
        pos_features = self.pos_encoder(position)
        
        # Combine features
        combined = torch.cat([cov_features, pos_features], dim=1)
        shared_out = self.shared_net(combined)
        
        # Get mean and log_std like SAC
        mean = self.mean(shared_out)
        log_std = self.log_std(shared_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        # CNN and position processing similar to Actor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        ).to(device)
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ).to(device)
        
        # Action processing
        self.action_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        ).to(device)
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(256 + 128 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        ).to(device)
        
        self.q_out = nn.Linear(256, 1).to(device)
        
    def forward(self, coverage, position, action):
        batch_size = coverage.size(0)
        cov_features = self.conv_layers(coverage).view(batch_size, -1)
        pos_features = self.pos_encoder(position)
        action_features = self.action_encoder(action)
        
        combined = torch.cat([cov_features, pos_features, action_features], dim=1)
        shared_out = self.shared_net(combined)
        q_value = self.q_out(shared_out)
        return q_value

class DDPG:
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        
        # Initialize networks
        self.actor = Actor(device).to(device)
        self.actor_target = Actor(device).to(device)
        self.critic1 = Critic(device).to(device)  # Dual critics like SAC
        self.critic2 = Critic(device).to(device)
        self.critic1_target = Critic(device).to(device)
        self.critic2_target = Critic(device).to(device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Initialize optimizers with SAC learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
        
        # Hyperparameters combining DDPG, PPO and SAC
        self.gamma = 0.99
        self.tau = 0.005  # Softer update like SAC
        self.batch_size = 256
        self.max_grad_norm = 0.5
        self.alpha = 0.2  # Temperature parameter from SAC
        
    def get_action(self, coverage, position, evaluate=False):
        with torch.no_grad():
            coverage = torch.FloatTensor(coverage).to(self.device)
            if len(coverage.shape) == 5:
                coverage = coverage.squeeze(0)
            position = torch.FloatTensor(position).to(self.device)
            if len(position.shape) == 3:
                position = position.squeeze(0).squeeze(0)
            
            mean, log_std = self.actor(coverage, position)
            
            if evaluate:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                x = normal.rsample()
                action = torch.tanh(x)
            
            action = action.cpu().numpy()
            # Ensure action is 2D
            if action.ndim == 1:
                action = action.reshape(-1)
            
            # Debug print
            print(f"Raw action shape: {action.shape}, values: {action}")
            
            return np.clip(action, self.env.action_space.low, self.env.action_space.high)
    
    def update(self, experiences):
        coverages, positions, actions, rewards, next_coverages, next_positions, dones = experiences
        
        # Convert to tensors
        coverages = torch.FloatTensor(coverages).unsqueeze(1).to(self.device)
        positions = torch.FloatTensor(positions).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_coverages = torch.FloatTensor(next_coverages).unsqueeze(1).to(self.device)
        next_positions = torch.FloatTensor(next_positions).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critics (using dual critics like SAC)
        with torch.no_grad():
            next_mean, next_log_std = self.actor_target(next_coverages, next_positions)
            next_std = next_log_std.exp()
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_actions = torch.tanh(next_normal.rsample())
            
            q1_next = self.critic1_target(next_coverages, next_positions, next_actions)
            q2_next = self.critic2_target(next_coverages, next_positions, next_actions)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + (1 - dones) * self.gamma * q_next
        
        # Update both critics
        current_q1 = self.critic1(coverages, positions, actions)
        current_q2 = self.critic2(coverages, positions, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()
        
        # Update actor using SAC-style update
        mean, log_std = self.actor(coverages, positions)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        actions_new = torch.tanh(x)
        
        q1 = self.critic1(coverages, positions, actions_new)
        q2 = self.critic2(coverages, positions, actions_new)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * normal.log_prob(x).sum(dim=-1) - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Soft update targets
        self._soft_update()
        
        return (critic1_loss.item() + critic2_loss.item()) / 2, actor_loss.item()
    
    def _soft_update(self):
        # Softer update like SAC
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
