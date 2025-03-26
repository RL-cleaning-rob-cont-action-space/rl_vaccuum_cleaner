"""Soft Actor-Critic (SAC) implementation with fixes for vacuum cleaner environment."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
import random
from typing import NamedTuple, List

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Transition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class SquashedGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)

class SAC:
    def __init__(self, state_dim, action_dim, action_range, device='cpu'):
        self.device = device
        self.action_range = action_range
        
        # Networks
        self.actor = SquashedGaussianPolicy(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        
        # Target networks
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        self.hard_update(self.q1_target, self.q1)
        self.hard_update(self.q2_target, self.q2)
        
        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)
        
        # Temperature
        self.alpha = 0.2
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau=0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            mean, log_std = self.actor(state)
            std = log_std.exp()
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                normal = torch.distributions.Normal(mean, std)
                z = normal.rsample()
                action = torch.tanh(z)
                
            action = action.cpu().numpy()
            return self.rescale_action(action)

    def rescale_action(self, action):
        return (action + 1) * (self.action_range[1] - self.action_range[0])/2 + self.action_range[0]

    def update_parameters(self, batch, tau=0.005):
        # Extract components from batch of Transitions
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t.action for t in batch])).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).unsqueeze(-1).to(self.device)

        # Q-function update
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_normal = torch.distributions.Normal(next_mean, next_std)
            next_z = next_normal.rsample()
            next_actions = torch.tanh(next_z)

            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)

            target_q = rewards + (1 - dones) * 0.99 * (q_next - self.alpha * next_normal.log_prob(next_z).sum(-1, keepdim=True))

        # Q losses
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # Policy update
        mean, log_std = self.actor(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        actions_pi = torch.tanh(z)

        q1_pi = self.q1(states, actions_pi)
        q2_pi = self.q2(states, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = self.alpha * normal.log_prob(z).sum(-1, keepdim=True) - q_pi
        policy_loss = policy_loss.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (normal.log_prob(z).sum(-1).detach() + self.target_entropy).mean())
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # Update targets
        self.soft_update(self.q1_target, self.q1, tau)
        self.soft_update(self.q2_target, self.q2, tau)

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.eval()
