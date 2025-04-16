import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128)
        )

        self.shared_net = nn.Sequential(
            nn.Linear(256 + 128, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )

        self.action_head = nn.Linear(256, 2)

    def forward(self, coverage, position):
        batch_size = coverage.size(0)
        cov_features = self.conv_layers(coverage).view(batch_size, -1)
        pos_features = self.pos_encoder(position)
        combined = torch.cat([cov_features, pos_features], dim=1)
        shared_out = self.shared_net(combined)

        # Output actions in the range [-1, 1]
        actions = torch.tanh(self.action_head(shared_out))
        return actions


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128)
        )

        self.action_encoder = nn.Sequential(nn.Linear(2, 64), nn.ReLU())

        self.shared_net = nn.Sequential(
            nn.Linear(256 + 128 + 64, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )

        self.value_head = nn.Linear(256, 1)

    def forward(self, coverage, position, action):
        batch_size = coverage.size(0)
        cov_features = self.conv_layers(coverage).view(batch_size, -1)
        pos_features = self.pos_encoder(position)
        action_features = self.action_encoder(action)

        combined = torch.cat([cov_features, pos_features, action_features], dim=1)
        shared_out = self.shared_net(combined)

        value = self.value_head(shared_out)
        return value


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


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self, coverage, position, action, reward, next_coverage, next_position, done
    ):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            coverage,
            position,
            action,
            reward,
            next_coverage,
            next_position,
            done,
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        coverage, position, action, reward, next_coverage, next_position, done = map(
            np.stack, zip(*batch)
        )
        return coverage, position, action, reward, next_coverage, next_position, done

    def __len__(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, env):
        self.env = env
        self.actor = Actor()
        self.actor_target = Actor()
        self.critic = Critic()
        self.critic_target = Critic()

        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.noise = OUNoise(self.env.action_space.shape[0])

        self.gamma = 0.99
        self.tau = 0.001  # Target network update rate

    def select_action(self, coverage, position, evaluate=False):
        coverage = torch.FloatTensor(coverage).unsqueeze(0)
        position = torch.FloatTensor(position).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(coverage, position).cpu().numpy()[0]
        self.actor.train()

        if not evaluate:
            noise = self.noise.sample()
            action = action + noise

        # Scale from [-1, 1] to environment action space
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high
        scaled_action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)
        scaled_action = np.clip(scaled_action, action_low, action_high)

        return scaled_action

    def update(self, batch):
        # Process batch data
        (
            coverages,
            positions,
            actions,
            rewards,
            next_coverages,
            next_positions,
            dones,
        ) = batch

        # Fix tensor dimensions - remove the extra dimension that's causing the error
        coverages = torch.FloatTensor(
            coverages
        )  # Shape should be [batch_size, 1, 50, 50]
        positions = torch.FloatTensor(positions)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_coverages = torch.FloatTensor(
            next_coverages
        )  # Shape should be [batch_size, 1, 50, 50]
        next_positions = torch.FloatTensor(next_positions)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute the target Q value
        with torch.no_grad():
            next_actions = self.actor_target(next_coverages, next_positions)
            target_q = self.critic_target(next_coverages, next_positions, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Get current Q estimate
        current_q = self.critic(coverages, positions, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_actions = self.actor(coverages, positions)
        actor_loss = -self.critic(coverages, positions, actor_actions).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._update_target_networks()

    def _update_target_networks(self):
        # Update target networks using Polyak averaging
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def act(self, coverage, position):
        # Reshape coverage to match expected dimensions
        coverage = coverage.reshape(1, 1, 50, 50)
        position = position.reshape(1, -1)

        return self.select_action(coverage, position)
