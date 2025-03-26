import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Coverage feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Position encoder
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128)
        )

        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(256 + 128, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )

        # Action output layer
        self.action_layer = nn.Linear(256, output_dim)

    def forward(self, coverage, position):
        # Ensure the input is 4D (batch, channel, height, width)
        if coverage.dim() == 5:
            coverage = coverage.squeeze(1)
        elif coverage.dim() == 3:
            coverage = coverage.unsqueeze(0)

        batch_size = coverage.size(0)

        # Extract features from coverage and position
        cov_features = self.conv_layers(coverage).view(batch_size, -1)
        pos_features = self.pos_encoder(position)

        # Combine features
        combined = torch.cat([cov_features, pos_features], dim=1)
        shared_out = self.shared_net(combined)

        # Generate action
        action = torch.tanh(self.action_layer(shared_out))
        return action


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()

        # Coverage feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Position encoder
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128)
        )

        # Q-value network
        self.q_network = nn.Sequential(
            nn.Linear(256 + 128 + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, coverage, position, action):
        # Ensure the input is 4D (batch, channel, height, width)
        if coverage.dim() == 5:
            coverage = coverage.squeeze(1)
        elif coverage.dim() == 3:
            coverage = coverage.unsqueeze(0)

        batch_size = coverage.size(0)

        # Extract features from coverage and position
        cov_features = self.conv_layers(coverage).view(batch_size, -1)
        pos_features = self.pos_encoder(position)

        # Combine features with action
        combined = torch.cat([cov_features, pos_features, action], dim=1)
        q_value = self.q_network(combined)

        return q_value


class DDPG:
    def __init__(
        self, env, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.001, noise_std=0.1
    ):
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = Actor(1, self.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = Critic(1, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std

    def act(self, coverage, position, explore=True):
        with torch.no_grad():
            # Reshape coverage to 2D grid
            coverage_grid = coverage.reshape(self.env.resolution, self.env.resolution)

            # Ensure correct input shape for coverage
            coverage_tensor = torch.FloatTensor(coverage_grid).unsqueeze(0).unsqueeze(0)

            # Ensure correct input shape for position
            position_tensor = torch.FloatTensor(position).unsqueeze(0)

            action = self.actor(coverage_tensor, position_tensor).squeeze(0).numpy()

            # Add exploration noise
            if explore:
                noise = np.random.normal(0, self.noise_std, size=self.action_dim)
                action = np.clip(
                    action + noise,
                    self.env.action_space.low,
                    self.env.action_space.high,
                )

            return action

    def update(self, batch):
        # Reshape coverage grids
        coverages = torch.FloatTensor(
            [
                b["coverage"].reshape(self.env.resolution, self.env.resolution)
                for b in batch
            ]
        ).unsqueeze(1)

        next_coverages = torch.FloatTensor(
            [
                b["next_coverage"].reshape(self.env.resolution, self.env.resolution)
                for b in batch
            ]
        ).unsqueeze(1)

        # Unpack other batch components
        positions = torch.FloatTensor(np.array([b["position"] for b in batch]))
        actions = torch.FloatTensor(np.array([b["action"] for b in batch]))
        rewards = torch.FloatTensor(np.array([b["reward"] for b in batch]))
        next_positions = torch.FloatTensor(
            np.array([b["next_position"] for b in batch])
        )
        dones = torch.FloatTensor(np.array([b["done"] for b in batch]))

        # Critic Update
        with torch.no_grad():
            next_actions = self.actor_target(next_coverages, next_positions)
            next_q_values = self.critic_target(
                next_coverages, next_positions, next_actions
            )
            target_q_values = (
                rewards + (1 - dones) * self.gamma * next_q_values.squeeze()
            )

        current_q_values = self.critic(coverages, positions, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        predicted_actions = self.actor(coverages, positions)
        actor_loss = -self.critic(coverages, positions, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, network, target_network):
        for param, target_param in zip(
            network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
