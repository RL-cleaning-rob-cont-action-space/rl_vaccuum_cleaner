import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
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

        # Action head
        self.actor = nn.Linear(256, 2).to(device)
        self.log_std = nn.Parameter(torch.zeros(2)).to(device)
        
        # Value head
        self.critic = nn.Linear(256, 1).to(device)

    def forward(self, coverage, position):
        batch_size = coverage.size(0)
        
        # Process coverage map
        cov_features = self.conv_layers(coverage).view(batch_size, -1)
        
        # Process position
        pos_features = self.pos_encoder(position)
        
        # Combine features
        combined = torch.cat([cov_features, pos_features], dim=1)
        shared_out = self.shared_net(combined)

        # Get policy parameters and value
        mean = self.actor(shared_out)
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.critic(shared_out).squeeze()

        return mean, std, value

class PPO:
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        self.policy = ActorCritic(device=device)
        
        # Hyperparameters
        self.lr = 1e-4
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.epsilon_clip = 0.2
        self.epochs = 10
        self.batch_size = 256
        self.max_grad_norm = 0.5
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def act(self, coverage, position):
        with torch.no_grad():
            coverage = torch.FloatTensor(coverage).to(self.device)
            position = torch.FloatTensor(position).to(self.device)
            mean, std, value = self.policy(coverage, position)
            
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            action = action.cpu().numpy().squeeze()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        return action, log_prob.item(), value.item()

    def update(self, rollout):
        # Prepare batch data
        coverages = np.array([t['coverage'] for t in rollout], dtype=np.float32)
        positions = np.array([t['position'] for t in rollout], dtype=np.float32)
        actions = np.array([t['action'] for t in rollout], dtype=np.float32)
        rewards = np.array([t['reward'] for t in rollout], dtype=np.float32)
        dones = np.array([t['done'] for t in rollout], dtype=np.float32)
        log_probs = np.array([t['log_prob'] for t in rollout], dtype=np.float32)
        values = np.array([t['value'] for t in rollout], dtype=np.float32)

        # Convert to tensors and move to device
        coverages = torch.FloatTensor(coverages).unsqueeze(1).to(self.device)
        positions = torch.FloatTensor(positions).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_values = torch.FloatTensor(values).to(self.device)

        # Compute advantages and returns
        advantages = self.compute_gae(rewards, old_values, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.epochs):
            indices = torch.randperm(len(coverages))
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                
                # Get batch data
                cov_batch = coverages[batch_idx]
                pos_batch = positions[batch_idx]
                act_batch = actions[batch_idx]
                old_lp_batch = old_log_probs[batch_idx]
                ret_batch = returns[batch_idx]
                adv_batch = advantages[batch_idx]

                # Forward pass
                mean, std, values = self.policy(cov_batch, pos_batch)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(act_batch).sum(dim=1)
                entropy = dist.entropy().mean()

                # Compute losses
                ratios = torch.exp(log_probs - old_lp_batch)
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * adv_batch
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_batch)
                loss = (policy_loss + 
                       self.value_coef * value_loss - 
                       self.entropy_coef * entropy)

                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # Return average losses
        num_updates = self.epochs * (len(coverages) // self.batch_size + 1)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }

    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards).to(self.device)
        last_advantage = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t] * next_non_terminal
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.lambda_gae * last_advantage
            last_advantage = advantages[t]
            next_value = values[t]

        return advantages
