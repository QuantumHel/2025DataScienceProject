from typing import Tuple
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.rl.env import Array3D

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)

def _build_model(n_qubits=4):
    class CNNFactorizedQNet(nn.Module):
        def __init__(self, n_qubits):
            super().__init__()
            self.n_qubits = n_qubits
            self.backbone = nn.Sequential(
                nn.Conv2d(8, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                nn.AdaptiveAvgPool2d((n_qubits, n_qubits)),
                nn.Flatten(),
                nn.LayerNorm(128 * n_qubits * n_qubits),
                nn.Linear(128 * n_qubits * n_qubits, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
            self.control_head = nn.Linear(256, n_qubits)
            self.target_head = nn.Linear(256, n_qubits)
            self.apply(init_weights)

        def forward(self, x):
            features = self.backbone(x)
            control_q = self.control_head(features)
            target_q = self.target_head(features)
            q_matrix = torch.einsum("bi,bj->bij", control_q, target_q)
            return q_matrix

    return CNNFactorizedQNet(n_qubits)

class DQNAgent:
    def __init__(self, n_qubits, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_model(n_qubits).to(self.device)
        self.target_model = _build_model(n_qubits).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=20000)
        self.archive = deque(maxlen=100)  # Top-performing episodes
        self.episode_buffer = []

        self.gamma = config["gamma"]
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.learning_rate = config["learning_rate"]
        self.gradient_clip_norm = config["gradient_clip_norm"]
        self.reward_clip = config.get("reward_clip", 20.0)
        self.final_reward = config["final_reward"]
        self.n_qubits = n_qubits
        self.k_explore = config.get("top_k", 5)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.episode_buffer.append((state, action, reward, next_state, done))

    def remember_episode(self, done: bool):
        if done:
            # Always save the current episode transitions to the full memory
            self.memory.extend(self.episode_buffer)

            try:
                # Try to get the final CX value from the last next_state
                cx_val = self.episode_buffer[-1][3][0][-1, 0, 0]  # (batch, channel, row, col)

                # Check if agent knows the true optimal CX for the environment
                if hasattr(self, 'true_optimal_cx') and self.true_optimal_cx is not None:
                    # Supervised fine-tuning: archive only if CX is close to optimal
                    if cx_val <= 1 * self.true_optimal_cx:
                        self.archive.append(list(self.episode_buffer))  # Save a copy
                else:
                    # Normal training (no true optimal known): archive best-performing episodes
                    if len(self.archive) < self.archive.maxlen or cx_val < max(
                        x[-1][3][0][-1, 0, 0] for x in self.archive
                    ):
                        self.archive.append(list(self.episode_buffer))  # Save a copy
            except Exception:
                # If anything fails (e.g., empty buffer), just skip archiving safely
                pass

        # Clear current episode buffer for next episode
        self.episode_buffer.clear()

    def act(self, state: Array3D, allowed_rows: list, allowed_cols: list, explore: bool = True) -> Tuple[int, int]:
        if explore and np.random.rand() <= self.epsilon:
            return random.choice(allowed_rows), random.choice(allowed_cols)

        with torch.no_grad():
            input_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.model(input_tensor)[0].cpu()

        mask = torch.full((self.n_qubits, self.n_qubits), float('-inf'))
        for r in allowed_rows:
            for c in allowed_cols:
                mask[r, c] = q_values[r, c]

        if explore:
            flat_values = mask.flatten()
            topk = min(self.k_explore, len(allowed_rows) * len(allowed_cols))
            topk_indices = torch.topk(flat_values, topk).indices
            topk_values = flat_values[topk_indices]
            probs = torch.softmax(topk_values, dim=0).numpy()
            chosen_idx = np.random.choice(topk_indices.numpy(), p=probs)
        else:
            chosen_idx = torch.argmax(mask).item()

        row_idx, col_idx = divmod(chosen_idx, self.n_qubits)
        return row_idx, col_idx

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        """Base model replay function."""
        """
        primary_batch = random.sample(self.memory, int(batch_size * 0.8))
        archive_batch = []
        if self.archive:
            for ep in random.sample(list(self.archive), min(len(self.archive), batch_size - len(primary_batch))):
                archive_batch.append(random.choice(ep))
        batch = primary_batch + archive_batch
        """
        """Sharp reward replay function for finetuning."""
        archive_ratio = 0.1
        archive_batch_size = int(batch_size * archive_ratio)
        memory_batch_size = batch_size - archive_batch_size

        memory_batch = random.sample(self.memory, min(len(self.memory), memory_batch_size))
        archive_batch = []

        if self.archive:
            archive_batch = [
                random.choice(ep) for ep in random.sample(list(self.archive), min(len(self.archive), archive_batch_size))
            ]
        batch = memory_batch + archive_batch

        states, actions, targets, rewards = [], [], [], []

        for s, a, r, s_next, done in batch:
            s_tensor = torch.from_numpy(s[0]).float().to(self.device)
            s_next_tensor = torch.from_numpy(s_next[0]).float().unsqueeze(0).to(self.device)
            rewards.append(r)

            with torch.no_grad():
                q_next = self.target_model(s_next_tensor)
                best_action = torch.argmax(q_next.view(-1)).item()
                max_q = q_next[0].view(-1)[best_action]

            y = r if done else r + self.gamma * max_q.item()
            y = np.clip(y, -self.reward_clip, self.reward_clip)

            states.append(s_tensor)
            actions.append(a)
            targets.append(y)

        states = torch.stack(states)
        actions = torch.tensor(actions).long().to(self.device)
        targets = torch.tensor(targets).float().to(self.device)
        rewards_batch = torch.tensor(rewards).float().to(self.device)

        q_pred = self.model(states)
        q_vals = q_pred[torch.arange(len(states)), actions[:, 0], actions[:, 1]]

        # loss = F.mse_loss(q_vals, targets) # Loss function for base model training with lower reward scheme
        
        # Loss function for base model training with sharp rewards
        losses = F.smooth_l1_loss(q_vals, targets, reduction='none')

        # whether you want to apply reward scaling
        USE_REWARD_SCALING = True 

        if USE_REWARD_SCALING:
            rewards_batch = torch.tensor(rewards).float().to(self.device)
            final_reward = self.final_reward
            alpha = 0.5

            reward_factor = (1 - alpha * (rewards_batch / final_reward))
            scaled_losses = losses * reward_factor
            final_loss = scaled_losses.mean()
        else:
            final_loss = losses.mean()

        self.optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        self.losses.append(final_loss.item())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)