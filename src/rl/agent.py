from typing import Tuple
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.rl.env import Array3D, get_optimal_cx_estimate

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
                nn.Conv2d(4, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                ResidualBlock(64),
                ResidualBlock(64),
                nn.AdaptiveAvgPool2d((n_qubits, n_qubits)),
                nn.Flatten(),
                nn.Linear(64 * n_qubits * n_qubits, 256),
                nn.ReLU(),
            )
            self.control_head = nn.Linear(256, n_qubits)
            self.target_head = nn.Linear(256, n_qubits)
            self.aux_head = nn.Linear(256, 1)

        def forward(self, x):
            features = self.backbone(x)
            control_q = self.control_head(features)
            target_q = self.target_head(features)
            q_matrix = torch.einsum("bi,bj->bij", control_q, target_q)
            optimality_score = torch.sigmoid(self.aux_head(features))
            return q_matrix, optimality_score

    return CNNFactorizedQNet(n_qubits)

class DQNAgent:
    def __init__(self, n_qubits, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_model(n_qubits).to(self.device)
        self.target_model = _build_model(n_qubits).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.memory = deque(maxlen=20000)
        self.n_step_buffer = deque(maxlen=3)

        self.gamma = config["gamma"]
        self.epsilon = 1.0
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.learning_rate = config["learning_rate"]
        self.gradient_clip_norm = config["gradient_clip_norm"]
        self.n_qubits = n_qubits
        self.aux_weight = config.get("aux_loss_weight", 0.25)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, final_cx=None):
        aux_label = 0.0
        if done and final_cx is not None:
            optimal = get_optimal_cx_estimate(self.n_qubits)
            margin = 8
            diff = final_cx - optimal
            if diff <= 0:
                aux_label = 1.0 + (abs(diff) / margin)
            else:
                aux_label = max(0.0, 1.0 - (diff / margin))

            aux_label = float(np.clip(aux_label, 0.0, 1.2))  # clamp just in case

        self.n_step_buffer.append((state, action, reward, next_state, done, aux_label))
        if len(self.n_step_buffer) == self.n_step_buffer.maxlen:
            R = sum([(self.gamma ** i) * self.n_step_buffer[i][2] for i in range(len(self.n_step_buffer))])
            s0, a0 = self.n_step_buffer[0][:2]
            s_next, d_last = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
            final_label = self.n_step_buffer[-1][5]
            self.memory.append((s0, a0, R, s_next, d_last, final_label))

    def act(self, state: Array3D, allowed_rows: list, allowed_cols: list) -> Tuple[int, int]:
        if np.random.rand() <= self.epsilon:
            return random.choice(allowed_rows), random.choice(allowed_cols)

        with torch.no_grad():
            input_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values, _ = self.model(input_tensor)
            q_values = q_values[0].cpu()

        allowed_q_values = q_values[allowed_rows][:, allowed_cols]
        best_idx = torch.argmax(allowed_q_values).item()
        row_idx, col_idx = divmod(best_idx, allowed_q_values.size(1))
        return allowed_rows[row_idx], allowed_cols[col_idx]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state_batch, action_batch, targets, labels = [], [], [], []

        for s, a, r, s_next, done, aux_label in minibatch:
            state_tensor = torch.from_numpy(s[0]).float().to(self.device)
            next_tensor = torch.from_numpy(s_next[0]).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_online, _ = self.model(next_tensor)
                q_target, _ = self.target_model(next_tensor)
                best_action = torch.argmax(q_online.view(-1)).item()
                best_row = best_action // self.n_qubits
                best_col = best_action % self.n_qubits
                max_q = q_target[0, best_row, best_col].item()

            y = r if done else r + self.gamma * max_q
            state_batch.append(state_tensor)
            action_batch.append(a)
            targets.append(y)
            labels.append(float(np.clip(aux_label, 0.0, 1.0)))

        state_batch = torch.stack(state_batch).to(self.device)
        targets = torch.tensor(targets).float().to(self.device)
        labels = torch.tensor(labels).float().to(self.device)
        action_indices = torch.tensor(action_batch).long().to(self.device)

        q_pred, opt_pred = self.model(state_batch)
        b_idx = torch.arange(len(minibatch))
        q_values = q_pred[b_idx, action_indices[:, 0], action_indices[:, 1]]

        loss_q = F.huber_loss(q_values, targets)
        loss_aux = F.binary_cross_entropy(opt_pred.squeeze(), labels)
        loss = loss_q + self.aux_weight * loss_aux

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        self.losses.append(loss.item())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay