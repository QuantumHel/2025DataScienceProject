import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.rl.env import Array3D


def _build_model():
    return nn.Sequential(
        nn.Conv2d(3, 512, kernel_size=3),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 1, kernel_size=3),
    )


class DQNAgent:
    def __init__(self, n_qubits: int, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.997) -> None:
        self.model = _build_model()
        self.memory = deque(maxlen=20000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = 1e-4
        self.n_qubits = n_qubits
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        all_current_q_values = []
        all_target_q_values = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                with torch.no_grad():
                    target = target + self.gamma * torch.max(self.model(torch.from_numpy(next_state[0]).float()))

            target_q_values = torch.zeros(size=(1, self.n_qubits, self.n_qubits))
            target_q_values[0][action] = target

            current_q_values = torch.from_numpy(state[0]).float()

            all_target_q_values.append(target_q_values)
            all_current_q_values.append(current_q_values)
        target_q_values = torch.stack([item for item in all_target_q_values])
        current_q_values = self.model(torch.stack([item for item in all_current_q_values]))
        self.optimizer_step(current_q_values, target_q_values)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state: Tuple[Array3D, list, list],
                 action: Tuple[int, int],
                 reward: float,
                 next_state: Tuple[Array3D, list, list], done: bool):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: Array3D, allowed_rows: list, allowed_cols: list) -> Tuple[int, int]:
        if np.random.rand() <= self.epsilon:
            row = random.choice(allowed_rows)
            col = random.choice(allowed_cols)
            return row, col
        q_values = self.model(torch.from_numpy(state).float()).cpu().detach()[0]
        q_values = q_values[allowed_rows][:, allowed_cols]
        row_idx, col_idx = divmod(torch.argmax(q_values).item(), q_values.size(1))

        selected_row = allowed_rows[row_idx]
        selected_col = allowed_cols[col_idx]

        return selected_row, selected_col

    def optimizer_step(self, state_action_values, expected_state_action_values):
        criterion = nn.HuberLoss()
        loss = criterion(expected_state_action_values, state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping

        self.optimizer.step()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
