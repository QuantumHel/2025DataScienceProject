import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.rl.env import Array3D


class DuelingDQN(nn.Module):
    def __init__(self, input_channels=3, board_size=4, conv_filters=64):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, conv_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters)
        self.conv3 = nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_filters)
        self.feature_size = conv_filters * board_size * board_size
        self.fc_value = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, board_size * board_size)
        )
        self.board_size = board_size

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        value = self.fc_value(x)
        advantage = self.fc_advantage(x) 
        advantage = advantage.view(-1, self.board_size * self.board_size)
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_vals = q_vals.view(-1, self.board_size, self.board_size)
        return q_vals


class DQNAgent:
    def __init__(self, n_qubits: int, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.997) -> None:
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = 1e-4
        
        # Main network and target network
        self.model = DuelingDQN(input_channels=3, board_size=n_qubits)
        self.target_model = DuelingDQN(input_channels=3, board_size=n_qubits)
        self.update_target_network()  # initialize target network
        
        self.memory = deque(maxlen=20000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Counter and frequency for target network updates
        self.update_counter = 0
        self.target_update_freq = 1000  

    def update_target_network(self):
        # Hard update: copy model parameters to target model
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        all_current_q_values = []
        all_target_q_values = []
        
        for state, action, reward, next_state, done in minibatch:
            # Prepare state tensor; note: state[0] is assumed to be the array input
            state_tensor = torch.from_numpy(state[0]).float().unsqueeze(0)  # shape: (1, n_qubits, n_qubits)
            # Compute current Q-values from the model
            current_q_values = self.model(state_tensor)  # shape: (1, n_qubits, n_qubits)
            target_q_values = current_q_values.clone().detach()

            # Update target for the taken action using Double DQN approach:
            if not done:
                next_state_tensor = torch.from_numpy(next_state[0]).float().unsqueeze(0)
                with torch.no_grad():
                    # Get actions from the main network
                    main_q_next = self.model(next_state_tensor).view(1, -1)
                    best_action = main_q_next.argmax(dim=1, keepdim=True)
                    # Evaluate that action using the target network
                    target_q_next = self.target_model(next_state_tensor).view(1, -1)
                    next_q = target_q_next.gather(1, best_action).item()
                target = reward + self.gamma * next_q
            else:
                target = reward

            # Update the Q-value for the taken action.
            # The action is given as a tuple (row, col); update the corresponding index.
            target_q_values[0][action] = target
            
            # Remove batch dimension for stacking
            all_target_q_values.append(target_q_values.squeeze(0))
            all_current_q_values.append(current_q_values.squeeze(0))
        
        # Stack all transitions into a batch
        target_batch = torch.stack(all_target_q_values)
        current_batch = torch.stack(all_current_q_values)
        
        # Perform one optimization step
        self.optimizer_step(current_batch, target_batch)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update the target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()

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
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        q_values = self.model(state_tensor).cpu().detach()[0]
        # Filter Q-values based on allowed actions
        q_values = q_values[allowed_rows][:, allowed_cols]
        row_idx, col_idx = divmod(torch.argmax(q_values).item(), q_values.size(1))
        selected_row = allowed_rows[row_idx]
        selected_col = allowed_cols[col_idx]
        return selected_row, selected_col

    def optimizer_step(self, state_action_values, expected_state_action_values):
        criterion = nn.HuberLoss()
        loss = criterion(expected_state_action_values, state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()
