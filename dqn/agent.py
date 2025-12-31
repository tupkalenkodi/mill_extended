import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn.network import DQNNetwork
from dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, player_id, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.player_id = player_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQNNetwork().to(self.device)
        self.target_net = DQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 64

    def action_to_index(self, action):
        """Convert action [from, to, capture] to single index"""
        return action[0] * 625 + action[1] * 25 + action[2]

    def index_to_action(self, index):
        """Convert index back to action [from, to, capture]"""
        from_pos = index // 625
        remainder = index % 625
        to_pos = remainder // 25
        capture = remainder % 25
        return [from_pos, to_pos, capture]

    def choose_move(self, state, legal_moves):
        """Choose action using epsilon-greedy policy with legal move masking"""

        # Exploration
        if np.random.random() < self.epsilon:
            return legal_moves[np.random.choice(len(legal_moves))]

        # Exploitation with legal move masking
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

        # Mask illegal moves
        legal_indices = [self.action_to_index(move) for move in legal_moves]
        masked_q_values = np.full_like(q_values, -np.inf)
        masked_q_values[legal_indices] = q_values[legal_indices]

        # Choose best legal action
        best_index = np.argmax(masked_q_values)
        return self.index_to_action(best_index)

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        action_index = self.action_to_index(action)
        self.memory.push(state, action_index, reward, next_state, done)

    def train(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save the model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']