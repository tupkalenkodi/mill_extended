import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, state_size=24, action_size=25 * 25 * 25):
        super(DQNNetwork, self).__init__()

        # Simple feedforward network
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)  # Q-values for all actions
    