import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, num_vehicles, num_bs):
        super(Actor, self).__init__()
        self.num_vehicles = num_vehicles
        self.num_bs = num_bs

        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        # Output layer, output b, f and q respectively
        self.b_output = nn.Linear(128, num_vehicles * num_bs)
        self.f_output = nn.Linear(128, num_vehicles * num_bs)
        self.q_output = nn.Linear(128, num_vehicles * num_bs)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Use softmax activation function for b and f
        b_output = self.b_output(x).view(-1, self.num_bs, self.num_vehicles)
        f_output = self.f_output(x).view(-1, self.num_bs, self.num_vehicles)

        b_output = F.softmax(b_output, dim=-1)
        f_output = F.softmax(f_output, dim=-1)

        # Use sigmoid activation function for q
        q_output = torch.sigmoid(self.q_output(x).view(-1, self.num_bs, self.num_vehicles))

        # Merge Output
        action = torch.cat([q_output.view(-1, self.num_bs * self.num_vehicles),
                            b_output.view(-1, self.num_bs * self.num_vehicles),
                            f_output.view(-1, self.num_bs * self.num_vehicles)], dim=-1)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.q_out = nn.Linear(256, 3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.q_out(x)
        return q_values