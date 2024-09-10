import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = []  # Priority queue
#        self.state_dim = state_dim
#        self.action_dim = action_dim

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        # print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")

    def sample(self, batch_size, alpha=0.8):
        priorities = np.array(self.priorities)
        probabilities = priorities ** alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = map(lambda x: torch.stack(x), zip(*batch))
        return state, action, reward, next_state, done, indices  # Return Index

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)