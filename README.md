# Resource-Allocation-for-IoV-based-on-MADDPG-Algorithm
Resource Allocation for Internet of Vehicles based on Multi-Agent Deep Deterministic Policy Gradient Algorithm

models:
This directory contains the core architecture for the neural networks used in the algorithm.
It includes:
actor network: Responsible for selecting actions based on the current state of the agent.
critic network: Evaluates the action taken by the actor by predicting the Q-value, which is used for training.

config:
This file stores all the hyperparameters and settings required for the training process.
It may include details like learning rate, discount factor, buffer size, batch size, and other configurable options

replay_buffer:
Defines the memory buffer that stores experiences from the agent's interaction with the environment.
Experiences are typically stored as tuples of state, action, reward, next state, and done flag, which are sampled during the training phase to update the models.

cost_functions:
Contains the formulas or functions for calculating the cost in the system.
This would be used to define penalties or rewards that help guide the training of the agents in the vehicular environment.

vehicular_env:
This file defines the environment.

test:
Contains the training loop or testing procedures to execute the MADDPG algorithm.
It manages the interaction between the agents and the environment, running episodes and updating the actor and critic networks through the learning process.

utils:
Holds utility functions that are used across the project.
Functions like soft update (gradually updating the target network) and hard update (completely replacing target networks at specific intervals) help maintain the stability of the learning process.
