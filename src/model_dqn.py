import torch
import torch.nn as nn


# Define the DQN model
def build_dqn(n_actions, input_dims, fc1_dims):
    model = nn.Sequential(
        nn.Conv2d(in_channels=input_dims[0], out_channels=32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, fc1_dims),  # 3136 = 64 * 7 * 7
        nn.ReLU(),
        nn.Linear(fc1_dims, n_actions)
    )

    return model
