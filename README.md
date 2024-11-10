# DQN Pong 🎮

[Pong Animation](https://github.com/ClovisDyArx/rl-tp4-pong/raw/main/pong_trained_model.mp4)  
*A trained Deep Q-Network (DQN) agent playing Atari Pong.*

## Project Overview

This project implements a DQN agent using PyTorch and Gymnasium to play the classic Atari game **Pong**.

The key components include:
- **DQN Model**: A neural network that estimates Q-values for action selection.
- **Replay Buffer**: A memory buffer that stores past experiences to stabilize training.
- **Target Network**: A second Q-network used to improve learning stability.