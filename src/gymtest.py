import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

gym.make("ALE/Pong-v5")
