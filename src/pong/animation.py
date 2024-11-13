import numpy as np
from agent import Agent
from utils import make_env, plot_learning_curve

import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
from matplotlib import animation

gym.register_envs(ale_py)


def animate_model(env, agent, num_episodes=1):
    frames = []

    for i in range(num_episodes):
        done = False
        observation, _ = env.reset()

        while not done:
            frames.append(env.render())
            action = agent.choose_action(observation)
            observation, reward, done, _, info = env.step(action)

    env.close()
    return frames


def save_animation(frames, filename="pong_animation.mp4"):
    fig = plt.figure()
    plt.axis('off')

    im = plt.imshow(frames[0])

    def update_frame(i):
        im.set_array(frames[i])
        return [im]

    anim = animation.FuncAnimation(fig, update_frame, frames=len(frames), interval=50, blit=True)
    anim.save(filename, writer='ffmpeg')


if __name__ == '__main__':
    env = make_env("PongNoFrameskip-v4")
    
    load_checkpoint = True
    agent = Agent(gamma=0.99, epsilon=0.0, alpha=0.0001,
                  input_dims=(4, 80, 80),
                  n_actions=6, mem_size=25000, epsilon_min=0.02,
                  batch_size=32, replace=1000, epsilon_dec=1e-5)

    if load_checkpoint:
        agent.load_models()

    frames = animate_model(env, agent, num_episodes=1)

    save_animation(frames, filename="pong_trained_model.mp4")
