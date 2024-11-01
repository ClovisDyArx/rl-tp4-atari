import numpy as np
from agent import Agent
from utils import make_env, plot_learning_curve

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

if __name__ == '__main__':
    env = make_env("PongNoFrameskip-v4")
    num_games = 10000
    load_checkpoint = False # LAISSEZ A TRUE POUR CHARGER LE MODELE
    best_score = -21
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0001,
                  input_dims=(4, 80, 80),
                  n_actions=6, mem_size=25000, epsilon_min=0.02,
                  batch_size=32, replace=1000, epsilon_dec=1e-5)

    if load_checkpoint:
        agent.load_models()

    fname = 'PongNoFrameskip-v4.png'

    scores, eps_history = [], []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation, _ = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action)
            n_steps += 1
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
            # else:
            #    env.render('human')
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[-100:])

        print(f"episode {i},"
              f"score: {score},"
              f"average score: {avg_score},"
              f"best score: {best_score},"
              f"epsilon: {agent.epsilon},"
              f"steps: {n_steps}"
              "\n\n")

        if avg_score > best_score:
            agent.save_models()
            print(f"New best score: {avg_score},"
                  f"better than score: {best_score}"
                  f"\n")
            best_score = avg_score

    # x = [i+1 for i in range(num_games)]
    # plot_learning_curve(x, scores, eps_history, fname)
