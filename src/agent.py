from replay_buffer import ReplayBuffer
from model_dqn import build_dqn

import numpy as np
import torch
from torch.optim import Adam
from torch.nn import MSELoss


class Agent(object):  # agent class.
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, replace, input_dims,
                 epsilon_dec=1e-3, epsilon_min=0.01, mem_size=1000000,
                 eval_fname='dqn_eval.pth', target_fname='dqn_target.pth'):
        self.action_space = [i for i in range(n_actions)]
        self.alpha = alpha  # Learning rate.
        self.gamma = gamma  # Discount factor.
        self.epsilon = epsilon  # Exploration rate.
        self.eps_dec = epsilon_dec  # Exploration rate decay.
        self.eps_min = epsilon_min  # Minimum exploration rate.
        self.batch_size = batch_size  # Batch size for experience replay.
        self.replace = replace  # How often to replace the target network.
        self.eval_model_file = eval_fname  # File name for the evaluation network.
        self.target_model_file = target_fname  # File name for the target network.
        self.learn_step_counter = 0  # Keeping track of how many steps the agent has taken.
        self.memory = ReplayBuffer(mem_size, input_dims)  # Replay buffer for experience replay.

        self.q_eval = build_dqn(n_actions, input_dims, 512)  # Evaluation network.
        self.q_eval.optimizer = Adam(self.q_eval.parameters(), lr=alpha)  # Optimizer for the evaluation network.
        self.q_eval.loss = MSELoss()  # Loss function for the evaluation network.

        self.q_next = build_dqn(n_actions, input_dims, 512)  # Target network.
        self.q_next.optimizer = Adam(self.q_next.parameters(), lr=alpha)  # Optimizer for the target network.
        self.q_next.loss = MSELoss()  # Loss function for the target network.

    def replace_target_network(self):  # Replacing the target network with the evaluation network.
        if self.replace != 0 and self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def store_transition(self, state, action, reward, new_state, done):  # Storing the experience in the replay buffer.
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):  # Choosing an action based on the epsilon-greedy policy.
        if np.random.random() > self.epsilon:  # Greedy action, exploit.
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:  # Random action, explore.
            action = np.random.choice(self.action_space)
        return action

    def learn(self):  # Learning from the experiences in the replay buffer.
        if self.memory.mem_cntr < self.batch_size:  # If there are not enough experiences in the replay buffer, return.
            state, action, reward, new_state, done = self.memory.sample_buffer(self.memory.mem_cntr)
            self.replace_target_network()

            q_eval = self.q_eval.forward(torch.tensor(state).to(self.q_eval.device))
            q_next = self.q_next.forward(torch.tensor(new_state).to(self.q_next.device))
            q_next[done] = 0.0

            indices = np.arange(self.batch_size)
            q_target = q_eval.clone()
            q_target[indices, action] = reward + self.gamma * torch.max(q_next, dim=1)[0]

            self.q_eval.optimizer.zero_grad()
            loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
            loss.backward()
            self.q_eval.optimizer.step()

            # FIXME: q_next network update ?

            self.learn_step_counter += 1

            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save(self.eval_model_file)
        self.q_next.save(self.target_model_file)

    def load_models(self):
        self.q_eval.load(self.eval_model_file)
        self.q_next.load(self.target_model_file)
