from replay_buffer import ReplayBuffer
from model_dqn import build_dqn

import numpy as np
import torch
from torch.optim import Adam
from torch.nn import MSELoss


class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, replace, input_dims,
                 epsilon_dec=1e-3, epsilon_min=0.01, mem_size=1000000,
                 eval_fname='dqn_eval.pth', target_fname='dqn_target.pth'):
        self.action_space = [i for i in range(n_actions)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_min
        self.batch_size = batch_size
        self.replace = replace
        self.eval_model_file = eval_fname
        self.target_model_file = target_fname
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)

        # Set device to GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models and move them to the device
        self.q_eval = build_dqn(n_actions, input_dims, 512).to(self.device)
        self.q_eval.optimizer = Adam(self.q_eval.parameters(), lr=alpha)
        self.q_eval.loss = MSELoss()

        self.q_next = build_dqn(n_actions, input_dims, 512).to(self.device)
        self.q_next.optimizer = Adam(self.q_next.parameters(), lr=alpha)
        self.q_next.loss = MSELoss()

    def replace_target_network(self):
        if self.replace != 0 and self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(np.array([observation]), dtype=torch.float).to(self.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        self.replace_target_network()

        # Convert states and actions to tensors and move to the device
        state = torch.tensor(state).to(self.device)
        new_state = torch.tensor(new_state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)

        # Forward pass and calculation of target Q-values
        q_eval = self.q_eval.forward(state)
        q_next = self.q_next.forward(new_state)
        q_next[done] = 0.0

        indices = np.arange(self.batch_size)
        q_target = q_eval.clone()
        q_target[indices, action] = reward + self.gamma * torch.max(q_next, dim=1)[0]

        self.q_eval.optimizer.zero_grad()
        loss = self.q_eval.loss(q_target, q_eval).to(self.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

    def save_models(self):
        print("Saving models...")
        torch.save(self.q_eval.state_dict(), self.eval_model_file)
        torch.save(self.q_next.state_dict(), self.target_model_file)
        print("Models saved successfully.")

    def load_models(self):
        print("Loading models...")
        self.q_eval.load_state_dict(torch.load(self.eval_model_file, map_location=self.device, weights_only=True))
        self.q_next.load_state_dict(torch.load(self.target_model_file, map_location=self.device, weights_only=True))
        print("Models loaded successfully.")
