import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import ale_py

gym.register_envs(ale_py)


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        truncated = None

        for i in range(self._skip):
            # Accumulate the reward and repeat the same action
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, truncated, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (80, 80, 1)
        self.observation_space = gym.spaces.Box(low=0.0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, obs):
        return PreprocessFrame.process(obs)

    @staticmethod
    def process(frame):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        grayscaled = 0.299 * new_frame[:, :, 0] + 0.587 * new_frame[:, :, 1] + 0.114 * new_frame[:, :, 2]
        grayscaled = grayscaled[35:195:2, ::2].reshape(80, 80, 1)
        return grayscaled.astype(np.uint8)


class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(MoveImgChannel, self).__init__(env)
        self.shape = (env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(n_steps, axis=0),
                                                env.observation_space.high.repeat(n_steps, axis=0),
                                                dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreprocessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)


def plot_learning_curve(x, scores, eps_history, fname):
    if len(eps_history) == 0:
        raise ValueError("eps_history is empty. Ensure it is populated with data before plotting.")

    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, eps_history, color='C0')
    ax.set_xlabel('Game', color='C0')
    ax.set_ylabel('Epsilon', color='C0')
    ax.tick_params(axis='x', colors='C0')
    ax.tick_params(axis='y', colors='C0')

    N = len(scores)
    run_avg = np.empty(N)
    for t in range(N):
        run_avg[t] = np.mean(scores[max(0, t - 10):(t + 1)])

    ax2.scatter(x, run_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')

    plt.savefig(fname)
