import gym
import numpy as np


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            # Accumulate the reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


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
        self.shape = (self.shape[-1], self.shape[0], self.shape[1])
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

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

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
