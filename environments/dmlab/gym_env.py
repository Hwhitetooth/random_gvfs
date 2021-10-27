import os
import shutil

import deepmind_lab as lab
import gym
import numpy as np


class LocalLevelCache(object):
    def __init__(self, cache_dir='/shared/home/zeyu/dmlab_level_cache'):
        self._cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

    def fetch(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)
        if os.path.isfile(path):
            # Copy the cached file to the path expected by DeepMind Lab.
            shutil.copyfile(path, pk3_path)
            return True
        return False

    def write(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)
        if not os.path.isfile(path):
            try:
                os.mknod(path)
            except FileExistsError:
                pass
            # Copy the cached file DeepMind Lab has written to the cache directory.
            shutil.copyfile(pk3_path, path)


def make_lab_game(level, cache, config=None):
    if cache == 'local':
        cache = LocalLevelCache()
    _config = {
        'width': 96,
        'height': 72,
        'logLevel': 'WARN',
    }
    _config.update(config or {})
    config = {k: str(v) for k, v in _config.items()}

    game = lab.Lab(
        level=level,
        observations=['RGB_INTERLEAVED'],
        config=config,
        renderer='hardware',
        level_cache=cache,
    )

    return game


DEFAULT_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),  # Forward
    (0, 0, 0, -1, 0, 0, 0),  # Backward
    (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),  # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),  # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),  # Fire.
)


class LabEnv(gym.Env):
    def __init__(self, level, cache, config=None, seed=None, action_repeat=4):
        self._game = make_lab_game(level, cache, config)
        self._seed = seed
        self._rng = np.random.RandomState(seed=seed)
        self._action_repeat = action_repeat

        observation_spec = self._game.observation_spec()
        for ob_spec in observation_spec:
            if ob_spec['name'] == 'RGB_INTERLEAVED':
                ob_shape = ob_spec['shape']
                break
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=ob_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(len(DEFAULT_ACTION_SET))

        self._observation_tm1 = None

    def reset(self):
        seed = self._rng.randint(0, 10 ** 3)
        self._game.reset(seed=seed)
        return self._observation()

    def step(self, action):
        if action == -1:
            # Noop.
            raw_action = (0, 0, 0, 0, 0, 0, 0)
        else:
            raw_action = DEFAULT_ACTION_SET[action]
        reward = self._game.step(np.array(raw_action, dtype=np.intc), num_steps=self._action_repeat)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(not self._game.is_running())
        return self._observation(), reward, done, {}

    def close(self):
        self._game.close()

    def _observation(self):
        # NOTE: It seems that DM Lab does not provide access to the end-of-episode observation. This is a hack to keep
        #   it consistent with our algorithm implementation. If it is the end of the episode, we copy the previous
        #   observation as the observation at this final time step.
        if self._game.is_running():
            observations = self._game.observations()
            observation = observations['RGB_INTERLEAVED']
            self._observation_tm1 = observation
        else:
            observation = self._observation_tm1
        return observation

    @property
    def rng(self):
        return self._rng


if __name__ == '__main__':
    level = 'seekavoid_arena_01'
    env = LabEnv(level, seed=42)

    observation = env.reset()
    print(observation.shape, observation.dtype)
    done = False
    ret = 0
    t = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        ret += reward
        t += 1
    print(t, ret)
