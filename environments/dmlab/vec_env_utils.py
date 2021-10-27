"""
Helpers for constructing vector environments.
"""
import os
os.environ.setdefault('PATH', '')

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym

from environments.atari.vec_env_utils import Monitor, ClipRewardEnv
from environments.dmlab.gym_env import LabEnv
from vec_env import DummyVecEnv, ShmemVecEnv


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = -1

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.env.rng.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


def make_env(env_id, cache, noop_max, rank, seed, env_kwargs):
    env = LabEnv(
        level=env_id,
        cache=cache,
        config=env_kwargs,
        seed=seed*10000+rank,
    )
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = Monitor(env, allow_early_resets=True)
    env = ClipRewardEnv(env)
    return env


def make_vec_env(env_id, cache, noop_max, num_env, seed, env_kwargs=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            cache=cache,
            noop_max=noop_max,
            rank=rank,
            seed=seed,
            env_kwargs=env_kwargs,
        )

    if num_env > 1:
        return ShmemVecEnv([make_thunk(i) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i) for i in range(num_env)])


if __name__ == '__main__':
    env_id = 'contributed/dmlab30/lasertag_three_opponents_small'
    env_kwargs = {
        'width': 96,
        'height': 72,
    }
    env = make_vec_env(env_id, 'local', noop_max=30, num_env=2, seed=0, env_kwargs=env_kwargs)
    print(env.num_envs)
    print(env.observation_space)
    print(env.action_space)
    ob = env.reset()
    print(ob.shape)
