import collections

import jax
import numpy as onp

from algorithms import utils


ActorOutput = collections.namedtuple(
    'ActorOutput', [
        'rnn_state',
        'action_tm1',
        'reward',
        'discount',
        'first',
        'observation',
    ]
)


class Actor(object):
    def __init__(self, envs, agent, nsteps):
        self._envs = envs
        self._agent = agent
        self._nsteps = nsteps
        nenvs = self._envs.num_envs
        self._state = agent.initial_state(nenvs)
        self._timestep = ActorOutput(
            rnn_state=jax.device_get(self._state),
            action_tm1=onp.zeros((nenvs,), dtype=onp.int32),  # dummy actions
            reward=onp.zeros((nenvs,), dtype=onp.float32),  # dummy reward
            discount=onp.ones((nenvs,), dtype=onp.float32),  # dummy discount
            first=onp.ones((nenvs,), dtype=onp.float32),  # dummy first
            observation=self._envs.reset(),
        )

    def rollout(self, rngkey, params):
        state = self._state
        timestep = self._timestep
        timesteps = [timestep]
        epinfos = []
        for t in range(self._nsteps):
            timestep = jax.device_put(timestep)
            rngkey, action, agent_output, state = self._agent.step(rngkey, params, timestep, state)
            action = jax.device_get(action)  # This is crucial for a higher throughput!!!
            observation, reward, terminate, info = self._envs.step(action)
            timestep = ActorOutput(
                rnn_state=jax.device_get(state),
                action_tm1=action,
                reward=reward.astype(onp.float32),
                discount=1.-terminate.astype(onp.float32),
                first=1.-jax.device_get(timestep.discount),
                observation=observation,
            )
            timesteps.append(timestep)
            for i in info:
                maybeepinfo = i.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
        self._state = state
        self._timestep = timestep
        return rngkey, utils.pack_namedtuple_onp(timesteps, axis=1), epinfos
