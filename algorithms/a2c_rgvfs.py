import time
import collections
import copy
import logging

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp
import optax
import ray
from ray import tune
import rlax
import tree

from algorithms import utils
from algorithms.actor import Actor, ActorOutput
from algorithms.haiku_nets import ConvTorso
from algorithms.random_features import get_random_feature_fn
from algorithms.td_network import random_td_network

AgentOutput = collections.namedtuple(
    'AgentOutput', (
        'state',
        'logits',
        'value',
        'aux_pred',
    )
)

A2CLog = collections.namedtuple(
    'A2CLog', (
        'entropy',
        'value',
        'ret',
        'pg_loss',
        'baseline_loss',
    )
)

AuxLog = collections.namedtuple(
    'AuxLog', (
        'aux_loss',
    )
)


class ActorCriticAuxNet(hk.RNNCore):
    def __init__(self, num_actions, num_pred, torso_kwargs, use_rnn, head_layers, stop_ac_grad, scale, name=None):

        super(ActorCriticAuxNet, self).__init__(name=name)
        self._num_actions = num_actions
        self._num_pred = num_pred
        self._torso_kwargs = torso_kwargs
        self._use_rnn = use_rnn
        if use_rnn:
            core = hk.GRU(512, w_h_init=hk.initializers.Orthogonal())
        else:
            core = hk.IdentityCore()
        self._core = hk.ResetCore(core)
        self._head_layers = head_layers
        self._stop_ac_grad = stop_ac_grad
        self._scale = scale

    def __call__(self, timesteps, state):
        torso_net = ConvTorso(**self._torso_kwargs)
        torso_output = torso_net(timesteps.observation)

        if self._use_rnn:
            core_input = jnp.concatenate([
                hk.one_hot(timesteps.action_tm1, self._num_actions),
                timesteps.reward[:, None],
                torso_output
            ], axis=1)
            should_reset = timesteps.first
            core_output, next_state = hk.dynamic_unroll(self._core, (core_input, should_reset), state)
        else:
            core_output, next_state = torso_output, state

        aux_head = []
        for dim in self._head_layers:
            aux_head.append(hk.Linear(dim))
            aux_head.append(jax.nn.relu)
        aux_input = hk.Sequential(aux_head)(core_output)
        aux_input = self._scale * aux_input + lax.stop_gradient((1 - self._scale) * aux_input)
        aux_pred = hk.Linear(self._num_pred)(aux_input)

        main_head = []
        if self._stop_ac_grad:
            main_head.append(lax.stop_gradient)
        for dim in self._head_layers:
            main_head.append(hk.Linear(dim))
            main_head.append(jax.nn.relu)
        h = hk.Sequential(main_head)(core_output)
        logits = hk.Linear(self._num_actions)(h)
        value = hk.Linear(1)(h)

        agent_output = AgentOutput(
            state=core_output,
            logits=logits,
            value=value.squeeze(-1),
            aux_pred=aux_pred,
        )
        return agent_output, next_state

    def initial_state(self, batch_size):
        return self._core.initial_state(batch_size)


class Agent(object):
    def __init__(self, ob_space, action_space, num_pred, torso_kwargs, head_layers, use_rnn, stop_ac_grad, scale):
        self._ob_space = ob_space
        num_actions = action_space.n
        _, self._initial_state_apply_fn = hk.without_apply_rng(
            hk.transform(lambda batch_size: ActorCriticAuxNet(
                num_actions=num_actions,
                num_pred=num_pred,
                torso_kwargs=torso_kwargs,
                use_rnn=use_rnn,
                head_layers=head_layers,
                stop_ac_grad=stop_ac_grad,
                scale=scale,
            ).initial_state(batch_size))
        )
        self._init_fn, self._apply_fn = hk.without_apply_rng(
            hk.transform(lambda inputs, state: ActorCriticAuxNet(
                num_actions=num_actions,
                num_pred=num_pred,
                torso_kwargs=torso_kwargs,
                use_rnn=use_rnn,
                head_layers=head_layers,
                stop_ac_grad=stop_ac_grad,
                scale=scale,
            )(inputs, state))
        )
        self.step = jax.jit(self._step)

    def init(self, rngkey):
        dummy_observation = tree.map_structure(lambda t: jnp.zeros(t.shape, t.dtype), self._ob_space)
        dummy_observation = tree.map_structure(lambda t: t[None], dummy_observation)
        dummy_reward = jnp.zeros((1,), dtype=jnp.float32)
        dummy_action = jnp.zeros((1,), dtype=jnp.int32)
        dummy_discount = jnp.zeros((1,), dtype=jnp.float32)
        dummy_first = jnp.zeros((1,), dtype=jnp.float32)
        dummy_state = self.initial_state(None)
        dummy_input = ActorOutput(
            rnn_state=dummy_state,
            action_tm1=dummy_action,
            reward=dummy_reward,
            discount=dummy_discount,
            first=dummy_first,
            observation=dummy_observation,
        )
        return self._init_fn(rngkey, dummy_input, dummy_state)

    def initial_state(self, batch_size):
        return self._initial_state_apply_fn(None, batch_size)

    def _step(self, rngkey, params, timesteps, states):
        rngkey, subkey = jax.random.split(rngkey)
        timesteps = tree.map_structure(lambda t: t[:, None, ...], timesteps)  # [B, 1, ...]
        agent_output, next_states = jax.vmap(self._apply_fn, (None, 0, 0))(params, timesteps, states)
        agent_output = tree.map_structure(lambda t: t.squeeze(axis=1), agent_output)  # [B, ...]
        action = hk.multinomial(subkey, agent_output.logits, num_samples=1).squeeze(axis=-1)
        return rngkey, action, agent_output, next_states

    def unroll(self, params, timesteps, state):
        return self._apply_fn(params, timesteps, state)  # [T, ...]


def gen_a2c_update_fn(agent, opt_update, gamma, vf_coef, entropy_reg):
    def a2c_loss(theta, trajs):
        rnn_states = tree.map_structure(lambda t: t[:, 0], trajs.rnn_state)
        learner_output, _ = jax.vmap(agent.unroll, (None, 0, 0))(theta, trajs, rnn_states)  # [B, T + 1, ...]
        rewards = trajs.reward[:, 1:]
        discounts = trajs.discount[:, 1:] * gamma
        bootstrap_value = learner_output.value[:, -1]
        returns = jax.vmap(rlax.discounted_returns)(rewards, discounts, bootstrap_value)
        advantages = returns - learner_output.value[:, :-1]

        masks = trajs.discount[:, :-1]
        pg_loss = jax.vmap(rlax.policy_gradient_loss)(
            learner_output.logits[:, :-1], trajs.action_tm1[:, 1:], advantages, masks)
        ent_loss = jax.vmap(rlax.entropy_loss)(learner_output.logits[:, :-1], masks)
        baseline_loss = 0.5 * jnp.mean(
            jnp.square(learner_output.value[:, :-1] - lax.stop_gradient(returns)) * masks, axis=1)
        loss = jnp.mean(pg_loss + vf_coef * baseline_loss + entropy_reg * ent_loss)

        a2c_log = A2CLog(
            entropy=-ent_loss,
            value=learner_output.value,
            ret=returns,
            pg_loss=pg_loss,
            baseline_loss=baseline_loss,
        )
        return loss, a2c_log

    def a2c_update(theta, opt_state, trajs):
        grads, logs = jax.grad(a2c_loss, has_aux=True)(theta, trajs)
        updates, new_opt_state = opt_update(grads, opt_state)
        new_theta = optax.apply_updates(theta, updates)
        return new_theta, new_opt_state, logs

    return a2c_update


def gen_td_net_update_fn(agent, opt_update, td_mat, td_masks, feature_fn):
    def compute_td_target(pred_tp1):
        return jnp.matmul(td_mat, pred_tp1)

    def td_net_loss(theta, feature_params, trajs):
        rnn_states = tree.map_structure(lambda t: t[:, 0], trajs.rnn_state)
        agent_output, _ = jax.vmap(agent.unroll, (None, 0, 0))(theta, trajs, rnn_states)  # [B, T, ...]
        pred = agent_output.aux_pred[:, :-1]
        pred_tp1 = agent_output.aux_pred[:, 1:] * trajs.discount[:, 1:, None]

        target_feature = jax.vmap(feature_fn, (None, 0))(feature_params, trajs)
        pred_masks = td_masks[trajs.action_tm1[:, 1:]]
        transition_masks = trajs.discount[:, :-1]

        feature_and_pred_tp1 = jnp.concatenate([target_feature, pred_tp1], axis=-1)
        _td_target = jax.vmap(jax.vmap(compute_td_target, 0), 0)(feature_and_pred_tp1)
        td_target = _td_target[..., target_feature.shape[-1]:]

        # Flatten the tensors: [B, T, ...] -> [B * T, ...]
        pred, td_target, pred_masks, transition_masks = tree.map_structure(
            lambda t: t.reshape((t.shape[0] * t.shape[1],) + t.shape[2:]),
            (pred, td_target, pred_masks, transition_masks)
        )

        pred_losses = 0.5 * jnp.square(pred - lax.stop_gradient(td_target)) * pred_masks
        aux_loss = jnp.mean(jnp.sum(pred_losses, axis=-1) * transition_masks)

        aux_log = AuxLog(
            aux_loss=aux_loss,
        )
        return aux_loss, aux_log

    def td_net_update(theta, feature_params, opt_state, trajs):
        grads, logs = jax.grad(td_net_loss, has_aux=True)(theta, feature_params, trajs)
        updates, new_opt_state = opt_update(grads, opt_state)
        new_theta = optax.apply_updates(theta, updates)
        return new_theta, new_opt_state, logs

    return td_net_update


class Experiment(tune.Trainable):
    def setup(self, config):
        self._config = config
        platform = jax.lib.xla_bridge.get_backend().platform
        logging.warning("Running on %s", platform)

        if config['stop_ac_grad']:
            logging.warning("Stop gradients from the actor-critic loss!")

        # Environment setup.
        if config['env_id'].startswith('dmlab/'):
            import environments.dmlab.vec_env_utils as dmlab_vec_env
            env_id = config['env_id'][6:]
            gpu_id = ray.get_gpu_ids()[0]
            env_kwargs = copy.deepcopy(config['env_kwargs'])
            env_kwargs['gpuDeviceIndex'] = gpu_id
            self._envs = dmlab_vec_env.make_vec_env(
                env_id, config['cache'], config['noop_max'], config['nenvs'], config['seed'], env_kwargs)
            self._frame_skip = 4
        elif config['env_id'][-14:] == 'NoFrameskip-v4':
            import environments.atari.vec_env_utils as atari_vec_env
            from vec_env import VecFrameStack
            envs = atari_vec_env.make_vec_env(
                config['env_id'],
                config['nenvs'],
                config['seed'],
            )
            if config['use_rnn']:
                self._envs = envs
            else:
                self._envs = VecFrameStack(envs, 4)
            self._frame_skip = 4
        else:
            raise KeyError
        self._nsteps = config['nsteps']

        # Random features.
        jax_seed = onp.random.randint(2 ** 31 - 1)
        self._rngkey = jax.random.PRNGKey(jax_seed)
        self._rngkey, subkey = jax.random.split(self._rngkey)
        feature_fn, self._feature_params, num_targets = get_random_feature_fn(
            rngkey=subkey,
            observation_space=self._envs.observation_space,
            **config['random_feature_kwargs'],
        )

        # Random TD network.
        num_actions = self._envs.action_space.n
        num_pred, td_mat, td_masks, self._dep = random_td_network(
            num_actions=num_actions, num_targets=num_targets, **config['td_net_kwargs'])
        self._depth = self._dep.max() + 1
        print('{} features, {} predictions in total.'.format(num_targets, num_pred))

        active_predictions = len(config['td_net_kwargs']['discount_factors']) * num_targets + \
                             config['td_net_kwargs']['depth'] * config['td_net_kwargs']['repeat']
        scale = 1. / onp.sqrt(active_predictions)

        agent = Agent(
            ob_space=self._envs.observation_space,
            action_space=self._envs.action_space,
            num_pred=num_pred,
            torso_kwargs=config['torso_kwargs'],
            use_rnn=config['use_rnn'],
            head_layers=config['head_layers'],
            stop_ac_grad=config['stop_ac_grad'],
            scale=scale,
        )
        self._actor = Actor(self._envs, agent, self._nsteps)

        a2c_opt = optax.rmsprop(**config['a2c_opt_kwargs'])
        if config['max_a2c_grad_norm'] > 0:
            a2c_opt = optax.chain(
                optax.clip_by_global_norm(config['max_a2c_grad_norm']),
                a2c_opt,
            )
        a2c_opt_init, a2c_opt_update = a2c_opt
        aux_opt = optax.adam(**config['aux_opt_kwargs'])
        if config['max_aux_grad_norm'] > 0:
            aux_opt = optax.chain(
                optax.clip_by_global_norm(config['max_aux_grad_norm']),
                aux_opt,
            )
        aux_opt_init, aux_opt_update = aux_opt

        a2c_update_fn = gen_a2c_update_fn(
            agent=agent,
            opt_update=a2c_opt_update,
            gamma=config['gamma'],
            vf_coef=config['vf_coef'],
            entropy_reg=config['entropy_reg'],
        )

        aux_update_fn = gen_td_net_update_fn(
            agent=agent,
            opt_update=aux_opt_update,
            td_mat=jax.device_put(td_mat),
            td_masks=jax.device_put(td_masks),
            feature_fn=feature_fn,
        )
        self._a2c_update_fn = jax.jit(a2c_update_fn)
        self._aux_update_fn = jax.jit(aux_update_fn)

        self._rngkey, subkey = jax.random.split(self._rngkey)
        self._theta = agent.init(subkey)
        self._a2c_opt_state = a2c_opt_init(self._theta)
        self._aux_opt_state = aux_opt_init(self._theta)

        self._epinfo_buf = collections.deque(maxlen=100)
        self._num_iter = 0
        self._num_frames = 0
        self._tstart = time.time()

    def step(self):
        t0 = time.time()
        rngkey = self._rngkey
        theta = self._theta
        num_frames_this_iter = 0
        for _ in range(self._config['log_interval']):
            rngkey, trajs, epinfos = self._actor.rollout(rngkey, theta)
            self._epinfo_buf.extend(epinfos)

            trajs = jax.device_put(trajs)
            theta, self._a2c_opt_state, a2c_log = self._a2c_update_fn(
                theta, self._a2c_opt_state, trajs)
            theta, self._aux_opt_state, aux_log = self._aux_update_fn(
                theta, self._feature_params, self._aux_opt_state, trajs)

            self._num_iter += 1
            num_frames_this_iter += self._config['nenvs'] * self._nsteps * self._frame_skip
        self._rngkey = rngkey
        self._theta = theta
        self._num_frames += num_frames_this_iter

        a2c_log = jax.device_get(a2c_log)
        aux_log = jax.device_get(aux_log)
        ev = utils.explained_variance(a2c_log.value[:, :-1].flatten(), a2c_log.ret.flatten())
        log = {
            'label': self._config['label'],
            'episode_return': onp.mean([epinfo['r'] for epinfo in self._epinfo_buf]),
            'episode_length': onp.mean([epinfo['l'] for epinfo in self._epinfo_buf]),
            'entropy': a2c_log.entropy.mean(),
            'explained_variance': ev,
            'pg_loss': a2c_log.pg_loss.mean(),
            'baseline_loss': a2c_log.baseline_loss.mean(),
            'aux_loss': aux_log.aux_loss.mean(),
            'num_iterations': self._num_iter,
            'num_frames': self._num_frames,
            'fps': num_frames_this_iter / (time.time() - t0),
        }
        return log


if __name__ == '__main__':
    config = {
        'label': 'rgvfs',
        'env_id': 'BreakoutNoFrameskip-v4',
        'env_kwargs': {},

        'torso_kwargs': {
            'dense_layers': (),
        },
        'use_rnn': False,
        'head_layers': (512,),
        'stop_ac_grad': True,

        'nenvs': 16,
        'nsteps': 20,
        'gamma': 0.99,
        'lambda_': 1.0,
        'vf_coef': 0.5,
        'entropy_reg': 0.01,

        'a2c_opt_kwargs': {
            'learning_rate': 7E-4,
            'decay': 0.99,
            'eps': 1E-5,
        },
        'max_a2c_grad_norm': 0.5,

        'aux_opt_kwargs': {
            'learning_rate': 7E-4,
            'b1': 0.,
            'b2': 0.99,
            'eps_root': 1E-5,
        },
        'max_aux_grad_norm': 0.,

        'td_net_type': 'mixed_open_loop_planning',
        'td_net_kwargs': {
            'seed': None,
            'depth': 8,
            'repeat': 16,
            'discount_factors': (0.95,),
        },

        'random_feature_kwargs': {
            'conv_layers': ((1, 21, 21),),
            'dense_layers': (),
            'padding': 'VALID',
            'w_init': 'orthogonal',
            'w_init_scale': 8.,
            'delta': True,
            'absolute': True,
            'only_last_channel': True,
        },

        'log_interval': 100,
        'seed': 42,
    }
    analysis = tune.run(
        Experiment,
        name='debug',
        config=config,
        stop={
            'num_frames': 200 * 10 ** 6,
        },
        resources_per_trial={
            'gpu': 1,
        },
    )
