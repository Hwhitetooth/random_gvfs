import haiku as hk
import jax
import jax.numpy as jnp
import tree


class ConvNet(hk.Module):
    def __init__(self, conv_layers=(), dense_layers=(), w_init=None, w_init_scale=1., padding='VALID', name=None,
                 **unused_kwargs):
        super(ConvNet, self).__init__(name=name)
        self._conv_layers = conv_layers
        self._dense_layers = dense_layers
        if w_init == 'orthogonal':
            w_init = hk.initializers.Orthogonal(scale=w_init_scale)
        elif w_init in ['fan_in', 'fan_out', 'fan_avg']:
            w_init = hk.initializers.VarianceScaling(scale=w_init_scale, mode=w_init)
        self._w_init = w_init
        self._padding = padding

    def __call__(self, x):
        net = [lambda x: x / 255.]
        for i, (ch, k, s) in enumerate(self._conv_layers):
            net.append(hk.Conv2D(ch, kernel_shape=k, stride=s, w_init=self._w_init, padding=self._padding))
            if i < len(self._conv_layers) - 1:
                net.append(jax.nn.relu)
        net.append(hk.Flatten())
        for dim in self._dense_layers:
            net.append(jax.nn.relu)
            net.append(hk.Linear(dim, w_init=self._w_init))
        return hk.Sequential(net)(x)


class RandomConvNet(object):
    def __init__(self, ob_space, only_last_channel=False, **kwargs):
        self._ob_space = ob_space
        self._only_last_channel = only_last_channel
        self._init_fn, self._apply_fn = hk.without_apply_rng(hk.transform(lambda x: ConvNet(**kwargs)(x)))

    def init(self, rngkey):
        dummy_observation = tree.map_structure(lambda t: jnp.zeros(t.shape, t.dtype), self._ob_space)
        dummy_observation = tree.map_structure(lambda t: t[None], dummy_observation)
        if self._only_last_channel:
            dummy_observation = tree.map_structure(lambda t: t[..., -1:], dummy_observation)
        params = self._init_fn(rngkey, dummy_observation)
        dummy_output = self._apply_fn(params, dummy_observation)
        return params, dummy_output.shape[-1]

    def apply(self, params, inputs):
        if self._only_last_channel:
            inputs = tree.map_structure(lambda t: t[..., -1:], inputs)
        return self._apply_fn(params, inputs)  # [T, ...]


def get_random_feature_fn(rngkey, observation_space, delta, absolute=False, **kwargs):
    random_feature_net = RandomConvNet(observation_space, **kwargs)
    params, num_targets = random_feature_net.init(rngkey)

    def fn(params, traj):
        y_tp1 = random_feature_net.apply(params, traj.observation[1:])
        if delta:
            y = random_feature_net.apply(params, traj.observation[:-1])
            c = y_tp1 - y
        else:
            c = y_tp1
        if absolute:
            c = jnp.abs(c)
        return c

    return fn, params, num_targets
