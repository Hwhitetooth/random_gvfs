import haiku as hk
import jax


class ConvTorso(hk.Module):
    """Shallow convolutional torso from the DQN paper."""

    def __init__(self, dense_layers, name=None, **unused_kwargs):
        super(ConvTorso, self).__init__(name=name)
        self._dense_layers = dense_layers

    def __call__(self, x):
        torso_net = [
            lambda x: x / 255.,
            hk.Conv2D(32, kernel_shape=[8, 8], stride=[4, 4], padding='VALID'),
            jax.nn.relu,
            hk.Conv2D(64, kernel_shape=[4, 4], stride=[2, 2], padding='VALID'),
            jax.nn.relu,
            hk.Conv2D(64, kernel_shape=[3, 3], stride=[1, 1], padding='VALID'),
            jax.nn.relu,
            hk.Flatten(),
        ]
        for dim in self._dense_layers:
            torso_net.append(hk.Linear(dim))
            torso_net.append(jax.nn.relu)
        return hk.Sequential(torso_net)(x)
