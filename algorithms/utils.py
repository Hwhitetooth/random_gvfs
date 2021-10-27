import jax.numpy as jnp
import numpy as onp
import tree


def pack_namedtuple_jnp(xs, axis=0):
    return tree.map_structure(lambda *xs: jnp.stack(xs, axis=axis), *xs)


def pack_namedtuple_onp(xs, axis=0):
    return tree.map_structure(lambda *xs: onp.stack(xs, axis=axis), *xs)


def unpack_namedtuple_jnp(structure, axis=0):
    transposed = tree.map_structure(lambda t: jnp.moveaxis(t, axis, 0), structure)
    flat = tree.flatten(transposed)
    unpacked = list(map(lambda xs: tree.unflatten_as(structure, xs), zip(*flat)))
    return unpacked


def unpack_namedtuple_onp(structure, axis=0):
    transposed = tree.map_structure(lambda t: onp.moveaxis(t, axis, 0), structure)
    flat = tree.flatten(transposed)
    unpacked = list(map(lambda xs: tree.unflatten_as(structure, xs), zip(*flat)))
    return unpacked


def explained_variance(y_pred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and y_pred.ndim == 1
    var_y = onp.var(y)
    return onp.nan if var_y == 0 else 1 - onp.var(y - y_pred) / var_y
