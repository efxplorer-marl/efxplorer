import jax
import jax.numpy as jnp
import optax
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict


def stack_params(params_list):
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params_list)


class ParamsHolder:
    def __init__(self, params):
        self.params = params


def save_params(params, filename):
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)


def load_params(filename):
    flat_dict = load_file(filename)
    tuple_key_dict = {tuple(k.split(',')): v for k, v in flat_dict.items()}
    return unflatten_dict(tuple_key_dict)


def vectorized_apply_gradients(train_states, stacked_grads):
    stacked_params = stack_params([s.params for s in train_states])

    stacked_opt_state = jax.tree.map(
        lambda *xs: jnp.stack(xs), *[s.opt_state for s in train_states]
    )

    tx = train_states[0].tx

    def _apply_single(params, opt_state, grads):
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    new_stacked_params, new_stacked_opt_state = jax.vmap(_apply_single)(
        stacked_params, stacked_opt_state, stacked_grads
    )

    num_agents = len(train_states)
    return [
        train_states[i].replace(
            params=jax.tree.map(lambda x: x[i], new_stacked_params),
            opt_state=jax.tree.map(lambda x: x[i], new_stacked_opt_state),
            step=train_states[i].step + 1,
        )
        for i in range(num_agents)
    ]
