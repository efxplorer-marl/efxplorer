from typing import NamedTuple
import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    action_logits: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    z_i: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    agent_positions: jnp.ndarray
    agent_local_features_curr: jnp.ndarray = None
    agent_local_features_next: jnp.ndarray = None
    agent_idx: jnp.ndarray = None


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def discounted_sampling(ranges, discount, rng):
    assert 0 < discount < 1
    seeds = jax.random.uniform(rng, shape=ranges.shape)
    samples = jnp.log(1 - (1 - discount**ranges) * seeds) / jnp.log(discount)
    samples = jnp.minimum(jnp.floor(samples).astype(jnp.int32), ranges - 1)
    return samples


def make_linear_schedule(lr: float, num_minibatches: int, update_epochs: int, num_updates: int):
    def schedule(count):
        frac = 1.0 - (count // (num_minibatches * update_epochs)) / num_updates
        return lr * frac

    return schedule
