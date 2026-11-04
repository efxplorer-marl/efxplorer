from typing import Dict, List, Tuple, Union, Callable

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax

from .data_utils import discounted_sampling
from .utils import stack_params


def mrn_distance(x, y):
    eps = 1e-6
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = jnp.maximum(0, x_prefix - y_prefix).max(axis=-1)
    l2_component = jnp.sqrt(jnp.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    return max_component + l2_component


class PotentialNet(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return value


class S_Encoder(nn.Module):
    latent_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        encoded = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return encoded


class DeepPotentialNet(nn.Module):
    latent_dim: int
    num_blocks: int = 3

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        x = nn.relu(x)

        for _ in range(self.num_blocks):
            identity = x
            x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
            x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
            x = x + identity

        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return value


class DeepS_Encoder(nn.Module):
    latent_dim: int
    output_dim: int
    num_blocks: int = 3

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        x = nn.relu(x)

        for _ in range(self.num_blocks):
            identity = x
            x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
            x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
            x = x + identity

        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        encoded = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return encoded


class MetricPreservingEncoder(nn.Module):
    latent_dim: int = 64
    output_dim: int = 32

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(state)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Dense(self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        z = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

        return z


def create_tdd_train_states(
    rng: jax.random.PRNGKey,
    config: Dict,
    num_agents: int,
    input_dim: int = 2,
) -> Tuple[List[TrainState], List[TrainState]]:
    latent_dim = config.get("TDD_LATENT_DIM", 64)
    output_dim = config.get("TDD_OUTPUT_DIM", 32)
    tdd_lr = config.get("TDD_LR", 1e-3)
    use_deep = config.get("TDD_USE_DEEP_NETWORK", False)
    num_blocks = config.get("TDD_NUM_RESIDUAL_BLOCKS", 3)
    max_grad_norm = config.get("MAX_GRAD_NORM", 0.5)

    def _make_tx():
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(tdd_lr, eps=1e-5),
        )

    dummy_input = jnp.zeros((1, input_dim))

    if use_deep:
        potential_nets = [DeepPotentialNet(latent_dim=latent_dim, num_blocks=num_blocks) for _ in range(num_agents)]
        s_encoders = [DeepS_Encoder(latent_dim=latent_dim, output_dim=output_dim, num_blocks=num_blocks) for _ in range(num_agents)]
    else:
        potential_nets = [PotentialNet(latent_dim=latent_dim) for _ in range(num_agents)]
        s_encoders = [S_Encoder(latent_dim=latent_dim, output_dim=output_dim) for _ in range(num_agents)]

    tdd_rngs = list(jax.random.split(rng, num_agents * 2))

    potential_params = [potential_nets[i].init(tdd_rngs[i * 2], dummy_input) for i in range(num_agents)]
    s_encoder_params = [s_encoders[i].init(tdd_rngs[i * 2 + 1], dummy_input) for i in range(num_agents)]

    tdd_potential_states = [
        TrainState.create(apply_fn=potential_nets[i].apply, params=potential_params[i], tx=_make_tx())
        for i in range(num_agents)
    ]
    tdd_s_encoder_states = [
        TrainState.create(apply_fn=s_encoders[i].apply, params=s_encoder_params[i], tx=_make_tx())
        for i in range(num_agents)
    ]

    return tdd_potential_states, tdd_s_encoder_states


def create_metric_encoder_state(
    rng: jax.random.PRNGKey,
    config: Dict,
    num_agents: int,
    input_dim: int = 2,
) -> List[TrainState]:
    latent_dim = config.get("TDD_LATENT_DIM", 64)
    output_dim = config.get("METRIC_ENCODER_DIM", 32)
    metric_lr = config.get("TDD_LR", 1e-3)
    max_grad_norm = config.get("MAX_GRAD_NORM", 0.5)

    def _make_tx():
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(metric_lr, eps=1e-5),
        )

    dummy_input = jnp.zeros((1, input_dim))

    metric_encoders = [
        MetricPreservingEncoder(latent_dim=latent_dim, output_dim=output_dim)
        for _ in range(num_agents)
    ]

    rngs = list(jax.random.split(rng, num_agents))

    params_list = [
        metric_encoders[i].init(rngs[i], dummy_input)
        for i in range(num_agents)
    ]

    metric_encoder_states = [
        TrainState.create(apply_fn=metric_encoders[i].apply, params=params_list[i], tx=_make_tx())
        for i in range(num_agents)
    ]

    return metric_encoder_states


def metric_preservation_loss(
    encoder_params,
    apply_fn: Callable,
    s_t: jnp.ndarray,
    s_t_k: jnp.ndarray,
    k: jnp.ndarray,
    max_k: int,
) -> jnp.ndarray:
    z_t = apply_fn(encoder_params, s_t)
    z_t_k = apply_fn(encoder_params, s_t_k)

    latent_dist = jnp.linalg.norm(z_t - z_t_k, axis=-1)

    k_float = k.astype(jnp.float32)
    target_dist = 2.0 * jnp.log(k_float + 1.0) / jnp.log(max_k + 1.0)

    loss = jnp.mean((latent_dist - target_dist) ** 2)

    return loss


def update_metric_encoder(
    metric_states: List[TrainState],
    agent_features: jnp.ndarray,
    rng: jax.random.PRNGKey,
    tdd_config: Dict,
    num_agents: int,
    num_steps: int,
    num_envs: int,
    metric_pairs_per_update: int,
) -> Tuple[List[TrainState], jnp.ndarray]:
    agent_rngs = jax.random.split(rng, num_agents)
    metric_apply_fn = metric_states[0].apply_fn
    stacked_metric_params = stack_params([s.params for s in metric_states])

    def _metric_loss_single_agent(metric_params, agent_features_single, agent_rng):
        rng_t, rng_env, rng_k = jax.random.split(agent_rng, 3)

        t_indices = jax.random.randint(rng_t, (metric_pairs_per_update,), 0, num_steps - 1)
        env_indices = jax.random.randint(rng_env, (metric_pairs_per_update,), 0, num_envs)

        ranges = (num_steps - 1 - t_indices).astype(jnp.int32)
        ranges = jnp.maximum(ranges, 1)
        k_values = discounted_sampling(ranges, tdd_config["tdd_discount"], rng_k)
        k_values = jnp.maximum(k_values, 1)

        t_plus_k = jnp.clip(t_indices + k_values, 0, num_steps - 1)

        s_t = agent_features_single[t_indices, env_indices, :]
        s_t_k = agent_features_single[t_plus_k, env_indices, :]

        return metric_preservation_loss(
            metric_params, metric_apply_fn, s_t, s_t_k, k_values, num_steps
        )

    def _compute_loss_and_grad(metric_params, agent_features_single, agent_rng):
        loss, grads = jax.value_and_grad(_metric_loss_single_agent)(
            metric_params, agent_features_single, agent_rng
        )
        return loss, grads

    losses, grads = jax.vmap(_compute_loss_and_grad)(
        stacked_metric_params, agent_features, agent_rngs
    )

    new_metric_states = []
    for i in range(num_agents):
        grads_i = jax.tree.map(lambda x: x[i], grads)
        new_metric_states.append(metric_states[i].apply_gradients(grads=grads_i))

    return new_metric_states, losses.mean()
