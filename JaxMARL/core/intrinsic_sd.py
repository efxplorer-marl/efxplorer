import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Optional, Any, Callable, List

from .tdd_networks import mrn_distance
from .utils import stack_params


def compute_min_sd_single_agent(
    agent_features_curr: jnp.ndarray,
    agent_features_next: jnp.ndarray,
    s_encoder_apply_fn: Callable,
    s_encoder_params: Any,
    num_steps: int,
    num_envs: int,
) -> jnp.ndarray:
    def _compute_dist_matrix_for_env(env_idx):
        s_k_all = agent_features_curr[:, env_idx, :]
        s_t1_all = agent_features_next[:, env_idx, :]

        phi_k_all = jax.vmap(lambda s: s_encoder_apply_fn(s_encoder_params, s))(s_k_all)
        phi_t1_all = jax.vmap(lambda s: s_encoder_apply_fn(s_encoder_params, s))(s_t1_all)

        dist_matrix = mrn_distance(
            phi_k_all[None, :, :],
            phi_t1_all[:, None, :]
        )

        t_indices = jnp.arange(num_steps)[:, None]
        k_indices = jnp.arange(num_steps)[None, :]
        mask = k_indices <= t_indices

        large_val = 1e10
        dist_matrix_masked = jnp.where(mask, dist_matrix, large_val)
        min_sd_per_t = jnp.min(dist_matrix_masked, axis=1)

        return min_sd_per_t

    min_sd_all = jax.vmap(_compute_dist_matrix_for_env)(jnp.arange(num_envs))
    min_sd_all = min_sd_all.T

    return min_sd_all


def compute_intrinsic_reward_sd(
    agent_features_curr_all: jnp.ndarray,
    agent_features_next_all: jnp.ndarray,
    tdd_s_encoder_states: Any,
    tdd_shared: bool,
    num_agents: int,
    num_envs: int,
    num_steps: int,
) -> jnp.ndarray:
    if tdd_shared:
        s_encoder_apply_fn = tdd_s_encoder_states.apply_fn
        shared_s_encoder_params = tdd_s_encoder_states.params
        stacked_s_encoder_params = None
    else:
        s_encoder_apply_fn = tdd_s_encoder_states[0].apply_fn
        shared_s_encoder_params = None
        stacked_s_encoder_params = stack_params([s.params for s in tdd_s_encoder_states])

    if tdd_shared:
        intrinsic_reward_per_agent = jax.vmap(
            lambda curr, next_: compute_min_sd_single_agent(
                curr, next_, s_encoder_apply_fn, shared_s_encoder_params, num_steps, num_envs
            )
        )(agent_features_curr_all, agent_features_next_all)
    else:
        def _compute_sd_with_agent_params(curr, next_, params):
            return compute_min_sd_single_agent(
                curr, next_, s_encoder_apply_fn, params, num_steps, num_envs
            )
        intrinsic_reward_per_agent = jax.vmap(_compute_sd_with_agent_params)(
            agent_features_curr_all, agent_features_next_all, stacked_s_encoder_params
        )

    return intrinsic_reward_per_agent
