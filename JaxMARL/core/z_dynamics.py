from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp


class ZDynamicsState(NamedTuple):
    z_init_budget: jnp.ndarray
    G_ext_best: jnp.ndarray
    G_int_at_best_ext: jnp.ndarray


class ZDynamicsMetrics(NamedTuple):
    G_int_mean_all: jnp.ndarray
    top_k_G_ext_mean: jnp.ndarray
    z_init_update_triggered: jnp.ndarray


def roll_z_trajectory_forward(
    intrinsic_reward: jnp.ndarray,
    dones: jnp.ndarray,
    z_init_budget: jnp.ndarray,
    num_steps: int,
    num_agents: int,
    num_envs: int,
    gamma_int: float,
    z_clip_min: float,
    z_clip_max: float,
    z_budget_reset_on_done: bool,
) -> jnp.ndarray:
    r_int_reshaped = intrinsic_reward.reshape(num_steps, num_agents, num_envs)
    dones_reshaped = dones.reshape(num_steps, num_agents, num_envs)

    z_init_broadcast = z_init_budget[:, None]

    def _forward(z_t, inputs):
        r_t, done_t = inputs
        z_next = (z_t - r_t) / gamma_int
        z_next = jnp.clip(z_next, z_clip_min, z_clip_max)

        if z_budget_reset_on_done:
            z_next = jnp.where(done_t, z_init_broadcast, z_next)

        z_next = jax.lax.stop_gradient(z_next)
        return z_next, z_t

    initial_z = jnp.broadcast_to(z_init_broadcast, (num_agents, num_envs))
    _, z_traj = jax.lax.scan(
        _forward, initial_z,
        (r_int_reshaped, dones_reshaped),
        reverse=False,
    )
    return jax.lax.stop_gradient(z_traj)


def update_z_init_adaptive(
    intrinsic_reward_all: jnp.ndarray,
    reward_ext: jnp.ndarray,
    gamma_powers: jnp.ndarray,
    z_init_budget_current: jnp.ndarray,
    G_ext_best_current: jnp.ndarray,
    G_int_at_best_ext_current: jnp.ndarray,
    update_steps: int,
    num_steps: int,
    num_agents: int,
    num_envs: int,
    use_adaptive_z_init: bool,
    use_random_z_init: bool,
    random_z_init_min: float,
    random_z_init_max: float,
    alpha_z: float,
    z_init_use_top_k: bool,
    z_init_top_k: int,
    g_ext_best_decay: float,
    warmup_rollouts: int,
    seed: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, ZDynamicsMetrics]:
    if use_adaptive_z_init:
        r_int_reshaped = intrinsic_reward_all.reshape(num_steps, num_agents, num_envs)
        r_ext_reshaped = reward_ext.reshape(num_steps, num_agents, num_envs)

        gamma_powers_expanded = gamma_powers[:, None, None]
        G_int_discounted = jnp.sum(r_int_reshaped * gamma_powers_expanded, axis=0)

        G_int_mean_all = jnp.mean(G_int_discounted)

        G_ext_team_per_env = jnp.sum(r_ext_reshaped * gamma_powers_expanded, axis=(0, 1))

        if z_init_use_top_k:
            z_init_budget_new, G_ext_best_new, G_int_at_best_ext_new, z_init_update_triggered, top_k_G_ext_mean = \
                _update_z_init_top_k_mode(
                    G_int_discounted, G_ext_team_per_env,
                    z_init_budget_current, G_ext_best_current, G_int_at_best_ext_current,
                    update_steps, num_envs, warmup_rollouts,
                    alpha_z, z_init_top_k, g_ext_best_decay
                )
        else:
            z_init_budget_new, G_ext_best_new, G_int_at_best_ext_new, z_init_update_triggered, top_k_G_ext_mean = \
                _update_z_init_mean_mode(
                    G_int_discounted, z_init_budget_current,
                    G_ext_best_current, G_int_at_best_ext_current,
                    update_steps, alpha_z
                )
    else:
        z_init_budget_new, G_ext_best_new, G_int_at_best_ext_new, z_init_update_triggered, G_int_mean_all, top_k_G_ext_mean = \
            _update_z_init_fixed_or_random(
                z_init_budget_current, G_ext_best_current, G_int_at_best_ext_current,
                use_random_z_init, random_z_init_min, random_z_init_max,
                num_agents, update_steps, seed
            )

    metrics = ZDynamicsMetrics(
        G_int_mean_all=G_int_mean_all,
        top_k_G_ext_mean=top_k_G_ext_mean,
        z_init_update_triggered=z_init_update_triggered,
    )

    return z_init_budget_new, G_ext_best_new, G_int_at_best_ext_new, metrics


def _update_z_init_top_k_mode(
    G_int_discounted: jnp.ndarray,
    G_ext_team_per_env: jnp.ndarray,
    z_init_budget_current: jnp.ndarray,
    G_ext_best_current: jnp.ndarray,
    G_int_at_best_ext_current: jnp.ndarray,
    update_steps: int,
    num_envs: int,
    warmup_rollouts: int,
    alpha_z: float,
    z_init_top_k: int,
    g_ext_best_decay: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k_clamped = min(z_init_top_k, num_envs)
    top_k_G_ext_values, top_k_indices = jax.lax.top_k(G_ext_team_per_env, k=k_clamped)

    top_k_G_ext_mean = jnp.mean(top_k_G_ext_values)

    is_warmup = update_steps < warmup_rollouts
    best_env_warmup = jnp.argmax(G_ext_team_per_env)

    G_ext_best_decayed = jnp.where(
        G_ext_best_current >= 0,
        G_ext_best_current * g_ext_best_decay,
        G_ext_best_current / g_ext_best_decay
    )

    exceeds_mask = top_k_G_ext_values > G_ext_best_decayed
    any_exceeds = jnp.any(exceeds_mask)

    G_int_top_k = G_int_discounted[:, top_k_indices]
    G_int_team_top_k = jnp.sum(G_int_top_k, axis=0)

    exceed_G_int_team_masked = jnp.where(exceeds_mask, G_int_team_top_k, -jnp.inf)
    best_int_in_topk = jnp.argmax(exceed_G_int_team_masked)
    best_env_exceeds = top_k_indices[best_int_in_topk]

    G_int_for_z_init = G_int_discounted[:, best_env_exceeds]
    G_ext_of_best_int_env = G_ext_team_per_env[best_env_exceeds]

    G_ext_best_warmup_candidate = G_ext_team_per_env[best_env_warmup]
    warmup_should_update = G_ext_best_warmup_candidate > G_ext_best_current

    G_ext_best_warmup = jnp.where(
        warmup_should_update,
        G_ext_best_warmup_candidate,
        G_ext_best_current
    )

    G_int_at_best_ext_warmup = jnp.where(
        warmup_should_update,
        G_int_discounted[:, best_env_warmup],
        G_int_at_best_ext_current
    )

    G_ext_best_exceeds = G_ext_of_best_int_env
    G_int_at_best_ext_exceeds = G_int_for_z_init

    G_ext_best_new = jnp.where(
        is_warmup,
        G_ext_best_warmup,
        jnp.where(
            any_exceeds,
            G_ext_best_exceeds,
            G_ext_best_decayed
        )
    )

    G_int_at_best_ext_new = jnp.where(
        is_warmup,
        G_int_at_best_ext_warmup,
        jnp.where(
            any_exceeds,
            G_int_at_best_ext_exceeds,
            G_int_at_best_ext_current
        )
    )

    z_init_warmup = G_int_at_best_ext_warmup
    z_init_ema = alpha_z * G_int_for_z_init + (1 - alpha_z) * z_init_budget_current

    z_init_budget_new = jnp.where(
        is_warmup,
        jnp.clip(z_init_warmup, 0.1, 1000.0),
        jnp.where(
            any_exceeds,
            jnp.clip(z_init_ema, 0.1, 1000.0),
            z_init_budget_current
        )
    )

    z_init_update_triggered = jnp.logical_or(is_warmup, any_exceeds)

    return z_init_budget_new, G_ext_best_new, G_int_at_best_ext_new, z_init_update_triggered, top_k_G_ext_mean


def _update_z_init_mean_mode(
    G_int_discounted: jnp.ndarray,
    z_init_budget_current: jnp.ndarray,
    G_ext_best_current: jnp.ndarray,
    G_int_at_best_ext_current: jnp.ndarray,
    update_steps: int,
    alpha_z: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    G_int_for_ema_per_agent = jnp.mean(G_int_discounted, axis=1)

    top_k_G_ext_mean = jnp.array(0.0)

    z_candidate = jnp.where(
        update_steps == 0,
        G_int_for_ema_per_agent,
        alpha_z * G_int_for_ema_per_agent + (1 - alpha_z) * z_init_budget_current
    )
    z_init_budget_new = jnp.clip(z_candidate, 0.1, 1000.0)

    G_ext_best_new = G_ext_best_current
    G_int_at_best_ext_new = G_int_at_best_ext_current
    z_init_update_triggered = jnp.array(True)

    return z_init_budget_new, G_ext_best_new, G_int_at_best_ext_new, z_init_update_triggered, top_k_G_ext_mean


def _update_z_init_fixed_or_random(
    z_init_budget_current: jnp.ndarray,
    G_ext_best_current: jnp.ndarray,
    G_int_at_best_ext_current: jnp.ndarray,
    use_random_z_init: bool,
    random_z_init_min: float,
    random_z_init_max: float,
    num_agents: int,
    update_steps: int,
    seed: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if use_random_z_init:
        z_init_rng = jax.random.PRNGKey(seed * 10000 + update_steps)
        z_init_budget_new = jax.random.uniform(
            z_init_rng, shape=(num_agents,),
            minval=random_z_init_min, maxval=random_z_init_max
        )
    else:
        z_init_budget_new = z_init_budget_current

    G_ext_best_new = G_ext_best_current
    G_int_at_best_ext_new = G_int_at_best_ext_current
    G_int_mean_all = jnp.array(0.0)
    top_k_G_ext_mean = jnp.array(0.0)
    z_init_update_triggered = jnp.array(False)

    return z_init_budget_new, G_ext_best_new, G_int_at_best_ext_new, z_init_update_triggered, G_int_mean_all, top_k_G_ext_mean
