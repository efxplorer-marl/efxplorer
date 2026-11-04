import jax
import jax.numpy as jnp
from typing import Tuple, Dict


def _apply_z_failure_factor(z: jnp.ndarray, factor: float) -> jnp.ndarray:
    is_reset = factor < 1e-6

    z_adjusted = jnp.where(
        z >= 0,
        z * factor,
        z / jnp.maximum(factor, 1e-6)
    )

    return jnp.where(is_reset, jnp.zeros_like(z), z_adjusted)


def update_z0_from_episode_data(
    ext_episode_sums: jnp.ndarray,
    z0_episode_sums: jnp.ndarray,
    z_0_ema: jnp.ndarray,
    best_ext_reward: jnp.ndarray,
    z_ema_alpha: float,
    z0_best_reward_decay: float,
    z_init_decay_on_failure: bool = False,
    z_init_decay_factor: float = 0.95,
    z_failure_factor: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    num_episodes = ext_episode_sums.shape[0]

    use_factor_mode = z_failure_factor < 1.0 - 1e-6

    def empty_case():
        if use_factor_mode:
            new_z = _apply_z_failure_factor(z_0_ema, z_failure_factor)
            new_threshold = best_ext_reward
            z_factor_applied = 1.0
            z_decayed_flag = 0.0
        elif z_init_decay_on_failure:
            new_z = z_0_ema * z_init_decay_factor
            new_threshold = best_ext_reward
            z_factor_applied = 0.0
            z_decayed_flag = 1.0
        else:
            new_z = z_0_ema
            new_threshold = jnp.where(
                best_ext_reward >= 0,
                best_ext_reward * z0_best_reward_decay,
                best_ext_reward / z0_best_reward_decay
            )
            z_factor_applied = 0.0
            z_decayed_flag = 0.0
        update_info = {
            "passed_ratio": jnp.array(0.0, dtype=jnp.float32),
            "update_delta_mean": jnp.mean(jnp.abs(z_0_ema - new_z)),
            "z_decayed_on_failure": jnp.array(z_decayed_flag, dtype=jnp.float32),
            "z_factor_applied": jnp.array(z_factor_applied, dtype=jnp.float32),
        }
        return new_z, jnp.float32(new_threshold), update_info

    def non_empty_case():
        better_mask = ext_episode_sums > best_ext_reward
        num_better = jnp.sum(better_mask)
        passed_ratio = num_better / num_episodes

        def update_case():
            intrinsic_totals = jnp.sum(z0_episode_sums, axis=1)
            masked_intrinsic = jnp.where(better_mask, intrinsic_totals, -jnp.inf)
            selected_idx = jnp.argmax(masked_intrinsic)

            selected_z0 = z0_episode_sums[selected_idx]
            selected_ext = ext_episode_sums[selected_idx]

            new_z_0 = z_ema_alpha * selected_z0 + (1 - z_ema_alpha) * z_0_ema

            update_delta = jnp.abs(new_z_0 - z_0_ema)
            update_delta_mean = jnp.mean(update_delta)

            return new_z_0, jnp.float32(selected_ext), update_delta_mean, jnp.array(0.0, dtype=jnp.float32)

        def no_update_case():
            if use_factor_mode:
                new_z = _apply_z_failure_factor(z_0_ema, z_failure_factor)
                new_threshold = best_ext_reward
                z_factor_applied = 1.0
            elif z_init_decay_on_failure:
                new_z = z_0_ema * z_init_decay_factor
                new_threshold = best_ext_reward
                z_factor_applied = 0.0
            else:
                new_z = z_0_ema
                new_threshold = jnp.where(
                    best_ext_reward >= 0,
                    best_ext_reward * z0_best_reward_decay,
                    best_ext_reward / z0_best_reward_decay
                )
                z_factor_applied = 0.0
            z_change_delta = jnp.mean(jnp.abs(z_0_ema - new_z))
            return new_z, jnp.float32(new_threshold), z_change_delta, jnp.array(z_factor_applied, dtype=jnp.float32)

        new_z_0_ema, final_best_ext, update_delta_mean, z_factor_applied = jax.lax.cond(
            num_better > 0, update_case, no_update_case
        )

        z_decayed_flag = jnp.where(
            (num_better == 0) & z_init_decay_on_failure & (~use_factor_mode),
            1.0, 0.0
        )
        update_info = {
            "passed_ratio": jnp.float32(passed_ratio),
            "update_delta_mean": jnp.float32(update_delta_mean),
            "z_decayed_on_failure": jnp.float32(z_decayed_flag),
            "z_factor_applied": jnp.float32(z_factor_applied),
        }

        return new_z_0_ema, final_best_ext, update_info

    return jax.lax.cond(num_episodes > 0, non_empty_case, empty_case)


def collect_episode_data_simple(
    intrinsic_rewards: jnp.ndarray,
    extrinsic_rewards: jnp.ndarray,
    gamma: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    num_steps, num_agents, num_envs = intrinsic_rewards.shape

    discount_factors = jnp.power(gamma, jnp.arange(num_steps))[:, None, None]

    discounted_int = intrinsic_rewards * discount_factors
    z0_per_agent_env = jnp.sum(discounted_int, axis=0)
    z0_per_env = z0_per_agent_env.T

    ext_per_env = jnp.sum(extrinsic_rewards[:, 0, :], axis=0)

    return ext_per_env, z0_per_env


def collect_episode_data_with_boundaries(
    intrinsic_rewards: jnp.ndarray,
    extrinsic_rewards: jnp.ndarray,
    dones: jnp.ndarray,
    num_agents: int,
    num_envs: int,
    gamma: float,
    max_episodes_per_env: int = 2,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_steps = intrinsic_rewards.shape[0]
    max_total_episodes = num_envs * max_episodes_per_env

    int_rewards = intrinsic_rewards.reshape(num_steps, num_agents, num_envs)
    ext_rewards = extrinsic_rewards.reshape(num_steps, num_agents, num_envs)
    done_flags = dones.reshape(num_steps, num_agents, num_envs)

    done_agent0 = done_flags[:, 0, :]

    discount_factors = jnp.power(gamma, jnp.arange(num_steps))

    def process_env(env_idx):
        env_dones = done_agent0[:, env_idx]
        env_int = int_rewards[:, :, env_idx]
        env_ext = ext_rewards[:, 0, env_idx]

        done_positions = jnp.where(env_dones, jnp.arange(num_steps), -1)

        full_discounted_int = jnp.sum(env_int * discount_factors[:, None], axis=0)
        full_ext = jnp.sum(env_ext)

        has_episode = jnp.any(env_dones)

        z0_episodes = jnp.zeros((max_episodes_per_env, num_agents))
        z0_episodes = z0_episodes.at[0].set(full_discounted_int)

        ext_episodes = jnp.zeros(max_episodes_per_env)
        ext_episodes = ext_episodes.at[0].set(full_ext)

        valid = jnp.zeros(max_episodes_per_env, dtype=jnp.bool_)
        valid = valid.at[0].set(has_episode)

        return ext_episodes, z0_episodes, valid

    ext_all, z0_all, valid_all = jax.vmap(process_env)(jnp.arange(num_envs))

    ext_episode_sums = ext_all.reshape(max_total_episodes)
    z0_episode_sums = z0_all.reshape(max_total_episodes, num_agents)
    valid_mask = valid_all.reshape(max_total_episodes)

    return ext_episode_sums, z0_episode_sums, valid_mask
