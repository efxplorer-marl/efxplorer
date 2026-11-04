import jax
import jax.numpy as jnp
from typing import Tuple, Dict


def compute_joint_advantage_kstep(
    value_t: jnp.ndarray,
    V_int_t: jnp.ndarray,
    delta_ext_t: jnp.ndarray,
    delta_int_t: jnp.ndarray,
    z_i_t: jnp.ndarray,
    done_t: jnp.ndarray,
    gamma: float,
    lmbda: float,
    adv_clip_abs: float = 20.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    T, NUM_ACTORS = delta_ext_t.shape

    gamma_powers = jnp.power(gamma, jnp.arange(T))
    lambda_powers = jnp.power(lmbda, jnp.arange(T))

    def compute_kstep_for_actor(delta_ext, delta_int, value, V_int, z_i, done):

        delta_ext_padded = jnp.concatenate([delta_ext, jnp.zeros(T)])
        delta_int_padded = jnp.concatenate([delta_int, jnp.zeros(T)])

        def compute_kstep_row(t):
            deltas_ext_from_t = jax.lax.dynamic_slice(delta_ext_padded, (t,), (T,))
            deltas_int_from_t = jax.lax.dynamic_slice(delta_int_padded, (t,), (T,))

            weighted_ext = deltas_ext_from_t * gamma_powers
            weighted_int = deltas_int_from_t * gamma_powers

            cumsum_ext = jnp.cumsum(weighted_ext)
            cumsum_int = jnp.cumsum(weighted_int)

            G_ext_kstep_row = value[t] + cumsum_ext
            G_int_kstep_row = V_int[t] + cumsum_int

            valid_mask = jnp.arange(T) < (T - t)

            G_ext_kstep_row = jnp.where(valid_mask, G_ext_kstep_row, 0.0)
            G_int_kstep_row = jnp.where(valid_mask, G_int_kstep_row, 0.0)

            return G_ext_kstep_row, G_int_kstep_row, valid_mask

        G_ext_kstep, G_int_kstep, valid_masks = jax.vmap(compute_kstep_row)(jnp.arange(T))

        A_ext_kstep = G_ext_kstep - value[:, None]

        S_int_kstep = G_int_kstep - z_i[:, None]

        int_selected_mask = (S_int_kstep <= A_ext_kstep)

        V_epi_kstep = jnp.minimum(A_ext_kstep, S_int_kstep)

        int_baseline = (V_int - z_i)[:, None]
        A_epi_kstep = V_epi_kstep - int_baseline * int_selected_mask.astype(jnp.float32)

        A_epi_kstep = jnp.where(valid_masks, A_epi_kstep, 0.0)

        weighted = A_epi_kstep * lambda_powers[None, :] * valid_masks
        weighted_sum = jnp.sum(weighted, axis=1)
        lambda_sum = jnp.sum(lambda_powers[None, :] * valid_masks, axis=1)
        A_JT = weighted_sum / (lambda_sum + 1e-8)

        ext_selected = A_ext_kstep[:, 0] < S_int_kstep[:, 0]

        return A_JT, A_ext_kstep[:, 0], S_int_kstep[:, 0], ext_selected

    A_JT, A_ext_k0, S_int_k0, ext_selected = jax.vmap(
        compute_kstep_for_actor, in_axes=1, out_axes=(1, 1, 1, 1)
    )(delta_ext_t, delta_int_t, value_t, V_int_t, z_i_t, done_t)

    A_JT = jnp.clip(A_JT, -adv_clip_abs, adv_clip_abs)

    total_count = T * NUM_ACTORS
    ext_count = jnp.sum(ext_selected.astype(jnp.float32))
    ext_selection_rate = ext_count / total_count

    A_ext_abs_mean = jnp.mean(jnp.abs(A_ext_k0))
    S_int_abs_mean = jnp.mean(jnp.abs(S_int_k0))
    A_ext_std = jnp.std(A_ext_k0)
    S_int_std = jnp.std(S_int_k0)
    scale_ratio = S_int_abs_mean / (A_ext_abs_mean + 1e-8)
    std_ratio = S_int_std / (A_ext_std + 1e-8)

    stats = {
        "A_JT_mean": jnp.mean(A_JT),
        "A_JT_std": jnp.std(A_JT),
        "A_JT_min": jnp.min(A_JT),
        "A_JT_max": jnp.max(A_JT),
        "A_ext_kstep_mean": jnp.mean(A_ext_k0),
        "S_int_kstep_mean": jnp.mean(S_int_k0),
        "ext_selection_rate": ext_selection_rate,
        "int_selection_rate": 1.0 - ext_selection_rate,
        "A_ext_abs_mean": A_ext_abs_mean,
        "S_int_abs_mean": S_int_abs_mean,
        "A_ext_std": A_ext_std,
        "S_int_std": S_int_std,
        "scale_ratio_abs": scale_ratio,
        "scale_ratio_std": std_ratio,
    }

    return A_JT, stats


def compute_joint_advantage_kstep_popart(
    value_t: jnp.ndarray,
    V_int_t: jnp.ndarray,
    delta_ext_t: jnp.ndarray,
    delta_int_t: jnp.ndarray,
    z_i_t: jnp.ndarray,
    done_t: jnp.ndarray,
    gamma: float,
    lmbda: float,
    adv_clip_abs: float = 20.0,
    ext_mean: float = 0.0,
    ext_std: float = 1.0,
    int_mean: float = 0.0,
    int_std: float = 1.0,
    eps: float = 1e-5,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    T, NUM_ACTORS = delta_ext_t.shape

    gamma_powers = jnp.power(gamma, jnp.arange(T))
    lambda_powers = jnp.power(lmbda, jnp.arange(T))

    ext_std_safe = jnp.maximum(ext_std, eps)
    int_std_safe = jnp.maximum(int_std, eps)

    def compute_kstep_for_actor_popart(delta_ext, delta_int, value, V_int, z_i, done):
        delta_ext_padded = jnp.concatenate([delta_ext, jnp.zeros(T)])
        delta_int_padded = jnp.concatenate([delta_int, jnp.zeros(T)])

        def compute_kstep_row(t):
            deltas_ext_from_t = jax.lax.dynamic_slice(delta_ext_padded, (t,), (T,))
            deltas_int_from_t = jax.lax.dynamic_slice(delta_int_padded, (t,), (T,))

            weighted_ext = deltas_ext_from_t * gamma_powers
            weighted_int = deltas_int_from_t * gamma_powers

            cumsum_ext = jnp.cumsum(weighted_ext)
            cumsum_int = jnp.cumsum(weighted_int)

            G_ext_kstep_row = value[t] + cumsum_ext
            G_int_kstep_row = V_int[t] + cumsum_int

            valid_mask = jnp.arange(T) < (T - t)
            G_ext_kstep_row = jnp.where(valid_mask, G_ext_kstep_row, 0.0)
            G_int_kstep_row = jnp.where(valid_mask, G_int_kstep_row, 0.0)

            return G_ext_kstep_row, G_int_kstep_row, valid_mask

        G_ext_kstep, G_int_kstep, valid_masks = jax.vmap(compute_kstep_row)(jnp.arange(T))

        A_ext_kstep = G_ext_kstep - value[:, None]
        S_int_kstep = G_int_kstep - z_i[:, None]

        A_ext_norm = (A_ext_kstep - ext_mean) / ext_std_safe
        S_int_norm = (S_int_kstep - int_mean) / int_std_safe

        int_selected_mask = (S_int_norm <= A_ext_norm)

        V_epi_kstep = jnp.where(int_selected_mask, S_int_kstep, A_ext_kstep)

        int_baseline = (V_int - z_i)[:, None]
        A_epi_kstep = V_epi_kstep - int_baseline * int_selected_mask.astype(jnp.float32)

        A_epi_kstep = jnp.where(valid_masks, A_epi_kstep, 0.0)

        weighted = A_epi_kstep * lambda_powers[None, :] * valid_masks
        weighted_sum = jnp.sum(weighted, axis=1)
        lambda_sum = jnp.sum(lambda_powers[None, :] * valid_masks, axis=1)
        A_JT = weighted_sum / (lambda_sum + 1e-8)

        ext_selected = A_ext_norm[:, 0] < S_int_norm[:, 0]

        return A_JT, A_ext_kstep[:, 0], S_int_kstep[:, 0], ext_selected, A_ext_norm[:, 0], S_int_norm[:, 0]

    A_JT, A_ext_k0, S_int_k0, ext_selected, A_ext_norm_k0, S_int_norm_k0 = jax.vmap(
        compute_kstep_for_actor_popart, in_axes=1, out_axes=(1, 1, 1, 1, 1, 1)
    )(delta_ext_t, delta_int_t, value_t, V_int_t, z_i_t, done_t)

    A_JT = jnp.clip(A_JT, -adv_clip_abs, adv_clip_abs)

    total_count = T * NUM_ACTORS
    ext_count = jnp.sum(ext_selected.astype(jnp.float32))
    ext_selection_rate = ext_count / total_count

    A_ext_abs_mean = jnp.mean(jnp.abs(A_ext_k0))
    S_int_abs_mean = jnp.mean(jnp.abs(S_int_k0))
    A_ext_std_batch = jnp.std(A_ext_k0)
    S_int_std_batch = jnp.std(S_int_k0)
    scale_ratio = S_int_abs_mean / (A_ext_abs_mean + 1e-8)
    std_ratio = S_int_std_batch / (A_ext_std_batch + 1e-8)

    A_ext_norm_mean = jnp.mean(A_ext_norm_k0)
    S_int_norm_mean = jnp.mean(S_int_norm_k0)
    A_ext_norm_std = jnp.std(A_ext_norm_k0)
    S_int_norm_std = jnp.std(S_int_norm_k0)

    stats = {
        "A_JT_mean": jnp.mean(A_JT),
        "A_JT_std": jnp.std(A_JT),
        "A_JT_min": jnp.min(A_JT),
        "A_JT_max": jnp.max(A_JT),
        "A_ext_kstep_mean": jnp.mean(A_ext_k0),
        "S_int_kstep_mean": jnp.mean(S_int_k0),
        "ext_selection_rate": ext_selection_rate,
        "int_selection_rate": 1.0 - ext_selection_rate,
        "A_ext_abs_mean": A_ext_abs_mean,
        "S_int_abs_mean": S_int_abs_mean,
        "A_ext_std": A_ext_std_batch,
        "S_int_std": S_int_std_batch,
        "scale_ratio_abs": scale_ratio,
        "scale_ratio_std": std_ratio,
        "A_ext_norm_mean": A_ext_norm_mean,
        "S_int_norm_mean": S_int_norm_mean,
        "A_ext_norm_std": A_ext_norm_std,
        "S_int_norm_std": S_int_norm_std,
        "popart_ext_mean": ext_mean,
        "popart_ext_std": ext_std,
        "popart_int_mean": int_mean,
        "popart_int_std": int_std,
    }

    return A_JT, stats


def compute_joint_advantage_kstep_memory_efficient(
    value_t: jnp.ndarray,
    V_int_t: jnp.ndarray,
    delta_ext_t: jnp.ndarray,
    delta_int_t: jnp.ndarray,
    z_i_t: jnp.ndarray,
    done_t: jnp.ndarray,
    gamma: float,
    lmbda: float,
    adv_clip_abs: float = 20.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    T, NUM_ACTORS = delta_ext_t.shape

    lambda_powers = jnp.power(lmbda, jnp.arange(T))

    def scan_body(carry, inputs):
        acc_ext, acc_int, weighted_sum, lambda_sum = carry

        delta_ext, delta_int, v_ext, V_int, z_i, done, k_from_end = inputs
        not_done = 1.0 - done

        new_acc_ext = delta_ext + gamma * not_done * acc_ext
        new_acc_int = delta_int + gamma * not_done * acc_int

        G_ext_kstep = v_ext + new_acc_ext
        G_int_kstep = V_int + new_acc_int

        A_ext_kstep = G_ext_kstep - v_ext

        S_int_kstep = G_int_kstep - z_i

        int_selected = (S_int_kstep <= A_ext_kstep)

        V_epi_kstep = jnp.minimum(A_ext_kstep, S_int_kstep)

        int_baseline = V_int - z_i
        A_epi_kstep = V_epi_kstep - int_baseline * int_selected.astype(jnp.float32)

        lambda_weight = lambda_powers[k_from_end.astype(jnp.int32)]
        new_weighted_sum = weighted_sum + lambda_weight * A_epi_kstep * not_done
        new_lambda_sum = lambda_sum + lambda_weight * not_done

        new_acc_ext = jnp.where(done, 0.0, new_acc_ext)
        new_acc_int = jnp.where(done, 0.0, new_acc_int)

        A_JT = new_weighted_sum / (new_lambda_sum + 1e-8)

        ext_selected = (A_ext_kstep < S_int_kstep).astype(jnp.float32)

        new_carry = (new_acc_ext, new_acc_int, new_weighted_sum, new_lambda_sum)
        return new_carry, (A_JT, ext_selected)

    init_acc_ext = jnp.zeros(NUM_ACTORS)
    init_acc_int = jnp.zeros(NUM_ACTORS)
    init_weighted_sum = jnp.zeros(NUM_ACTORS)
    init_lambda_sum = jnp.zeros(NUM_ACTORS)
    init_carry = (init_acc_ext, init_acc_int, init_weighted_sum, init_lambda_sum)

    k_from_end = jnp.arange(T)[::-1].astype(jnp.float32)

    inputs = (delta_ext_t, delta_int_t, value_t, V_int_t, z_i_t, done_t,
              jnp.broadcast_to(k_from_end[:, None], (T, NUM_ACTORS)))

    _, (A_JT, ext_selected) = jax.lax.scan(scan_body, init_carry, inputs, reverse=True)

    A_JT = jnp.clip(A_JT, -adv_clip_abs, adv_clip_abs)

    ext_selection_rate = jnp.mean(ext_selected)
    stats = {
        "A_JT_mean": jnp.mean(A_JT),
        "A_JT_std": jnp.std(A_JT),
        "A_JT_min": jnp.min(A_JT),
        "A_JT_max": jnp.max(A_JT),
        "ext_selection_rate": ext_selection_rate,
        "int_selection_rate": 1.0 - ext_selection_rate,
    }

    return A_JT, stats
