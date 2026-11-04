import jax.numpy as jnp


def compute_delta_int_with_v_int(
    r_int: jnp.ndarray,
    v_int: jnp.ndarray,
    v_int_next: jnp.ndarray,
    gamma_int: float,
    dones: jnp.ndarray,
) -> jnp.ndarray:
    delta_int = r_int + gamma_int * v_int_next * (1 - dones) - v_int
    return delta_int
