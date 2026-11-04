from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax

from .networks import ScannedRNN


class IntrinsicCriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        s_encoder_embeddings, dones = x

        if s_encoder_embeddings.ndim == 2:
            s_encoder_embeddings = s_encoder_embeddings[None, :, :]
            dones = dones[None, :]
            squeeze_output = True
        else:
            squeeze_output = False

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(s_encoder_embeddings)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        v_int = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )(embedding)
        v_int = nn.relu(v_int)
        v_int = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v_int)
        v_int = jnp.squeeze(v_int, axis=-1)

        if squeeze_output:
            v_int = v_int.squeeze(0)

        return hidden, v_int


def create_intrinsic_critic_train_state(
    rng: jax.random.PRNGKey,
    config: Dict,
    s_encoder_output_dim: int,
    num_actors: int,
    lr_schedule: Optional[callable] = None,
) -> Tuple["IntrinsicCriticRNN", TrainState]:
    cr_init_x = (
        jnp.zeros((1, num_actors, s_encoder_output_dim)),
        jnp.zeros((1, num_actors)),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])

    intrinsic_critic_network = IntrinsicCriticRNN(config=config)
    intrinsic_critic_params = intrinsic_critic_network.init(rng, cr_init_hstate, cr_init_x)

    v_int_lr = config.get("V_INT_LR", config.get("CRITIC_LR", 0.0003))

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(v_int_lr, eps=1e-5),
    )

    intrinsic_critic_train_state = TrainState.create(
        apply_fn=intrinsic_critic_network.apply,
        params=intrinsic_critic_params,
        tx=tx,
    )
    return intrinsic_critic_network, intrinsic_critic_train_state


class IntrinsicCriticRNN_ZAug(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        s_encoder_embeddings, z, dones = x

        if s_encoder_embeddings.ndim == 2:
            s_encoder_embeddings = s_encoder_embeddings[None, :, :]
            z = z[None, :]
            dones = dones[None, :]
            squeeze_output = True
        else:
            squeeze_output = False

        z_expanded = z[..., None]

        combined = jnp.concatenate([s_encoder_embeddings, z_expanded], axis=-1)

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(combined)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        v_int = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )(embedding)
        v_int = nn.relu(v_int)
        v_int = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v_int)
        v_int = jnp.squeeze(v_int, axis=-1)

        if squeeze_output:
            v_int = v_int.squeeze(0)

        return hidden, v_int


def create_intrinsic_critic_z_aug_train_state(
    rng: jax.random.PRNGKey,
    config: Dict,
    s_encoder_output_dim: int,
    num_actors: int,
    lr_schedule: Optional[callable] = None,
) -> Tuple[IntrinsicCriticRNN_ZAug, TrainState]:
    cr_init_x = (
        jnp.zeros((1, num_actors, s_encoder_output_dim)),
        jnp.zeros((1, num_actors)),
        jnp.zeros((1, num_actors)),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])

    intrinsic_critic_z_aug_network = IntrinsicCriticRNN_ZAug(config=config)
    intrinsic_critic_z_aug_params = intrinsic_critic_z_aug_network.init(rng, cr_init_hstate, cr_init_x)

    v_int_lr = config.get("V_INT_LR", config.get("CRITIC_LR", 0.0003))

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(v_int_lr, eps=1e-5),
    )

    intrinsic_critic_z_aug_train_state = TrainState.create(
        apply_fn=intrinsic_critic_z_aug_network.apply,
        params=intrinsic_critic_z_aug_params,
        tx=tx,
    )
    return intrinsic_critic_z_aug_network, intrinsic_critic_z_aug_train_state


