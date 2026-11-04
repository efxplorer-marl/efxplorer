import functools
from typing import Dict, Sequence, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
import optax


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(rnn_state.shape[0], rnn_state.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi, action_logits


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)


class ActorCriticRNN(nn.Module):
    """Shared GRU backbone with separate actor logits and extrinsic value heads (IPPO-style)."""

    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, action_logits, jnp.squeeze(critic, axis=-1)


class IntrinsicQRNN(nn.Module):
    config: Dict
    action_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        agent_local_features, dones = x

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(agent_local_features)
        embedding = nn.relu(embedding)

        if embedding.ndim == 2:
            embedding = embedding[None, :, :]
            dones = dones[None, :]
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)
            embedding = embedding.squeeze(0)
        else:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_int = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        q_int = nn.relu(q_int)
        q_int = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(
            q_int
        )

        return hidden, q_int


class QExtRNN(nn.Module):
    config: Dict
    num_agents: int
    action_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        world_state, actions, dones = x

        if world_state.ndim == 2:
            world_state = world_state[None, :, :]
            actions = actions[None, :, :]
            dones = dones[None, :]
            squeeze_output = True
        else:
            squeeze_output = False

        seq_len, batch_size = world_state.shape[:2]

        ws_embed = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        ws_embed = nn.relu(ws_embed)

        actions_one_hot = jax.nn.one_hot(actions, self.action_dim)
        actions_flat = actions_one_hot.reshape(seq_len, batch_size, -1)
        act_embed = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actions_flat)
        act_embed = nn.relu(act_embed)

        combined = jnp.concatenate([ws_embed, act_embed], axis=-1)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(combined)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_out = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        q_out = nn.relu(q_out)
        q_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(q_out)
        q_out = jnp.squeeze(q_out, axis=-1)

        if squeeze_output:
            q_out = q_out.squeeze(0)

        return hidden, q_out


def create_actor_train_state(
    rng: jax.random.PRNGKey,
    config: Dict,
    action_dim: int,
    obs_dim: int,
    num_agents: int,
    lr_schedule: Optional[callable] = None,
    shared_params: bool = True,
) -> Tuple[Optional[ActorRNN], Optional[TrainState], Optional[List[ActorRNN]], Optional[List[TrainState]]]:
    num_envs = config["NUM_ENVS"]
    ac_init_x = (
        jnp.zeros((1, num_envs, obs_dim)),
        jnp.zeros((1, num_envs)),
    )
    ac_init_hstate = ScannedRNN.initialize_carry(num_envs, config["GRU_HIDDEN_DIM"])

    def _make_tx():
        lr = lr_schedule if lr_schedule is not None else config["ACTOR_LR"]
        return optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=1e-5),
        )

    if shared_params:
        actor_network = ActorRNN(action_dim, config=config)
        actor_params = actor_network.init(rng, ac_init_hstate, ac_init_x)
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_params,
            tx=_make_tx(),
        )
        return actor_network, actor_train_state, None, None
    else:
        actor_networks = [ActorRNN(action_dim, config=config) for _ in range(num_agents)]
        actor_rngs = list(jax.random.split(rng, num_agents))
        actor_params_list = [
            actor_networks[i].init(actor_rngs[i], ac_init_hstate, ac_init_x)
            for i in range(num_agents)
        ]
        actor_train_states = [
            TrainState.create(
                apply_fn=actor_networks[i].apply,
                params=actor_params_list[i],
                tx=_make_tx(),
            )
            for i in range(num_agents)
        ]
        return None, None, actor_networks, actor_train_states


def create_critic_train_state(
    rng: jax.random.PRNGKey,
    config: Dict,
    world_state_size: int,
    lr_schedule: Optional[callable] = None,
) -> Tuple[CriticRNN, TrainState]:
    num_envs = config["NUM_ENVS"]
    cr_init_x = (
        jnp.zeros((1, num_envs, world_state_size)),
        jnp.zeros((1, num_envs)),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(num_envs, config["GRU_HIDDEN_DIM"])

    critic_network = CriticRNN(config=config)
    critic_params = critic_network.init(rng, cr_init_hstate, cr_init_x)

    lr = lr_schedule if lr_schedule is not None else config["CRITIC_LR"]
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=lr, eps=1e-5),
    )

    critic_train_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_params,
        tx=tx,
    )
    return critic_network, critic_train_state


def create_actor_critic_train_state(
    rng: jax.random.PRNGKey,
    config: Dict,
    action_dim: int,
    obs_dim: int,
    lr_schedule: Optional[callable] = None,
) -> Tuple[ActorCriticRNN, TrainState]:
    """Single TrainState over fused ActorCriticRNN (local obs). Uses ``NUM_ACTORS`` batch axis."""
    num_batch = int(config["NUM_ACTORS"])
    init_x = (
        jnp.zeros((1, num_batch, obs_dim)),
        jnp.zeros((1, num_batch)),
    )
    init_hstate = ScannedRNN.initialize_carry(num_batch, config["GRU_HIDDEN_DIM"])

    network = ActorCriticRNN(action_dim, config=config)
    network_params = network.init(rng, init_hstate, init_x)

    lr = lr_schedule if lr_schedule is not None else config.get("LR", config.get("ACTOR_LR", 0.001))
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    return network, train_state


def create_intrinsic_q_train_states(
    rng: jax.random.PRNGKey,
    config: Dict,
    action_dim: int,
    num_agents: int,
    input_dim: int = 2,
    shared_params: bool = True,
) -> Tuple[Union[TrainState, List[TrainState]], Union[TrainState, List[TrainState]]]:
    num_envs = config["NUM_ENVS"]
    init_hstate = ScannedRNN.initialize_carry(num_envs, config["GRU_HIDDEN_DIM"])
    init_x = (
        jnp.zeros((1, num_envs, input_dim)),
        jnp.zeros((1, num_envs)),
    )

    intrinsic_q_net_lr = config.get("INTRINSIC_Q_NET_LR", 1e-3)

    def _make_tx():
        return optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(intrinsic_q_net_lr, eps=1e-5),
        )

    if shared_params:
        network = IntrinsicQRNN(config=config, action_dim=action_dim)
        params = network.init(rng, init_hstate, init_x)

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=_make_tx(),
        )
        target_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=optax.identity(),
        )
        return train_state, target_state
    else:
        rngs = list(jax.random.split(rng, num_agents))
        networks = [IntrinsicQRNN(config=config, action_dim=action_dim) for _ in range(num_agents)]
        params_list = [networks[i].init(rngs[i], init_hstate, init_x) for i in range(num_agents)]

        train_states = [
            TrainState.create(
                apply_fn=networks[i].apply,
                params=params_list[i],
                tx=_make_tx(),
            )
            for i in range(num_agents)
        ]
        target_states = [
            TrainState.create(
                apply_fn=networks[i].apply,
                params=params_list[i],
                tx=optax.identity(),
            )
            for i in range(num_agents)
        ]
        return train_states, target_states


def create_q_ext_train_state(
    rng: jax.random.PRNGKey,
    config: Dict,
    world_state_size: int,
    num_agents: int,
    action_dim: int,
) -> Tuple[QExtRNN, TrainState]:
    num_envs = config["NUM_ENVS"]

    init_hstate = ScannedRNN.initialize_carry(num_envs, config["GRU_HIDDEN_DIM"])
    init_x = (
        jnp.zeros((1, num_envs, world_state_size)),
        jnp.zeros((1, num_envs, num_agents), dtype=jnp.int32),
        jnp.zeros((1, num_envs)),
    )

    q_ext_network = QExtRNN(config=config, num_agents=num_agents, action_dim=action_dim)
    q_ext_params = q_ext_network.init(rng, init_hstate, init_x)

    q_ext_lr = config.get("Q_EXT_LR", 0.0005)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(q_ext_lr, eps=1e-5),
    )

    q_ext_train_state = TrainState.create(
        apply_fn=q_ext_network.apply,
        params=q_ext_params,
        tx=tx,
    )

    return q_ext_network, q_ext_train_state


def compute_coma_advantage(
    q_ext_network: QExtRNN,
    q_ext_params: Dict,
    world_state: jnp.ndarray,
    actions: jnp.ndarray,
    action_probs: jnp.ndarray,
    dones: jnp.ndarray,
    hidden: jnp.ndarray,
    num_agents: int,
    action_dim: int,
) -> jnp.ndarray:
    batch_size = world_state.shape[0]

    _, q_actual = q_ext_network.apply(q_ext_params, hidden, (world_state, actions, dones))

    advantages = []
    for agent_i in range(num_agents):
        q_counterfactual = jnp.zeros(batch_size)

        for action_a in range(action_dim):
            actions_cf = actions.at[:, agent_i].set(action_a)
            _, q_cf = q_ext_network.apply(q_ext_params, hidden, (world_state, actions_cf, dones))
            q_counterfactual += action_probs[:, agent_i, action_a] * q_cf

        advantage_i = q_actual - q_counterfactual
        advantages.append(advantage_i)

    return jnp.stack(advantages, axis=1)


def apply_coma_adaptive_scale(
    delta_int: jnp.ndarray,
    coma_advantages: jnp.ndarray,
    base_scale: float,
    coma_boost: float,
    num_agents: int,
    num_envs: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if coma_advantages.ndim == 2:
        num_steps = coma_advantages.shape[0]
        coma_per_agent = coma_advantages.reshape(num_steps, num_agents, num_envs)
        coma_mean = coma_per_agent.mean(axis=(0, 2))
    else:
        coma_mean = coma_advantages

    coma_min = coma_mean.min()
    coma_max = coma_mean.max()
    coma_normalized = (coma_mean - coma_min) / (coma_max - coma_min + 1e-8)

    scale_factor = 1.0 - coma_normalized

    agent_scales = base_scale * (1.0 + coma_boost * scale_factor)

    agent_scales_expanded = jnp.tile(agent_scales[:, None], (1, num_envs)).ravel()

    delta_int_scaled = delta_int * agent_scales_expanded[None, :]

    return delta_int_scaled, agent_scales
