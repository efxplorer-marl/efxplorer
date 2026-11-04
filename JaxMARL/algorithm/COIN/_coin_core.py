from __future__ import annotations

import copy
import os
from functools import partial
from typing import Any, Callable, Tuple

import chex
import flax.linen as nn
import flashbax as fbx
import hydra  # noqa: F401  (entrypoint scripts re-import)
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import CTRolloutManager

from .coin_modules.predictors import (
    COINPredictorTrainState,
    create_coin_predictor_states,
    get_log_pi,
)


class ScannedRNN(nn.Module):

    @partial(
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
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, rnn_output = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(rnn_output)

        return hidden, q_vals, rnn_output


class HyperNetwork(nn.Module):
    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        return x


class MixingNetwork(nn.Module):
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, q_vals, states):
        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0))

        w_1 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim * n_agents,
            init_scale=self.init_scale,
        )(states)
        b_1 = nn.Dense(
            self.embedding_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(states)
        w_2 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim,
            init_scale=self.init_scale,
        )(states)
        b_2 = HyperNetwork(
            hidden_dim=self.embedding_dim, output_dim=1, init_scale=self.init_scale
        )(states)

        w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)

        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot = jnp.matmul(hidden, w_2) + b_2

        return q_tot.squeeze()


@chex.dataclass(frozen=True)
class COINTimestep:
    obs: dict
    actions: dict
    actions_onehot: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class COINTrainState(TrainState):
    target_network_params: Any
    predict_os_state: COINPredictorTrainState
    predict_with_state: COINPredictorTrainState
    predict_without_state: COINPredictorTrainState
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    beta1: float = 1.0
    beta2: float = 1.0


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    n_agents = env.num_agents
    n_actions = env.action_space(env.agents[0]).n

    _steps_per_update = int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])
    if config.get("EPS_ANNEAL_ENV_STEPS") is not None:
        _eps_trans = max(1, int(config["EPS_ANNEAL_ENV_STEPS"]))
        _eps_use_timesteps = True
    else:
        _eps_trans = max(
            1, int(float(config["EPS_DECAY"]) * float(config["NUM_UPDATES"]))
        )
        _eps_use_timesteps = False
    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=_eps_trans,
    )

    def _decay_interval_updates(env_steps_key: str, updates_key: str, default_u: int) -> int:
        if config.get(env_steps_key) is not None:
            return max(1, int(config[env_steps_key]) // _steps_per_update)
        return int(config.get(updates_key, default_u))

    influence_decay_iu = _decay_interval_updates(
        "INFLUENCE_DECAY_ENV_STEPS", "INFLUENCE_DECAY_INTERVAL", 10000
    )
    curiosity_decay_iu = _decay_interval_updates(
        "CURIOSITY_DECAY_ENV_STEPS", "CURIOSITY_DECAY_INTERVAL", 10000
    )

    _test_interval_es = int(config["TEST_INTERVAL_ENV_STEPS"]) if config.get(
        "TEST_INTERVAL_ENV_STEPS"
    ) is not None else 0
    _log_interval_es = int(
        config.get("LOG_INTERVAL_ENV_STEPS")
        or config.get("RUNNER_LOG_INTERVAL_ENV_STEPS")
        or config.get("LEARNER_LOG_INTERVAL_ENV_STEPS")
        or 0
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return chosen_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):

        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"])

        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: wrapped_env.batch_sample(key_a[i], agent)
                for i, agent in enumerate(env.agents)
            }
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )

            actions_onehot = {agent: jax.nn.one_hot(actions[agent], n_actions) for agent in env.agents}

            timestep = COINTimestep(
                obs=obs,
                actions=actions,
                actions_onehot=actions_onehot,
                rewards=rewards,
                dones=dones,
                avail_actions=avail_actions,
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj)

        network = RNNQNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config["HIDDEN_SIZE"],
        )

        mixer = MixingNetwork(
            config["MIXER_EMBEDDING_DIM"],
            config["MIXER_HYPERNET_HIDDEN_DIM"],
            config["MIXER_INIT_SCALE"],
        )

        obs_size = wrapped_env.obs_size
        state_size = sample_traj.obs["__all__"].shape[-1]

        def create_agent(rng):
            init_x = (
                jnp.zeros((1, 1, wrapped_env.obs_size)),
                jnp.zeros((1, 1)),
            )
            init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)
            agent_params = network.init(rng, init_hs, *init_x)

            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros((len(env.agents), 1, 1))
            init_state = jnp.zeros((1, 1, state_size))
            mixer_params = mixer.init(_rng, init_x, init_state)

            network_params = {'agent': agent_params, 'mixer': mixer_params}

            rng, _rng = jax.random.split(rng)
            predict_os_state, predict_with_state, predict_without_state = create_coin_predictor_states(
                _rng,
                config,
                n_agents=n_agents,
                rnn_hidden_dim=config["HIDDEN_SIZE"],
                n_actions=n_actions,
                obs_dim=obs_size,
                state_dim=state_size,
            )

            lr_scheduler = optax.linear_schedule(
                init_value=config["LR"],
                end_value=1e-10,
                transition_steps=(config["NUM_EPOCHS"]) * config["NUM_UPDATES"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            optim = str(config.get("OPTIMIZER", "radam")).lower()
            if optim == "rmsprop":
                inner = optax.rmsprop(
                    lr,
                    decay=float(config.get("RMSPROP_DECAY", 0.99)),
                    eps=float(config.get("RMSPROP_EPS", 1e-5)),
                )
            else:
                inner = optax.radam(learning_rate=lr)
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), inner)

            train_state = COINTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                target_network_params=network_params,
                tx=tx,
                predict_os_state=predict_os_state,
                predict_with_state=predict_with_state,
                predict_without_state=predict_without_state,
                beta1=config.get("COIN_BETA1", 1.0),
                beta2=config.get("COIN_BETA2", 1.0),
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        def _update_step(runner_state, unused):

            train_state, buffer_state, test_state, rng = runner_state

            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals, _ = jax.vmap(
                    network.apply, in_axes=(None, 0, 0, 0)
                )(
                    train_state.params['agent'],
                    hs,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(axis=1)

                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)

                _eps_count = (
                    train_state.timesteps
                    if _eps_use_timesteps
                    else train_state.n_updates
                )
                eps = eps_scheduler(_eps_count)
                _rngs = jax.random.split(rng_a, env.num_agents)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                actions_dict = unbatchify(actions)

                actions_onehot = jax.vmap(lambda a: jax.nn.one_hot(a, n_actions))(actions)
                actions_onehot_dict = unbatchify(actions_onehot)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions_dict
                )

                timestep = COINTimestep(
                    obs=last_obs,
                    actions=actions_dict,
                    actions_onehot=actions_onehot_dict,
                    rewards=jax.tree.map(lambda x: config.get("REW_SCALE", 1) * x, rewards),
                    dones=last_dones,
                    avail_actions=avail_actions,
                )
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            expl_state = (init_hs, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            _ts_old = train_state.timesteps
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            _ts_new = train_state.timesteps

            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis],
                timesteps,
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            def _learn_phase(carry, _):

                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1),
                    minibatch,
                )

                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                _actions_onehot = batchify(minibatch.actions_onehot)
                _avail_actions = batchify(minibatch.avail_actions)

                _, q_next_target, _ = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.target_network_params['agent'],
                    init_hs,
                    _obs,
                    _dones,
                )

                _, _, rnn_embeddings = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    train_state.params['agent'],
                    init_hs,
                    _obs,
                    _dones,
                )
                hidden_for_coin = jnp.transpose(rnn_embeddings, (1, 0, 2, 3))

                hidden_transposed = jnp.transpose(hidden_for_coin, (2, 0, 1, 3))
                actions_transposed = jnp.transpose(_actions_onehot, (2, 1, 0, 3))

                bs = config["BUFFER_BATCH_SIZE"]
                time_steps = hidden_transposed.shape[1]

                hidden_action = jnp.concatenate([hidden_transposed, actions_transposed], axis=-1)
                os_input = hidden_action.reshape(bs, time_steps, -1)

                os_input_flat = os_input.reshape(-1, os_input.shape[-1])
                pred_obs_flat, pred_state_flat = train_state.predict_os_state.apply_fn(
                    train_state.predict_os_state.target_params, os_input_flat
                )
                pred_obs = pred_obs_flat.reshape(bs, time_steps, -1)
                pred_state = pred_state_flat.reshape(bs, time_steps, -1)

                obs_next_all = jnp.transpose(_obs, (2, 1, 0, 3))
                obs_next_flat = obs_next_all.reshape(bs, time_steps, -1)
                state_all = minibatch.obs["__all__"]
                state_next = jnp.transpose(state_all, (1, 0, 2))

                obs_error = jnp.sum(jnp.square(pred_obs[:, :-1] - obs_next_flat[:, 1:]), axis=-1)
                state_error = jnp.sum(jnp.square(pred_state[:, :-1] - state_next[:, 1:]), axis=-1)
                curiosity_reward = (obs_error + state_error) * train_state.beta2

                obs_next_for_inf = jnp.transpose(_obs, (1, 0, 2, 3))

                def compute_influence_for_agent_i(i):
                    influence_i = jnp.zeros((time_steps - 1, bs))

                    def add_influence_from_k(k_and_inf, _):
                        k, inf_accum = k_and_inf

                        def compute_single_k(inf, k_val):
                            should_skip = k_val == i

                            h_k = hidden_for_coin[:-1, k_val, :, :]
                            a_k = jnp.transpose(_actions_onehot, (1, 0, 2, 3))[:-1, k_val, :, :]
                            a_i = jnp.transpose(_actions_onehot, (1, 0, 2, 3))[:-1, i, :, :]
                            o_k_next = obs_next_for_inf[1:, k_val, :, :]

                            h_k_flat = h_k.reshape(-1, h_k.shape[-1])
                            a_k_flat = a_k.reshape(-1, a_k.shape[-1])
                            a_i_flat = a_i.reshape(-1, a_i.shape[-1])
                            o_k_next_flat = o_k_next.reshape(-1, o_k_next.shape[-1])

                            input_without = jnp.concatenate([h_k_flat, a_k_flat], axis=-1)
                            pred_without = train_state.predict_without_state.apply_fn(
                                train_state.predict_without_state.target_params, input_without
                            )

                            input_with = jnp.concatenate([h_k_flat, a_k_flat, a_i_flat], axis=-1)
                            pred_with = train_state.predict_with_state.apply_fn(
                                train_state.predict_with_state.target_params, input_with
                            )

                            log_p = get_log_pi(pred_without, o_k_next_flat)
                            log_q = get_log_pi(pred_with, o_k_next_flat)

                            log_p = log_p.reshape(time_steps - 1, bs)
                            log_q = log_q.reshape(time_steps - 1, bs)

                            contribution = config.get("COIN_BETA", 1.0) * log_q - log_p
                            contribution = jnp.where(should_skip, 0.0, contribution)

                            return inf + contribution, None

                        new_inf, _ = compute_single_k(inf_accum, k)
                        return (k + 1, new_inf), None

                    (_, final_influence), _ = jax.lax.scan(
                        add_influence_from_k,
                        (0, influence_i),
                        None,
                        length=n_agents
                    )

                    return final_influence * train_state.beta1

                influence_rewards = jax.vmap(compute_influence_for_agent_i)(jnp.arange(n_agents))
                influence_rewards = jnp.transpose(influence_rewards, (1, 2, 0))

                def _loss_fn(params):
                    _, q_vals, _ = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                        params['agent'],
                        init_hs,
                        _obs,
                        _dones,
                    )

                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        _actions[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)

                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals = q_vals - (unavailable_actions * 1e10)

                    q_next = jnp.take_along_axis(
                        q_next_target,
                        jnp.argmax(valid_q_vals, axis=-1)[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)

                    influence_transposed = jnp.transpose(influence_rewards, (2, 0, 1))

                    chosen_q_adjusted = chosen_action_q_vals[:, :-1, :] - influence_transposed

                    qmix = mixer.apply(params['mixer'], chosen_q_adjusted, minibatch.obs["__all__"][:-1])

                    qmix_next = mixer.apply(
                        train_state.target_network_params['mixer'],
                        q_next[:, 1:, :],
                        minibatch.obs["__all__"][1:]
                    )

                    curiosity_transposed = jnp.transpose(curiosity_reward, (1, 0))

                    qmix_target = (
                        minibatch.rewards["__all__"][:-1]
                        + curiosity_transposed
                        + (1 - minibatch.dones["__all__"][:-1])
                        * config["GAMMA"]
                        * qmix_next
                    )

                    loss = jnp.mean((qmix - jax.lax.stop_gradient(qmix_target)) ** 2)

                    return loss, (chosen_action_q_vals.mean(), curiosity_reward.mean(), influence_rewards.mean())

                (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)

                def _update_predictors(train_state):
                    def os_loss_fn(os_params):
                        pred_obs_flat, pred_state_flat = train_state.predict_os_state.apply_fn(
                            os_params, os_input_flat
                        )
                        pred_obs_up = pred_obs_flat.reshape(bs, time_steps, -1)
                        pred_state_up = pred_state_flat.reshape(bs, time_steps, -1)
                        obs_err = jnp.mean(jnp.square(pred_obs_up[:, :-1] - obs_next_flat[:, 1:]))
                        state_err = jnp.mean(jnp.square(pred_state_up[:, :-1] - state_next[:, 1:]))
                        return obs_err + state_err

                    os_loss, os_grads = jax.value_and_grad(os_loss_fn)(train_state.predict_os_state.params)
                    new_os_state = train_state.predict_os_state.apply_gradients(grads=os_grads)

                    def influence_predictor_loss_fn(with_params, without_params):
                        p_o_loss = 0.0
                        q_o_loss = 0.0

                        for i in range(n_agents):
                            for k in range(n_agents):
                                if k == i:
                                    continue

                                h_k = hidden_for_coin[:-1, k, :, :].reshape(-1, config["HIDDEN_SIZE"])
                                a_k = jnp.transpose(_actions_onehot, (1, 0, 2, 3))[:-1, k, :, :].reshape(-1, n_actions)
                                a_i = jnp.transpose(_actions_onehot, (1, 0, 2, 3))[:-1, i, :, :].reshape(-1, n_actions)
                                o_k_next = obs_next_for_inf[1:, k, :, :].reshape(-1, obs_size)

                                input_without = jnp.concatenate([h_k, a_k], axis=-1)
                                pred_without = train_state.predict_without_state.apply_fn(without_params, input_without)
                                p_o_loss += jnp.mean(jnp.sum(jnp.square(pred_without - o_k_next), axis=-1))

                                input_with = jnp.concatenate([h_k, a_k, a_i], axis=-1)
                                pred_with = train_state.predict_with_state.apply_fn(with_params, input_with)
                                q_o_loss += jnp.mean(jnp.sum(jnp.square(pred_with - o_k_next), axis=-1))

                        return p_o_loss, q_o_loss

                    def with_loss_fn(with_params):
                        _, q_o_loss = influence_predictor_loss_fn(with_params, train_state.predict_without_state.params)
                        return q_o_loss

                    def without_loss_fn(without_params):
                        p_o_loss, _ = influence_predictor_loss_fn(train_state.predict_with_state.params, without_params)
                        return p_o_loss

                    with_loss, with_grads = jax.value_and_grad(with_loss_fn)(train_state.predict_with_state.params)
                    without_loss, without_grads = jax.value_and_grad(without_loss_fn)(train_state.predict_without_state.params)

                    new_with_state = train_state.predict_with_state.apply_gradients(grads=with_grads)
                    new_without_state = train_state.predict_without_state.apply_gradients(grads=without_grads)

                    return new_os_state, new_with_state, new_without_state, os_loss, with_loss, without_loss

                new_os_state, new_with_state, new_without_state, os_loss, with_loss, without_loss = _update_predictors(train_state)

                train_state = train_state.replace(
                    predict_os_state=new_os_state,
                    predict_with_state=new_with_state,
                    predict_without_state=new_without_state,
                    grad_steps=train_state.grad_steps + 1,
                )

                qvals, curiosity_mean, influence_mean = aux
                return (train_state, rng), (loss, qvals, curiosity_mean, influence_mean, os_loss, with_loss, without_loss)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (
                train_state.timesteps > config["LEARNING_STARTS"]
            )

            (train_state, rng), learn_outputs = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config["NUM_EPOCHS"]
                ),
                lambda train_state, rng: (
                    (train_state, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),
                train_state,
                _rng,
            )
            loss, qvals, curiosity_mean, influence_mean, os_loss, with_loss, without_loss = learn_outputs

            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    ),
                    predict_os_state=train_state.predict_os_state.replace(
                        target_params=train_state.predict_os_state.params
                    ),
                    predict_with_state=train_state.predict_with_state.replace(
                        target_params=train_state.predict_with_state.params
                    ),
                    predict_without_state=train_state.predict_without_state.replace(
                        target_params=train_state.predict_without_state.params
                    ),
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            if config.get("INFLUENCE_EXP_DECAY", False):
                decay_interval = influence_decay_iu
                decay_rate = config.get("INFLUENCE_DECAY_RATE", 0.99)
                decay_stop = config.get("INFLUENCE_DECAY_STOP", 0.1)
                train_state = jax.lax.cond(
                    train_state.n_updates % decay_interval == 0,
                    lambda ts: ts.replace(beta1=jnp.maximum(ts.beta1 * decay_rate, decay_stop)),
                    lambda ts: ts,
                    train_state,
                )

            if config.get("CURIOSITY_EXP_DECAY", False):
                decay_interval = curiosity_decay_iu
                decay_rate = config.get("CURIOSITY_DECAY_RATE", 0.99)
                decay_stop = config.get("CURIOSITY_DECAY_STOP", 0.1)
                train_state = jax.lax.cond(
                    train_state.n_updates % decay_interval == 0,
                    lambda ts: ts.replace(beta2=jnp.maximum(ts.beta2 * decay_rate, decay_stop)),
                    lambda ts: ts,
                    train_state,
                )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
                "curiosity_reward": curiosity_mean.mean(),
                "influence_reward": influence_mean.mean(),
                "predictor_os_loss": os_loss.mean(),
                "predictor_with_loss": with_loss.mean(),
                "predictor_without_loss": without_loss.mean(),
                "beta1": train_state.beta1,
                "beta2": train_state.beta2,
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                if _test_interval_es > 0:
                    _do_test = (_ts_new // _test_interval_es) > (_ts_old // _test_interval_es)
                else:
                    _do_test = (
                        train_state.n_updates
                        % int(config["NUM_UPDATES"] * float(config.get("TEST_INTERVAL", 0.05)))
                        == 0
                    )
                test_state = jax.lax.cond(
                    _do_test,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            if config["WANDB_MODE"] != "disabled":

                def callback(do_log, metrics, original_seed):
                    dl = jax.device_get(do_log)
                    if hasattr(dl, "item"):
                        dl = dl.item()
                    if int(dl) == 0:
                        return
                    if config.get('WANDB_LOG_ALL_SEEDS', False):
                        metrics.update(
                            {f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()}
                        )
                    wandb.log(metrics)

                if _log_interval_es > 0:
                    _do_log = jnp.int32(
                        (_ts_new // _log_interval_es) > (_ts_old // _log_interval_es)
                    )
                else:
                    _do_log = jnp.int32(1)
                jax.debug.callback(callback, _do_log, metrics, original_seed)

            runner_state = (train_state, buffer_state, test_state, rng)
            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            params = train_state.params['agent']

            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals, _ = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params,
                    hstate,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config, env_factory: Callable[[dict], Tuple[Any, str]]):

    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "COIN")
    env, env_name = env_factory(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get("WANDB_RUN_NAME") or f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, _ in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config, env_factory: Callable[[dict], Tuple[Any, str]]):

    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "COIN")
    env, env_name = env_factory(copy.deepcopy(default_config))

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [0.005, 0.001, 0.0005, 0.0001, 0.00005]},
            "COIN_BETA1": {"values": [0.1, 0.5, 1.0, 2.0]},
            "COIN_BETA2": {"values": [0.1, 0.5, 1.0, 2.0]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=300)
