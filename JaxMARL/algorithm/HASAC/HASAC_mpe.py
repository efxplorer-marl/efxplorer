import os
import sys
import logging

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, JaxMARLWrapper
import wandb
import yaml
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
from time import time

class MPEWorldStateWrapper(JaxMARLWrapper):
    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(len(self._env.agents), axis=0)
        return all_obs

    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])


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
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        action_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)
        return hidden, pi, action_logits


class PerActionCriticRNN(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, other_actions_onehot, dones = x

        critic_input = jnp.concatenate([world_state, other_actions_onehot], axis=-1)

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(critic_input)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)

        q_init_scale = float(self.config.get("Q_INIT_SCALE", 0.01))
        q_values = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(q_init_scale),
            bias_init=constant(0.0)
        )(critic)

        return hidden, q_values


class ReplayBufferState(NamedTuple):
    obs: jnp.ndarray
    world_state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    next_world_state: jnp.ndarray
    done: jnp.ndarray
    pos: jnp.ndarray
    size: jnp.ndarray


def create_replay_buffer(config, obs_dim, world_state_dim, num_actors):
    buffer_size = config["BUFFER_SIZE"]

    return ReplayBufferState(
        obs=jnp.zeros((buffer_size, num_actors, obs_dim)),
        world_state=jnp.zeros((buffer_size, num_actors, world_state_dim)),
        action=jnp.zeros((buffer_size, num_actors), dtype=jnp.int32),
        reward=jnp.zeros((buffer_size, num_actors)),
        next_obs=jnp.zeros((buffer_size, num_actors, obs_dim)),
        next_world_state=jnp.zeros((buffer_size, num_actors, world_state_dim)),
        done=jnp.zeros((buffer_size, num_actors), dtype=jnp.bool_),
        pos=jnp.array(0),
        size=jnp.array(0),
    )


def add_to_buffer(buffer_state, transition):
    obs, world_state, action, reward, next_obs, next_world_state, done = transition
    buffer_size = buffer_state.obs.shape[0]

    pos = buffer_state.pos

    new_buffer = ReplayBufferState(
        obs=buffer_state.obs.at[pos].set(obs),
        world_state=buffer_state.world_state.at[pos].set(world_state),
        action=buffer_state.action.at[pos].set(action),
        reward=buffer_state.reward.at[pos].set(reward),
        next_obs=buffer_state.next_obs.at[pos].set(next_obs),
        next_world_state=buffer_state.next_world_state.at[pos].set(next_world_state),
        done=buffer_state.done.at[pos].set(done),
        pos=(pos + 1) % buffer_size,
        size=jnp.minimum(buffer_state.size + 1, buffer_size),
    )
    return new_buffer


def sample_from_buffer(buffer_state, rng, batch_size):
    size = buffer_state.size

    indices = jax.random.randint(rng, (batch_size,), 0, size)

    batch = {
        "obs": buffer_state.obs[indices],
        "world_state": buffer_state.world_state[indices],
        "action": buffer_state.action[indices],
        "reward": buffer_state.reward[indices],
        "next_obs": buffer_state.next_obs[indices],
        "next_world_state": buffer_state.next_world_state[indices],
        "done": buffer_state.done[indices],
    }
    return batch


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    next_obs: jnp.ndarray
    next_world_state: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((len(agent_list), num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def get_other_actions_onehot(actions_onehot, agent_idx, num_agents, action_dim):
    rolled = jnp.roll(actions_onehot, -agent_idx, axis=1)
    other_actions = rolled[:, 1:, :]
    return other_actions.reshape(actions_onehot.shape[0], -1)


def make_train(config):
    env_kwargs = config.get("ENV_KWARGS", {})
    env_name = config["ENV_NAME"]

    env = jaxmarl.make(env_name, **env_kwargs)
    num_agents = len(env.agents)
    num_envs = config["NUM_ENVS"]
    config["NUM_ACTORS"] = num_agents * num_envs
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // num_envs
    )

    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]

    env = MPEWorldStateWrapper(env)
    env = MPELogWrapper(env)
    world_state_dim = env.world_state_size()

    num_actors = config["NUM_ACTORS"]
    num_steps = config["NUM_STEPS"]
    buffer_size = config["BUFFER_SIZE"]
    buffer_batch_size = config["BUFFER_BATCH_SIZE"]
    learning_starts = config["LEARNING_STARTS"]
    gamma = config["GAMMA"]
    tau = config["TAU"]
    alpha_init = config["ALPHA_INIT"]
    fixed_alpha = config["FIXED_ALPHA"]
    target_entropy_ratio = config["TARGET_ENTROPY_RATIO"]
    max_grad_norm = config["MAX_GRAD_NORM"]
    num_epochs = config["NUM_EPOCHS"]
    use_advantage_norm = config.get("USE_ADVANTAGE_NORM", True)

    lr_linear_decay = config.get("LR_LINEAR_DECAY", False)
    if lr_linear_decay:
        total_training_steps = config["NUM_UPDATES"] * num_epochs
        actor_lr = optax.linear_schedule(
            init_value=config["ACTOR_LR"],
            end_value=1e-10,
            transition_steps=total_training_steps,
        )
        critic_lr = optax.linear_schedule(
            init_value=config["CRITIC_LR"],
            end_value=1e-10,
            transition_steps=total_training_steps,
        )
        alpha_lr = optax.linear_schedule(
            init_value=config["ALPHA_LR"],
            end_value=1e-10,
            transition_steps=total_training_steps,
        )
    else:
        actor_lr = config["ACTOR_LR"]
        critic_lr = config["CRITIC_LR"]
        alpha_lr = config["ALPHA_LR"]

    max_entropy = jnp.log(action_dim)
    target_entropy = target_entropy_ratio * max_entropy

    def train(rng):
        actor_network = ActorRNN(action_dim, config=config)
        critic1_network = PerActionCriticRNN(action_dim, config=config)
        critic2_network = PerActionCriticRNN(action_dim, config=config)

        rng, rng_actor, rng_critic1, rng_critic2 = jax.random.split(rng, 4)

        ac_init_x = (
            jnp.zeros((1, num_envs, obs_dim)),
            jnp.zeros((1, num_envs)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(num_envs, config["GRU_HIDDEN_DIM"])
        actor_params = actor_network.init(rng_actor, ac_init_hstate, ac_init_x)

        other_actions_dim = (num_agents - 1) * action_dim
        cr_init_x = (
            jnp.zeros((1, num_envs, world_state_dim)),
            jnp.zeros((1, num_envs, other_actions_dim)),
            jnp.zeros((1, num_envs)),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(num_envs, config["GRU_HIDDEN_DIM"])
        critic1_params = critic1_network.init(rng_critic1, cr_init_hstate, cr_init_x)
        critic2_params = critic2_network.init(rng_critic2, cr_init_hstate, cr_init_x)

        critic1_target_params = critic1_params
        critic2_target_params = critic2_params

        actor_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(actor_lr, eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(critic_lr, eps=1e-5),
        )
        alpha_tx = optax.adam(alpha_lr, eps=1e-5)

        actor_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_params,
            tx=actor_tx,
        )
        critic1_state = TrainState.create(
            apply_fn=critic1_network.apply,
            params=critic1_params,
            tx=critic_tx,
        )
        critic2_state = TrainState.create(
            apply_fn=critic2_network.apply,
            params=critic2_params,
            tx=critic_tx,
        )

        log_alpha = jnp.log(alpha_init)
        alpha_opt_state = alpha_tx.init(log_alpha)

        rng, rng_reset = jax.random.split(rng)
        reset_rng = jax.random.split(rng_reset, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        ac_hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])

        buffer_state = create_replay_buffer(config, obs_dim, world_state_dim, num_actors)

        train_states = (actor_state, critic1_state, critic2_state)
        target_params = (critic1_target_params, critic2_target_params)
        alpha_state = (log_alpha, alpha_opt_state)

        def _update_step(runner_state, unused):
            (train_states, target_params, alpha_state,
             env_state, last_obs, last_done, ac_hstate, buffer_state, rng, update_step) = runner_state

            actor_state, critic1_state, critic2_state = train_states
            critic1_target_params, critic2_target_params = target_params
            log_alpha, alpha_opt_state = alpha_state
            alpha = jnp.exp(log_alpha)

            def _env_step(step_state, unused):
                env_state, last_obs, last_done, ac_hstate, rng = step_state

                rng, rng_action = jax.random.split(rng)

                obs_batch = batchify(last_obs, env.agents, num_actors)
                world_state = last_obs["world_state"].swapaxes(0, 1).reshape((num_actors, -1))

                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                new_ac_hstate, pi, _ = actor_network.apply(actor_state.params, ac_hstate, ac_in)

                action = pi.sample(seed=rng_action)
                action = action.squeeze(0)

                env_act = unbatchify(action, env.agents, num_envs, num_agents)
                rng, rng_step = jax.random.split(rng)
                rng_steps = jax.random.split(rng_step, num_envs)
                next_obsv, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_steps, env_state, env_act
                )

                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)
                done_batch = batchify(done, env.agents, num_actors).squeeze()
                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                next_obs_batch = batchify(next_obsv, env.agents, num_actors)
                next_world_state = next_obsv["world_state"].swapaxes(0, 1).reshape((num_actors, -1))

                transition = Transition(
                    global_done=jnp.tile(done["__all__"], num_agents),
                    done=last_done,
                    action=action,
                    reward=reward_batch,
                    obs=obs_batch,
                    world_state=world_state,
                    next_obs=next_obs_batch,
                    next_world_state=next_world_state,
                    info=info,
                )

                step_state = (next_env_state, next_obsv, done_batch, new_ac_hstate, rng)
                return step_state, transition

            step_state = (env_state, last_obs, last_done, ac_hstate, rng)
            step_state, traj_batch = jax.lax.scan(_env_step, step_state, None, num_steps)
            env_state, last_obs, last_done, ac_hstate, rng = step_state

            def _add_transitions(buffer_state, transition):
                trans = (
                    transition.obs,
                    transition.world_state,
                    transition.action,
                    transition.reward,
                    transition.next_obs,
                    transition.next_world_state,
                    transition.global_done,
                )
                return add_to_buffer(buffer_state, trans), None

            buffer_state, _ = jax.lax.scan(_add_transitions, buffer_state, traj_batch)

            total_timesteps = update_step * num_envs * num_steps
            can_learn = total_timesteps >= learning_starts

            def _do_learning(state):
                train_states, target_params, alpha_state, buffer_state, rng = state
                actor_state, critic1_state, critic2_state = train_states
                critic1_target_params, critic2_target_params = target_params
                log_alpha, alpha_opt_state = alpha_state
                alpha = jnp.exp(log_alpha)

                critic_loss_acc = jnp.array(0.0)
                actor_loss_acc = jnp.array(0.0)
                alpha_loss_acc = jnp.array(0.0)
                entropy_acc = jnp.array(0.0)
                q_mean_acc = jnp.array(0.0)

                def _update_epoch(carry, unused):
                    (actor_state, critic1_state, critic2_state,
                     critic1_target_params, critic2_target_params,
                     log_alpha, alpha_opt_state, rng,
                     critic_loss, actor_loss, alpha_loss, entropy, q_mean) = carry

                    alpha = jnp.exp(log_alpha)

                    rng, rng_sample = jax.random.split(rng)
                    batch = sample_from_buffer(buffer_state, rng_sample, buffer_batch_size)

                    batch_size = buffer_batch_size
                    obs = batch["obs"].reshape(batch_size, num_agents, num_envs, obs_dim)
                    world_state = batch["world_state"].reshape(batch_size, num_agents, num_envs, world_state_dim)
                    action = batch["action"].reshape(batch_size, num_agents, num_envs)
                    reward = batch["reward"].reshape(batch_size, num_agents, num_envs)
                    next_obs = batch["next_obs"].reshape(batch_size, num_agents, num_envs, obs_dim)
                    next_world_state = batch["next_world_state"].reshape(batch_size, num_agents, num_envs, world_state_dim)
                    done = batch["done"].reshape(batch_size, num_agents, num_envs)

                    agent_idx = 0

                    flat_obs = obs[:, agent_idx, :, :].reshape(batch_size * num_envs, obs_dim)
                    flat_world_state = world_state[:, agent_idx, :, :].reshape(batch_size * num_envs, world_state_dim)
                    flat_action = action[:, agent_idx, :].reshape(batch_size * num_envs)
                    flat_reward = reward[:, agent_idx, :].reshape(batch_size * num_envs)
                    flat_next_obs = next_obs[:, agent_idx, :, :].reshape(batch_size * num_envs, obs_dim)
                    flat_next_world_state = next_world_state[:, agent_idx, :, :].reshape(batch_size * num_envs, world_state_dim)
                    flat_done = done[:, agent_idx, :].reshape(batch_size * num_envs)

                    all_actions_onehot = jax.nn.one_hot(
                        action.reshape(batch_size * num_envs, num_agents),
                        action_dim
                    )

                    other_actions = get_other_actions_onehot(
                        all_actions_onehot, agent_idx, num_agents, action_dim
                    )

                    dummy_done = jnp.zeros(batch_size * num_envs)

                    cr_hstate = ScannedRNN.initialize_carry(batch_size * num_envs, config["GRU_HIDDEN_DIM"])

                    def _critic_loss_fn(critic_params, target_critic1_params, target_critic2_params):
                        cr_input = (
                            flat_world_state[np.newaxis, :],
                            other_actions[np.newaxis, :],
                            dummy_done[np.newaxis, :],
                        )
                        _, q_values = critic1_network.apply(critic_params, cr_hstate, cr_input)
                        q_values = q_values.squeeze(0)

                        q_taken = q_values[jnp.arange(batch_size * num_envs), flat_action]

                        ac_hstate_fresh = ScannedRNN.initialize_carry(batch_size * num_envs, config["GRU_HIDDEN_DIM"])
                        ac_input = (flat_next_obs[np.newaxis, :], dummy_done[np.newaxis, :])
                        _, next_pi, _ = actor_network.apply(actor_state.params, ac_hstate_fresh, ac_input)
                        next_probs = next_pi.probs
                        next_probs = next_probs.squeeze(0)

                        next_other_actions = other_actions

                        next_cr_input = (
                            flat_next_world_state[np.newaxis, :],
                            next_other_actions[np.newaxis, :],
                            dummy_done[np.newaxis, :],
                        )
                        _, target_q1 = critic1_network.apply(target_critic1_params, cr_hstate, next_cr_input)
                        _, target_q2 = critic2_network.apply(target_critic2_params, cr_hstate, next_cr_input)
                        target_q1 = target_q1.squeeze(0)
                        target_q2 = target_q2.squeeze(0)

                        target_q = jnp.minimum(target_q1, target_q2)

                        next_log_probs = jnp.log(next_probs + 1e-8)
                        expected_q = jnp.sum(next_probs * (target_q - alpha * next_log_probs), axis=-1)

                        td_target = flat_reward + gamma * (1 - flat_done) * expected_q
                        td_target = jax.lax.stop_gradient(td_target)

                        td_error = q_taken - td_target

                        critic_loss = jnp.mean(td_error ** 2)

                        return critic_loss, {"td_error": td_error, "q_mean": q_taken.mean()}

                    critic1_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    (c1_loss, c1_info), c1_grads = critic1_grad_fn(
                        critic1_state.params, critic1_target_params, critic2_target_params
                    )

                    def _critic2_loss_fn(critic_params, target_critic1_params, target_critic2_params):
                        cr_input = (
                            flat_world_state[np.newaxis, :],
                            other_actions[np.newaxis, :],
                            dummy_done[np.newaxis, :],
                        )
                        _, q_values = critic2_network.apply(critic_params, cr_hstate, cr_input)
                        q_values = q_values.squeeze(0)
                        q_taken = q_values[jnp.arange(batch_size * num_envs), flat_action]

                        ac_hstate_fresh = ScannedRNN.initialize_carry(batch_size * num_envs, config["GRU_HIDDEN_DIM"])
                        ac_input = (flat_next_obs[np.newaxis, :], dummy_done[np.newaxis, :])
                        _, next_pi, _ = actor_network.apply(actor_state.params, ac_hstate_fresh, ac_input)
                        next_probs = next_pi.probs.squeeze(0)

                        next_other_actions = other_actions
                        next_cr_input = (
                            flat_next_world_state[np.newaxis, :],
                            next_other_actions[np.newaxis, :],
                            dummy_done[np.newaxis, :],
                        )
                        _, target_q1 = critic1_network.apply(target_critic1_params, cr_hstate, next_cr_input)
                        _, target_q2 = critic2_network.apply(target_critic2_params, cr_hstate, next_cr_input)
                        target_q = jnp.minimum(target_q1.squeeze(0), target_q2.squeeze(0))

                        next_log_probs = jnp.log(next_probs + 1e-8)
                        expected_q = jnp.sum(next_probs * (target_q - alpha * next_log_probs), axis=-1)
                        td_target = flat_reward + gamma * (1 - flat_done) * expected_q
                        td_target = jax.lax.stop_gradient(td_target)

                        critic_loss = jnp.mean((q_taken - td_target) ** 2)
                        return critic_loss, {}

                    critic2_grad_fn = jax.value_and_grad(_critic2_loss_fn, has_aux=True)
                    (c2_loss, _), c2_grads = critic2_grad_fn(
                        critic2_state.params, critic1_target_params, critic2_target_params
                    )

                    critic1_state = critic1_state.apply_gradients(grads=c1_grads)
                    critic2_state = critic2_state.apply_gradients(grads=c2_grads)

                    def _actor_loss_fn(actor_params):
                        ac_hstate_fresh = ScannedRNN.initialize_carry(batch_size * num_envs, config["GRU_HIDDEN_DIM"])
                        ac_input = (flat_obs[np.newaxis, :], dummy_done[np.newaxis, :])
                        _, pi, _ = actor_network.apply(actor_params, ac_hstate_fresh, ac_input)
                        probs = pi.probs.squeeze(0)
                        log_probs = jnp.log(probs + 1e-8)

                        cr_input = (
                            flat_world_state[np.newaxis, :],
                            other_actions[np.newaxis, :],
                            dummy_done[np.newaxis, :],
                        )
                        _, q1 = critic1_network.apply(critic1_state.params, cr_hstate, cr_input)
                        _, q2 = critic2_network.apply(critic2_state.params, cr_hstate, cr_input)
                        q_values = jnp.minimum(q1.squeeze(0), q2.squeeze(0))

                        if use_advantage_norm:
                            q_mean = jnp.sum(probs * q_values, axis=-1, keepdims=True)
                            q_std = jnp.sqrt(jnp.sum(probs * (q_values - q_mean) ** 2, axis=-1, keepdims=True) + 1e-8)
                            q_normalized = (q_values - q_mean) / q_std
                        else:
                            q_normalized = q_values

                        actor_loss = jnp.mean(
                            jnp.sum(probs * (alpha * log_probs - q_normalized), axis=-1)
                        )

                        entropy = -jnp.sum(probs * log_probs, axis=-1).mean()

                        return actor_loss, {"entropy": entropy}

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    (a_loss, a_info), a_grads = actor_grad_fn(actor_state.params)
                    actor_state = actor_state.apply_gradients(grads=a_grads)

                    def _alpha_loss_fn(log_alpha_val):
                        alpha_val = jnp.exp(log_alpha_val)
                        entropy_diff = a_info["entropy"] - target_entropy
                        return log_alpha_val * jax.lax.stop_gradient(entropy_diff)

                    def _update_alpha(state):
                        log_alpha_val, opt_state = state
                        al_loss, al_grads = jax.value_and_grad(_alpha_loss_fn)(log_alpha_val)
                        updates, new_opt_state = alpha_tx.update(al_grads, opt_state)
                        new_log_alpha = optax.apply_updates(log_alpha_val, updates)
                        alpha_min = config.get("ALPHA_MIN", 0.001)
                        alpha_max = config.get("ALPHA_MAX", 1.0)
                        new_log_alpha = jnp.clip(new_log_alpha, jnp.log(alpha_min), jnp.log(alpha_max))
                        return (new_log_alpha, new_opt_state), al_loss

                    def _no_update_alpha(state):
                        return state, jnp.array(0.0)

                    (log_alpha, alpha_opt_state), al_loss = jax.lax.cond(
                        fixed_alpha,
                        _no_update_alpha,
                        _update_alpha,
                        (log_alpha, alpha_opt_state)
                    )

                    critic1_target_params = jax.tree.map(
                        lambda t, s: tau * s + (1 - tau) * t,
                        critic1_target_params,
                        critic1_state.params
                    )
                    critic2_target_params = jax.tree.map(
                        lambda t, s: tau * s + (1 - tau) * t,
                        critic2_target_params,
                        critic2_state.params
                    )

                    critic_loss = critic_loss + (c1_loss + c2_loss) / 2
                    actor_loss = actor_loss + a_loss
                    alpha_loss = alpha_loss + al_loss
                    entropy = entropy + a_info["entropy"]
                    q_mean = q_mean + c1_info["q_mean"]

                    carry = (actor_state, critic1_state, critic2_state,
                             critic1_target_params, critic2_target_params,
                             log_alpha, alpha_opt_state, rng,
                             critic_loss, actor_loss, alpha_loss, entropy, q_mean)
                    return carry, None

                init_carry = (actor_state, critic1_state, critic2_state,
                              critic1_target_params, critic2_target_params,
                              log_alpha, alpha_opt_state, rng,
                              critic_loss_acc, actor_loss_acc, alpha_loss_acc, entropy_acc, q_mean_acc)

                final_carry, _ = jax.lax.scan(_update_epoch, init_carry, None, num_epochs)

                (actor_state, critic1_state, critic2_state,
                 critic1_target_params, critic2_target_params,
                 log_alpha, alpha_opt_state, rng,
                 critic_loss, actor_loss, alpha_loss, entropy, q_mean) = final_carry

                critic_loss = critic_loss / num_epochs
                actor_loss = actor_loss / num_epochs
                alpha_loss = alpha_loss / num_epochs
                entropy = entropy / num_epochs
                q_mean = q_mean / num_epochs

                train_states = (actor_state, critic1_state, critic2_state)
                target_params = (critic1_target_params, critic2_target_params)
                alpha_state = (log_alpha, alpha_opt_state)

                loss_info = {
                    "critic_loss": critic_loss,
                    "actor_loss": actor_loss,
                    "alpha_loss": alpha_loss,
                    "entropy": entropy,
                    "alpha": jnp.exp(log_alpha),
                    "q_mean": q_mean,
                }

                return (train_states, target_params, alpha_state, rng), loss_info

            def _skip_learning(state):
                train_states, target_params, alpha_state, buffer_state, rng = state
                log_alpha, _ = alpha_state
                loss_info = {
                    "critic_loss": jnp.array(0.0),
                    "actor_loss": jnp.array(0.0),
                    "alpha_loss": jnp.array(0.0),
                    "entropy": jnp.array(0.0),
                    "alpha": jnp.exp(log_alpha),
                    "q_mean": jnp.array(0.0),
                }
                return (train_states, target_params, alpha_state, rng), loss_info

            learn_state = (train_states, target_params, alpha_state, buffer_state, rng)
            (train_states, target_params, alpha_state, rng), loss_info = jax.lax.cond(
                can_learn,
                _do_learning,
                _skip_learning,
                learn_state
            )

            mean_reward = jnp.mean(traj_batch.reward)

            metric = {
                "returns": traj_batch.info["returned_episode_returns"][-1, :].mean(),
                "mean_reward": mean_reward,
                "reward/extrinsic": mean_reward,
                "env_step": update_step * num_envs * num_steps,
                "buffer_size": buffer_state.size,
                **loss_info,
            }

            def _print_progress(m):
                step = int(m["env_step"])
                returns = float(m["returns"])
                entropy = float(m["entropy"])
                alpha = float(m["alpha"])
                critic_loss = float(m["critic_loss"])
                actor_loss = float(m["actor_loss"])
                buf_size = int(m["buffer_size"])
                mean_reward = float(m["mean_reward"])
                q_mean = float(m.get("q_mean", 0.0))
                update = step // (num_envs * num_steps)
                total_updates = int(config["NUM_UPDATES"])
                progress_pct = 100 * update / total_updates
                if update % 10 == 0 or update == total_updates:
                    print(f"[{update:5d}/{total_updates} {progress_pct:5.1f}%] step={step:9d} | "
                          f"ret={returns:8.2f} r={mean_reward:6.3f} | "
                          f"ent={entropy:.3f} alpha={alpha:.4f} | "
                          f"c={critic_loss:7.3f} a={actor_loss:7.3f} Q={q_mean:7.2f} | buf={buf_size}")

            jax.debug.callback(_print_progress, metric)

            if config.get("WANDB_MODE") != "disabled":
                jax.debug.callback(lambda m: wandb.log(m), metric)

            runner_state = (train_states, target_params, alpha_state,
                           env_state, last_obs, last_done, ac_hstate, buffer_state, rng, update_step + 1)
            return runner_state, metric

        rng, rng_run = jax.random.split(rng)
        init_done = jnp.zeros((num_actors,), dtype=bool)

        runner_state = (
            train_states, target_params, alpha_state,
            env_state, obsv, init_done, ac_hstate, buffer_state, rng_run, 0
        )

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def get_config_name():
    config_name = "HASAC_corridor"
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith("--config=") or arg.startswith("config="):
            config_name = arg.split("=")[1]
        elif arg.startswith("--") and "=" in arg:
            new_arg = arg[2:]
            new_argv.append(new_arg)
        else:
            new_argv.append(arg)
    sys.argv = new_argv
    return config_name


CONFIG_NAME = get_config_name()


@hydra.main(version_base=None, config_path="config", config_name=CONFIG_NAME)
def main(config):
    config = OmegaConf.to_container(config)

    print("=" * 80)
    print("SAC-Discrete for Multi-Agent Particle Environments")
    print("=" * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Training Steps: {config['TOTAL_TIMESTEPS']:.0f}")
    print(f"Alpha Init: {config['ALPHA_INIT']}, Fixed: {config['FIXED_ALPHA']}")
    print(f"Use Advantage Norm: {config.get('USE_ADVANTAGE_NORM', True)}")
    print("=" * 80)
    print()

    wandb_project = config.get("PROJECT", "JaxMARL")
    env_display_name = config["ENV_NAME"].replace("/", "-").replace("_", "-")
    run_name = config.get("RUN_NAME") or f"SAC-Discrete-{env_display_name}"

    run = wandb.init(
        
        project=wandb_project,
        name=run_name,
        
        tags=["SAC-Discrete", "Off-Policy", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    print("Starting training...")

    try:
        with jax.disable_jit(False):
            train_jit = jax.jit(make_train(config))
            start_time = time()
            out = train_jit(rng)
            elapsed_time = time() - start_time
            print()
            print("=" * 80)
            print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
            print("=" * 80)
    except KeyboardInterrupt:
        print("\n> Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n> Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    train_states = out["runner_state"][0]
    actor_params = train_states[0].params
    critic1_params = train_states[1].params
    critic2_params = train_states[2].params

    if config.get("WANDB_MODE") != "disabled" and hasattr(run, "dir"):
        save_dir = os.path.join(run.dir, "params")
        os.makedirs(save_dir, exist_ok=True)

        save_file(flatten_dict(actor_params, sep=','), os.path.join(save_dir, "actor_params.safetensors"))
        save_file(flatten_dict(critic1_params, sep=','), os.path.join(save_dir, "critic1_params.safetensors"))
        save_file(flatten_dict(critic2_params, sep=','), os.path.join(save_dir, "critic2_params.safetensors"))

        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Checkpoints saved to {save_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
