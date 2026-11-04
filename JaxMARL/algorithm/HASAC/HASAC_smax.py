

from __future__ import annotations

import os
import sys
import logging
from functools import partial
from time import time
from typing import Any, Dict, NamedTuple, Tuple

import chex
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import SMAXLogWrapper, SMAXWorldStateWrapper


class Actor(nn.Module):

    action_dim: int
    hidden_size: int = 64
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, avail_actions: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        unavail = 1.0 - avail_actions
        logits = logits - unavail * 1e10
        return logits


class JointQCritic(nn.Module):

    hidden_size: int = 64
    num_layers: int = 2

    @nn.compact
    def __call__(
        self, world_state: jnp.ndarray, joint_actions_onehot: jnp.ndarray
    ) -> jnp.ndarray:
        x = jnp.concatenate([world_state, joint_actions_onehot], axis=-1)
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
        q = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(q, axis=-1)


def gumbel_softmax_sample(
    logits: jnp.ndarray,
    rng: jax.Array,
    temperature: float = 1.0,
    hard: bool = True,
) -> jnp.ndarray:
    u = jax.random.uniform(rng, logits.shape, minval=1e-20, maxval=1.0)
    g = -jnp.log(-jnp.log(u))
    y = jax.nn.softmax((logits + g) / temperature, axis=-1)
    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), logits.shape[-1])
        y = jax.lax.stop_gradient(y_hard - y) + y
    return y


def categorical_log_prob(logits: jnp.ndarray, action_onehot: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(action_onehot * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def categorical_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


class ReplayBufferState(NamedTuple):
    obs: jnp.ndarray
    world_state: jnp.ndarray
    action: jnp.ndarray
    avail_actions: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray
    next_world_state: jnp.ndarray
    next_avail_actions: jnp.ndarray
    n_step_return: jnp.ndarray
    n_step_discount: jnp.ndarray
    pos: jnp.ndarray
    size: jnp.ndarray


def create_replay_buffer(
    buffer_size: int,
    num_agents: int,
    obs_dim: int,
    world_state_dim: int,
    action_dim: int,
) -> ReplayBufferState:
    return ReplayBufferState(
        obs=jnp.zeros((buffer_size, num_agents, obs_dim)),
        world_state=jnp.zeros((buffer_size, world_state_dim)),
        action=jnp.zeros((buffer_size, num_agents), dtype=jnp.int32),
        avail_actions=jnp.zeros((buffer_size, num_agents, action_dim)),
        reward=jnp.zeros((buffer_size,)),
        done=jnp.zeros((buffer_size,), dtype=jnp.bool_),
        next_obs=jnp.zeros((buffer_size, num_agents, obs_dim)),
        next_world_state=jnp.zeros((buffer_size, world_state_dim)),
        next_avail_actions=jnp.zeros((buffer_size, num_agents, action_dim)),
        n_step_return=jnp.zeros((buffer_size,)),
        n_step_discount=jnp.zeros((buffer_size,)),
        pos=jnp.array(0),
        size=jnp.array(0),
    )


def add_to_buffer(
    buffer_state: ReplayBufferState, transition: Dict[str, jnp.ndarray]
) -> ReplayBufferState:
    buffer_size = buffer_state.obs.shape[0]
    pos = buffer_state.pos
    return ReplayBufferState(
        obs=buffer_state.obs.at[pos].set(transition["obs"]),
        world_state=buffer_state.world_state.at[pos].set(transition["world_state"]),
        action=buffer_state.action.at[pos].set(transition["action"]),
        avail_actions=buffer_state.avail_actions.at[pos].set(transition["avail_actions"]),
        reward=buffer_state.reward.at[pos].set(transition["reward"]),
        done=buffer_state.done.at[pos].set(transition["done"]),
        next_obs=buffer_state.next_obs.at[pos].set(transition["next_obs"]),
        next_world_state=buffer_state.next_world_state.at[pos].set(transition["next_world_state"]),
        next_avail_actions=buffer_state.next_avail_actions.at[pos].set(transition["next_avail_actions"]),
        n_step_return=buffer_state.n_step_return.at[pos].set(transition["n_step_return"]),
        n_step_discount=buffer_state.n_step_discount.at[pos].set(transition["n_step_discount"]),
        pos=(pos + 1) % buffer_size,
        size=jnp.minimum(buffer_state.size + 1, buffer_size),
    )


def sample_from_buffer(
    buffer_state: ReplayBufferState, rng: jax.Array, batch_size: int
) -> Dict[str, jnp.ndarray]:
    indices = jax.random.randint(rng, (batch_size,), 0, buffer_state.size)
    return {
        "obs": buffer_state.obs[indices],
        "world_state": buffer_state.world_state[indices],
        "action": buffer_state.action[indices],
        "avail_actions": buffer_state.avail_actions[indices],
        "reward": buffer_state.reward[indices],
        "done": buffer_state.done[indices],
        "next_obs": buffer_state.next_obs[indices],
        "next_world_state": buffer_state.next_world_state[indices],
        "next_avail_actions": buffer_state.next_avail_actions[indices],
        "n_step_return": buffer_state.n_step_return[indices],
        "n_step_discount": buffer_state.n_step_discount[indices],
    }


class HASACTrainStates(NamedTuple):
    actor_state: TrainState
    critic1_state: TrainState
    critic2_state: TrainState
    critic1_target_params: Any
    critic2_target_params: Any
    log_alphas: jnp.ndarray
    alpha_opt_states: Any


def make_actor_obs_dim(obs_dim: int, num_agents: int, share_params: bool) -> int:
    return obs_dim + (num_agents if share_params else 0)


def append_agent_id(obs: jnp.ndarray, num_agents: int) -> jnp.ndarray:
    one_hot = jnp.eye(num_agents)
    new_shape = (1,) * (obs.ndim - 2) + one_hot.shape
    one_hot = one_hot.reshape(new_shape)
    one_hot = jnp.broadcast_to(one_hot, obs.shape[:-1] + (num_agents,))
    return jnp.concatenate([obs, one_hot], axis=-1)


def init_train_states(
    rng: jax.Array,
    *,
    num_agents: int,
    obs_dim: int,
    world_state_dim: int,
    action_dim: int,
    config: Dict[str, Any],
) -> Tuple[Actor, JointQCritic, JointQCritic, HASACTrainStates]:
    share_params: bool = bool(config.get("SHARE_PARAMS", False))
    actor_hidden = int(config.get("ACTOR_HIDDEN_SIZE", config.get("HIDDEN_SIZE", 64)))
    critic_hidden = int(config.get("CRITIC_HIDDEN_SIZE", config.get("HIDDEN_SIZE", 64)))
    num_layers = int(config.get("NUM_LAYERS", 2))

    actor = Actor(action_dim=action_dim, hidden_size=actor_hidden, num_layers=num_layers)
    critic1 = JointQCritic(hidden_size=critic_hidden, num_layers=num_layers)
    critic2 = JointQCritic(hidden_size=critic_hidden, num_layers=num_layers)

    actor_obs_dim = make_actor_obs_dim(obs_dim, num_agents, share_params)

    actor_lr = float(config["ACTOR_LR"])
    critic_lr = float(config["CRITIC_LR"])
    alpha_lr = float(config["ALPHA_LR"])
    max_grad_norm = float(config["MAX_GRAD_NORM"])

    actor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )
    alpha_tx = optax.adam(alpha_lr, eps=1e-5)

    rng, rng_a, rng_c1, rng_c2 = jax.random.split(rng, 4)

    dummy_obs = jnp.zeros((1, actor_obs_dim))
    dummy_avail = jnp.ones((1, action_dim))

    if share_params:
        actor_params = actor.init(rng_a, dummy_obs, dummy_avail)
        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_tx)
    else:
        rngs_per_agent = jax.random.split(rng_a, num_agents)
        actor_params = jax.vmap(lambda r: actor.init(r, dummy_obs, dummy_avail))(rngs_per_agent)
        actor_opt_state = jax.vmap(actor_tx.init)(actor_params)
        actor_state = TrainState(
            step=jnp.array(0),
            apply_fn=actor.apply,
            params=actor_params,
            tx=actor_tx,
            opt_state=actor_opt_state,
        )

    dummy_ws = jnp.zeros((1, world_state_dim))
    dummy_joint = jnp.zeros((1, num_agents * action_dim))
    critic1_params = critic1.init(rng_c1, dummy_ws, dummy_joint)
    critic2_params = critic2.init(rng_c2, dummy_ws, dummy_joint)
    critic1_state = TrainState.create(apply_fn=critic1.apply, params=critic1_params, tx=critic_tx)
    critic2_state = TrainState.create(apply_fn=critic2.apply, params=critic2_params, tx=critic_tx)

    log_alpha_init = jnp.log(jnp.full((num_agents,), float(config["ALPHA_INIT"])))
    alpha_opt_states = jax.vmap(alpha_tx.init)(log_alpha_init)

    states = HASACTrainStates(
        actor_state=actor_state,
        critic1_state=critic1_state,
        critic2_state=critic2_state,
        critic1_target_params=critic1_params,
        critic2_target_params=critic2_params,
        log_alphas=log_alpha_init,
        alpha_opt_states=alpha_opt_states,
    )
    return actor, critic1, critic2, states


def _prepare_actor_obs(obs_per_agent: jnp.ndarray, num_agents: int, share_params: bool) -> jnp.ndarray:
    if share_params:
        return append_agent_id(obs_per_agent, num_agents)
    return obs_per_agent


def sample_all_agents(
    actor: Actor,
    actor_params: Any,
    obs: jnp.ndarray,
    avail: jnp.ndarray,
    rng: jax.Array,
    *,
    num_agents: int,
    share_params: bool,
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    actor_obs = _prepare_actor_obs(obs, num_agents, share_params)
    rngs = jax.random.split(rng, num_agents)

    def per_agent(params, o, av, r):
        logits = actor.apply(params, o, av)
        action_oh = gumbel_softmax_sample(logits, r, temperature=temperature, hard=True)
        log_prob = categorical_log_prob(logits, action_oh)
        return action_oh, log_prob

    if share_params:
        v = jax.vmap(per_agent, in_axes=(None, 1, 1, 0), out_axes=(1, 1))
        action_oh, log_prob = v(actor_params, actor_obs, avail, rngs)
    else:
        v = jax.vmap(per_agent, in_axes=(0, 1, 1, 0), out_axes=(1, 1))
        action_oh, log_prob = v(actor_params, actor_obs, avail, rngs)
    return action_oh, log_prob


def get_focal_actor_params(actor_state: TrainState, agent_id: jnp.ndarray, share_params: bool):
    if share_params:
        return actor_state.params
    return jax.tree.map(lambda x: x[agent_id], actor_state.params)


def get_focal_actor_opt_state(actor_state: TrainState, agent_id: jnp.ndarray, share_params: bool):
    if share_params:
        return actor_state.opt_state
    return jax.tree.map(lambda x: x[agent_id], actor_state.opt_state)


def write_focal_actor(
    actor_state: TrainState,
    agent_id: jnp.ndarray,
    new_focal_params: Any,
    new_focal_opt_state: Any,
    share_params: bool,
) -> TrainState:
    if share_params:
        return actor_state.replace(
            step=actor_state.step + 1,
            params=new_focal_params,
            opt_state=new_focal_opt_state,
        )
    new_params = jax.tree.map(
        lambda full, sl: full.at[agent_id].set(sl), actor_state.params, new_focal_params
    )
    new_opt = jax.tree.map(
        lambda full, sl: full.at[agent_id].set(sl), actor_state.opt_state, new_focal_opt_state
    )
    return actor_state.replace(step=actor_state.step + 1, params=new_params, opt_state=new_opt)


def sac_update(
    states: HASACTrainStates,
    batch: Dict[str, jnp.ndarray],
    rng: jax.Array,
    *,
    actor: Actor,
    critic1: JointQCritic,
    critic2: JointQCritic,
    config: Dict[str, Any],
    num_agents: int,
    action_dim: int,
) -> Tuple[HASACTrainStates, Dict[str, jnp.ndarray]]:

    share_params: bool = bool(config.get("SHARE_PARAMS", False))
    fixed_alpha: bool = bool(config["FIXED_ALPHA"])
    fixed_order: bool = bool(config.get("FIXED_ORDER", False))
    gamma: float = float(config["GAMMA"])
    tau: float = float(config["TAU"])
    alpha_min: float = float(config.get("ALPHA_MIN", 0.001))
    alpha_max: float = float(config.get("ALPHA_MAX", 1.0))
    target_entropy_ratio: float = float(config["TARGET_ENTROPY_RATIO"])
    target_entropy = target_entropy_ratio * float(np.log(action_dim))
    gumbel_temp: float = float(config.get("GUMBEL_TEMP", 1.0))
    use_huber: bool = bool(config.get("USE_HUBER", True))
    huber_delta: float = float(config.get("HUBER_DELTA", 10.0))

    alpha_lr = float(config["ALPHA_LR"])
    alpha_tx = optax.adam(alpha_lr, eps=1e-5)

    obs = batch["obs"]
    world_state = batch["world_state"]
    action = batch["action"]
    avail = batch["avail_actions"]
    next_obs = batch["next_obs"]
    next_world_state = batch["next_world_state"]
    next_avail = batch["next_avail_actions"]
    n_step_return = batch["n_step_return"]
    n_step_discount = batch["n_step_discount"]

    B = obs.shape[0]
    actions_taken_onehot = jax.nn.one_hot(action, action_dim)
    joint_actions_taken = actions_taken_onehot.reshape(B, num_agents * action_dim)

    rng, rng_next = jax.random.split(rng)
    next_joint_oh, next_log_probs = sample_all_agents(
        actor,
        states.actor_state.params,
        next_obs,
        next_avail,
        rng_next,
        num_agents=num_agents,
        share_params=share_params,
        temperature=gumbel_temp,
    )
    next_joint_oh = jax.lax.stop_gradient(next_joint_oh)
    next_log_probs = jax.lax.stop_gradient(next_log_probs)
    next_joint_flat = next_joint_oh.reshape(B, num_agents * action_dim)

    target_q1 = critic1.apply(states.critic1_target_params, next_world_state, next_joint_flat)
    target_q2 = critic2.apply(states.critic2_target_params, next_world_state, next_joint_flat)
    target_q_min = jnp.minimum(target_q1, target_q2)

    alphas = jnp.exp(states.log_alphas)
    entropy_term = jnp.sum(alphas[None, :] * next_log_probs, axis=-1)
    soft_target = target_q_min - entropy_term
    td_target = n_step_return + n_step_discount * soft_target
    td_target = jax.lax.stop_gradient(td_target)

    def _critic_loss_fn(params, network: JointQCritic):
        q = network.apply(params, world_state, joint_actions_taken)
        if use_huber:
            loss = optax.huber_loss(q, td_target, delta=huber_delta).mean()
        else:
            loss = jnp.mean((q - td_target) ** 2)
        return loss, q.mean()

    (c1_loss, c1_q_mean), c1_grads = jax.value_and_grad(
        lambda p: _critic_loss_fn(p, critic1), has_aux=True
    )(states.critic1_state.params)
    (c2_loss, _), c2_grads = jax.value_and_grad(
        lambda p: _critic_loss_fn(p, critic2), has_aux=True
    )(states.critic2_state.params)

    new_critic1_state = states.critic1_state.apply_gradients(grads=c1_grads)
    new_critic2_state = states.critic2_state.apply_gradients(grads=c2_grads)

    rng, rng_pre = jax.random.split(rng)
    current_joint_oh, current_log_probs = sample_all_agents(
        actor,
        states.actor_state.params,
        obs,
        avail,
        rng_pre,
        num_agents=num_agents,
        share_params=share_params,
        temperature=gumbel_temp,
    )
    current_joint_oh = jax.lax.stop_gradient(current_joint_oh)

    rng, rng_perm = jax.random.split(rng)
    if fixed_order:
        agent_order = jnp.arange(num_agents)
    else:
        agent_order = jax.random.permutation(rng_perm, num_agents)

    def _one_agent_step(carry, agent_id):
        actor_state, log_alphas, alpha_opt_states, joint_oh, rng = carry

        rng, rng_grad, rng_resample, rng_alpha = jax.random.split(rng, 4)

        focal_obs = obs[:, agent_id, :]
        focal_avail = avail[:, agent_id, :]
        focal_actor_obs = focal_obs
        if share_params:
            agent_id_oh = jax.nn.one_hot(agent_id, num_agents)
            focal_actor_obs = jnp.concatenate(
                [focal_obs, jnp.broadcast_to(agent_id_oh, (B, num_agents))], axis=-1
            )

        focal_params = get_focal_actor_params(actor_state, agent_id, share_params)
        focal_opt_state = get_focal_actor_opt_state(actor_state, agent_id, share_params)

        def actor_loss_fn(focal_params):
            logits = actor.apply(focal_params, focal_actor_obs, focal_avail)
            action_oh = gumbel_softmax_sample(
                logits, rng_grad, temperature=gumbel_temp, hard=True
            )
            log_prob = categorical_log_prob(logits, action_oh)
            new_joint = joint_oh.at[:, agent_id, :].set(action_oh)
            joint_flat = new_joint.reshape(B, num_agents * action_dim)
            q1 = critic1.apply(new_critic1_state.params, world_state, joint_flat)
            q2 = critic2.apply(new_critic2_state.params, world_state, joint_flat)
            q_min = jnp.minimum(q1, q2)
            alpha_focal = jnp.exp(log_alphas[agent_id])
            loss = jnp.mean(alpha_focal * log_prob - q_min)
            aux = {
                "log_prob_mean": jnp.mean(log_prob),
                "entropy": -jnp.mean(log_prob),
                "q_mean": q_min.mean(),
                "actor_loss": loss,
            }
            return loss, aux

        (a_loss, aux), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(focal_params)

        updates, new_focal_opt_state = actor_state.tx.update(
            grads, focal_opt_state, focal_params
        )
        new_focal_params = optax.apply_updates(focal_params, updates)
        new_actor_state = write_focal_actor(
            actor_state, agent_id, new_focal_params, new_focal_opt_state, share_params
        )

        new_focal_for_resample = get_focal_actor_params(new_actor_state, agent_id, share_params)
        new_logits = actor.apply(new_focal_for_resample, focal_actor_obs, focal_avail)
        new_focal_action = jax.lax.stop_gradient(
            gumbel_softmax_sample(new_logits, rng_resample, temperature=gumbel_temp, hard=True)
        )
        new_joint_oh = joint_oh.at[:, agent_id, :].set(new_focal_action)

        log_alpha_focal = log_alphas[agent_id]
        focal_alpha_opt = jax.tree.map(lambda x: x[agent_id], alpha_opt_states)

        def alpha_loss_fn(log_alpha_val):
            log_prob_const = jax.lax.stop_gradient(aux["log_prob_mean"])
            return -log_alpha_val * (log_prob_const + target_entropy)

        al_loss, al_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha_focal)

        def _do_alpha(carry):
            log_alpha_val, opt_state = carry
            updates, new_opt = alpha_tx.update(al_grad, opt_state)
            new_log_alpha = optax.apply_updates(log_alpha_val, updates)
            new_log_alpha = jnp.clip(
                new_log_alpha, jnp.log(alpha_min), jnp.log(alpha_max)
            )
            return new_log_alpha, new_opt

        def _skip_alpha(carry):
            return carry

        new_log_alpha_focal, new_focal_alpha_opt = jax.lax.cond(
            fixed_alpha, _skip_alpha, _do_alpha, (log_alpha_focal, focal_alpha_opt)
        )
        new_log_alphas = log_alphas.at[agent_id].set(new_log_alpha_focal)
        new_alpha_opt_states = jax.tree.map(
            lambda full, sl: full.at[agent_id].set(sl), alpha_opt_states, new_focal_alpha_opt
        )

        new_carry = (
            new_actor_state,
            new_log_alphas,
            new_alpha_opt_states,
            new_joint_oh,
            rng,
        )
        scan_out = {
            "actor_loss": aux["actor_loss"],
            "entropy": aux["entropy"],
            "q_mean": aux["q_mean"],
            "alpha_loss": al_loss,
            "alpha": jnp.exp(new_log_alpha_focal),
        }
        return new_carry, scan_out

    init_carry = (
        states.actor_state,
        states.log_alphas,
        states.alpha_opt_states,
        current_joint_oh,
        rng,
    )
    final_carry, scan_out = jax.lax.scan(_one_agent_step, init_carry, agent_order)
    new_actor_state, new_log_alphas, new_alpha_opt_states, _, rng = final_carry

    new_critic1_target = jax.tree.map(
        lambda t, s: tau * s + (1.0 - tau) * t,
        states.critic1_target_params,
        new_critic1_state.params,
    )
    new_critic2_target = jax.tree.map(
        lambda t, s: tau * s + (1.0 - tau) * t,
        states.critic2_target_params,
        new_critic2_state.params,
    )

    new_states = HASACTrainStates(
        actor_state=new_actor_state,
        critic1_state=new_critic1_state,
        critic2_state=new_critic2_state,
        critic1_target_params=new_critic1_target,
        critic2_target_params=new_critic2_target,
        log_alphas=new_log_alphas,
        alpha_opt_states=new_alpha_opt_states,
    )

    info = {
        "critic1_loss": c1_loss,
        "critic2_loss": c2_loss,
        "critic_loss": 0.5 * (c1_loss + c2_loss),
        "td_target_mean": td_target.mean(),
        "q_taken_mean": c1_q_mean,
        "actor_loss": jnp.mean(scan_out["actor_loss"]),
        "entropy_per_agent_mean": jnp.mean(scan_out["entropy"]),
        "q_pi_mean": jnp.mean(scan_out["q_mean"]),
        "alpha_loss": jnp.mean(scan_out["alpha_loss"]),
        "alpha_mean": jnp.mean(scan_out["alpha"]),
        "alpha_min": jnp.min(scan_out["alpha"]),
        "alpha_max": jnp.max(scan_out["alpha"]),
    }
    return new_states, info


def acting_logits_per_agent(
    actor: Actor,
    actor_params: Any,
    obs: jnp.ndarray,
    avail: jnp.ndarray,
    *,
    num_agents: int,
    share_params: bool,
) -> jnp.ndarray:
    actor_obs = _prepare_actor_obs(obs, num_agents, share_params)
    if share_params:
        v = jax.vmap(actor.apply, in_axes=(None, 1, 1), out_axes=1)
    else:
        v = jax.vmap(actor.apply, in_axes=(0, 1, 1), out_axes=1)
    return v(actor_params, actor_obs, avail)


def _stack_per_agent(d: Dict[str, jnp.ndarray], agent_list) -> jnp.ndarray:
    stacked = jnp.stack([d[a] for a in agent_list], axis=0)
    return jnp.swapaxes(stacked, 0, 1)


def _unbatch_actions(action_be: jnp.ndarray, agent_list):
    return {a: action_be[:, i] for i, a in enumerate(agent_list)}


def compute_n_step(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
    n: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    T = rewards.shape[0]
    pad_r = jnp.pad(rewards, ((0, n), (0, 0)))
    pad_d = jnp.pad(dones.astype(jnp.float32), ((0, n), (0, 0)))

    n_step_return = jnp.zeros_like(rewards)
    mask = jnp.ones_like(rewards)
    gamma_k = 1.0
    for k in range(n):
        n_step_return = n_step_return + mask * gamma_k * pad_r[k : k + T]
        mask = mask * (1.0 - pad_d[k : k + T])
        gamma_k = gamma_k * gamma

    n_step_discount = mask * gamma_k
    return n_step_return, n_step_discount


def make_train(config: Dict[str, Any]):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    base_env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    num_agents = base_env.num_agents
    num_envs = int(config["NUM_ENVS"])
    config["NUM_ACTORS"] = num_agents * num_envs
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // int(config["NUM_STEPS"]) // num_envs
    )

    action_dim = base_env.action_space(base_env.agents[0]).n
    obs_dim = base_env.observation_space(base_env.agents[0]).shape[0]

    obs_with_agent_id = bool(config.get("OBS_WITH_AGENT_ID", True))
    env = SMAXWorldStateWrapper(base_env, obs_with_agent_id=obs_with_agent_id)
    env = SMAXLogWrapper(env)
    world_state_dim = env.world_state_size()

    num_steps = int(config["NUM_STEPS"])
    buffer_size = int(config["BUFFER_SIZE"])
    buffer_batch_size = int(config["BUFFER_BATCH_SIZE"])
    learning_starts = int(config["LEARNING_STARTS"])
    num_epochs = int(config["NUM_EPOCHS"])
    share_params = bool(config.get("SHARE_PARAMS", False))
    n_step = int(config.get("N_STEP", 1))
    gamma = float(config["GAMMA"])
    if n_step >= num_steps:
        raise ValueError(
            f"N_STEP={n_step} must be < NUM_STEPS={num_steps} so that there is at "
            "least one valid (t, t+n_step) pair per rollout."
        )
    valid_steps = num_steps - n_step + 1

    def train(rng):
        rng, rng_init = jax.random.split(rng)
        actor, critic1, critic2, states = init_train_states(
            rng_init,
            num_agents=num_agents,
            obs_dim=obs_dim,
            world_state_dim=world_state_dim,
            action_dim=action_dim,
            config=config,
        )

        rng, rng_reset = jax.random.split(rng)
        reset_rng = jax.random.split(rng_reset, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        buffer_state = create_replay_buffer(
            buffer_size=buffer_size,
            num_agents=num_agents,
            obs_dim=obs_dim,
            world_state_dim=world_state_dim,
            action_dim=action_dim,
        )

        def _update_step(runner_state, unused):
            states, env_state, last_obs, buffer_state, rng, update_step = runner_state

            def _env_step(step_state, unused):
                states, env_state, last_obs, rng = step_state

                rng, rng_action = jax.random.split(rng)
                avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail = jax.lax.stop_gradient(_stack_per_agent(avail, env.agents))

                obs_be = _stack_per_agent(last_obs, env.agents)
                logits = acting_logits_per_agent(
                    actor,
                    states.actor_state.params,
                    obs_be,
                    avail,
                    num_agents=num_agents,
                    share_params=share_params,
                )
                actions = jax.random.categorical(rng_action, logits, axis=-1)

                env_act = _unbatch_actions(actions, env.agents)
                rng, rng_step = jax.random.split(rng)
                rng_steps = jax.random.split(rng_step, num_envs)
                next_obsv, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, env_state, env_act)

                next_avail = jax.vmap(env.get_avail_actions)(next_env_state.env_state)
                next_avail = jax.lax.stop_gradient(_stack_per_agent(next_avail, env.agents))

                world_state = jnp.swapaxes(last_obs["world_state"], 0, 1)
                world_state_ep = world_state[:, 0, :]
                next_world_state = jnp.swapaxes(next_obsv["world_state"], 0, 1)
                next_world_state_ep = next_world_state[:, 0, :]

                next_obs_be = _stack_per_agent(next_obsv, env.agents)
                team_reward = reward[env.agents[0]]
                global_done = done["__all__"]

                transition = dict(
                    obs=obs_be,
                    world_state=world_state_ep,
                    action=actions,
                    avail_actions=avail,
                    reward=team_reward,
                    done=global_done,
                    next_obs=next_obs_be,
                    next_world_state=next_world_state_ep,
                    next_avail_actions=next_avail,
                    info=info,
                )

                step_state = (states, next_env_state, next_obsv, rng)
                return step_state, transition

            step_state = (states, env_state, last_obs, rng)
            step_state, traj_batch = jax.lax.scan(_env_step, step_state, None, num_steps)
            states, env_state, last_obs, rng = step_state

            n_step_return_full, n_step_discount_full = compute_n_step(
                traj_batch["reward"], traj_batch["done"], gamma, n_step,
            )

            def _add(buffer_state, t_idx):
                bootstrap_idx = t_idx + n_step - 1

                def _add_one(buffer_state, e_idx):
                    one = {
                        "obs": traj_batch["obs"][t_idx, e_idx],
                        "world_state": traj_batch["world_state"][t_idx, e_idx],
                        "action": traj_batch["action"][t_idx, e_idx],
                        "avail_actions": traj_batch["avail_actions"][t_idx, e_idx],
                        "reward": traj_batch["reward"][t_idx, e_idx],
                        "done": traj_batch["done"][t_idx, e_idx],
                        "next_obs": traj_batch["next_obs"][bootstrap_idx, e_idx],
                        "next_world_state": traj_batch["next_world_state"][bootstrap_idx, e_idx],
                        "next_avail_actions": traj_batch["next_avail_actions"][bootstrap_idx, e_idx],
                        "n_step_return": n_step_return_full[t_idx, e_idx],
                        "n_step_discount": n_step_discount_full[t_idx, e_idx],
                    }
                    return add_to_buffer(buffer_state, one), None
                buffer_state, _ = jax.lax.scan(_add_one, buffer_state, jnp.arange(num_envs))
                return buffer_state, None

            buffer_state, _ = jax.lax.scan(_add, buffer_state, jnp.arange(valid_steps))

            total_timesteps = update_step * num_envs * num_steps
            can_learn = total_timesteps >= learning_starts

            def _do_learning(carry):
                states, buffer_state, rng = carry

                def _epoch(carry, unused):
                    states, rng = carry
                    rng, rng_sample, rng_update = jax.random.split(rng, 3)
                    batch = sample_from_buffer(buffer_state, rng_sample, buffer_batch_size)
                    states, info = sac_update(
                        states,
                        batch,
                        rng_update,
                        actor=actor,
                        critic1=critic1,
                        critic2=critic2,
                        config=config,
                        num_agents=num_agents,
                        action_dim=action_dim,
                    )
                    return (states, rng), info

                (states, rng), infos = jax.lax.scan(
                    _epoch, (states, rng), None, num_epochs
                )
                avg_info = jax.tree.map(lambda x: jnp.mean(x), infos)
                return (states, buffer_state, rng), avg_info

            def _skip_learning(carry):
                states, buffer_state, rng = carry
                zero_info = {
                    "critic1_loss": jnp.array(0.0),
                    "critic2_loss": jnp.array(0.0),
                    "critic_loss": jnp.array(0.0),
                    "td_target_mean": jnp.array(0.0),
                    "q_taken_mean": jnp.array(0.0),
                    "actor_loss": jnp.array(0.0),
                    "entropy_per_agent_mean": jnp.array(0.0),
                    "q_pi_mean": jnp.array(0.0),
                    "alpha_loss": jnp.array(0.0),
                    "alpha_mean": jnp.exp(states.log_alphas).mean(),
                    "alpha_min": jnp.exp(states.log_alphas).min(),
                    "alpha_max": jnp.exp(states.log_alphas).max(),
                }
                return (states, buffer_state, rng), zero_info

            (states, buffer_state, rng), loss_info = jax.lax.cond(
                can_learn, _do_learning, _skip_learning, (states, buffer_state, rng)
            )

            info = traj_batch["info"]
            traj_info = jax.tree.map(
                lambda x: x.reshape((num_steps, num_envs, num_agents)), info
            )
            episode_done = traj_info["returned_episode"][:, :, 0]
            won = traj_info["returned_won_episode"][:, :, 0]
            ep_returns = traj_info["returned_episode_returns"][:, :, 0]

            def _safe_mean(values, mask):
                return jnp.where(
                    mask.any(),
                    jnp.where(mask, values, 0.0).sum() / jnp.maximum(mask.sum(), 1),
                    0.0,
                )

            metric = {
                "returns": _safe_mean(ep_returns, episode_done),
                "win_rate": _safe_mean(won, episode_done),
                "mean_reward": jnp.mean(traj_batch["reward"]),
                "env_step": update_step * num_envs * num_steps,
                "buffer_size": buffer_state.size,
                **loss_info,
            }

            def _log(m):
                step = int(m["env_step"])
                update = step // (num_envs * num_steps)
                total_updates = int(config["NUM_UPDATES"])
                progress_pct = 100 * update / max(total_updates, 1)
                if update % 10 == 0 or update == total_updates:
                    print(
                        f"[{update:5d}/{total_updates} {progress_pct:5.1f}%] "
                        f"step={step:9d} | ret={float(m['returns']):8.2f} "
                        f"wr={float(m['win_rate']):5.3f} "
                        f"r={float(m['mean_reward']):6.3f} | "
                        f"ent={float(m['entropy_per_agent_mean']):.3f} "
                        f"a={float(m['alpha_mean']):.4f} "
                        f"[{float(m['alpha_min']):.3f},{float(m['alpha_max']):.3f}] | "
                        f"c={float(m['critic_loss']):7.3f} a={float(m['actor_loss']):7.3f} "
                        f"Q={float(m['q_taken_mean']):7.2f} | buf={int(m['buffer_size'])}"
                    )

            jax.debug.callback(_log, metric)
            if config.get("WANDB_MODE") != "disabled":
                jax.debug.callback(lambda m: wandb.log(m), metric)

            runner_state = (states, env_state, last_obs, buffer_state, rng, update_step + 1)
            return runner_state, metric

        rng, rng_run = jax.random.split(rng)
        runner_state = (states, env_state, obsv, buffer_state, rng_run, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, int(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def get_config_name():
    config_name = "HASAC_smax"
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith("--config=") or arg.startswith("config="):
            config_name = arg.split("=")[1]
        elif arg.startswith("--") and "=" in arg:
            new_argv.append(arg[2:])
        else:
            new_argv.append(arg)
    sys.argv = new_argv
    return config_name


CONFIG_NAME = get_config_name()


@hydra.main(version_base=None, config_path="config", config_name=CONFIG_NAME)
def main(config):
    config = OmegaConf.to_container(config)

    print("=" * 80)
    print("HASAC (faithful, MLP) on SMAX")
    print("=" * 80)
    print(f"Map: {config['MAP_NAME']}")
    print(f"SHARE_PARAMS: {config.get('SHARE_PARAMS', False)}")
    print(f"FIXED_ORDER:  {config.get('FIXED_ORDER', False)}")
    print(f"FIXED_ALPHA:  {config['FIXED_ALPHA']}")
    print(f"N_STEP:       {config.get('N_STEP', 1)}")
    print(f"USE_HUBER:    {config.get('USE_HUBER', True)} (delta={config.get('HUBER_DELTA', 10.0)})")
    print(f"NUM_ENVS x NUM_STEPS x NUM_EPOCHS: {config['NUM_ENVS']} x {config['NUM_STEPS']} x {config['NUM_EPOCHS']}")
    print(f"TOTAL_TIMESTEPS: {float(config['TOTAL_TIMESTEPS']):.0f}")
    print("=" * 80, flush=True)

    map_name = config["MAP_NAME"].replace("/", "-").replace("_", "-")
    run_name = config.get("RUN_NAME") or f"HASAC-SMAX-{map_name}-seed{config['SEED']}"
    entity = config.get("ENTITY") or None
    run = wandb.init(
        entity=entity,
        project=config.get("PROJECT", "JaxMARL"),
        name=run_name,
        tags=["HASAC", "SMAX", config["MAP_NAME"], f"seed{config['SEED']}"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    print("Starting training...")
    try:
        train_jit = jax.jit(make_train(config))
        t0 = time()
        out = train_jit(rng)
        elapsed = time() - t0
        print()
        print("=" * 80)
        print(f"Training completed in {elapsed:.2f} s ({elapsed/60:.2f} min)")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n> Training interrupted by user")
        raise
    wandb.finish()


if __name__ == "__main__":
    main()
