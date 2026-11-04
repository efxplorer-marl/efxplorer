import os
import sys
import logging

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import hydra
from omegaconf import OmegaConf
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper
import wandb
from time import time
from typing import NamedTuple, Dict

from core import (
    MPEWorldStateWrapper, ScannedRNN, create_actor_train_state, create_critic_train_state,
    create_tdd_train_states,
    Transition, batchify, unbatchify, make_linear_schedule, discounted_sampling,
    compute_video_steps, stack_params, save_params, CallbackHandler,
    compute_intrinsic_reward_sd,
    mrn_distance,
    vectorized_apply_gradients,
    ActorRNN,
)


class TrainerConfigLinear(NamedTuple):
    num_envs: int
    num_steps: int
    num_agents: int
    num_actors: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    anneal_ent: bool
    ent_coef_end: float
    max_grad_norm: float
    num_updates: int
    update_epochs: int

    intrinsic_reward_beta: float
    intrinsic_reward_scale: float
    warmup_rollouts: int
    seed: int

    tdd_latent_dim: int
    tdd_output_dim: int
    tdd_lr: float
    tdd_batch_size: int
    tdd_update_epochs: int
    tdd_discount: float


def create_trainer_config_linear(config: Dict, num_agents: int) -> TrainerConfigLinear:
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    return TrainerConfigLinear(
        num_envs=config["NUM_ENVS"],
        num_steps=config["NUM_STEPS"],
        num_agents=num_agents,
        num_actors=config["NUM_ACTORS"],
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=config["CLIP_EPS"],
        ent_coef=config["ENT_COEF"],
        anneal_ent=config.get("ANNEAL_ENT", False),
        ent_coef_end=config.get("ENT_COEF_END", config["ENT_COEF"]),
        max_grad_norm=config["MAX_GRAD_NORM"],
        num_updates=config["NUM_UPDATES"],
        update_epochs=config["UPDATE_EPOCHS"],

        intrinsic_reward_beta=config.get("INTRINSIC_REWARD_BETA", 0.1),
        intrinsic_reward_scale=config.get("INTRINSIC_REWARD_SCALE", 1.0),
        warmup_rollouts=config.get("WARMUP_ROLLOUTS", 0),
        seed=config["SEED"],

        tdd_latent_dim=config.get("TDD_LATENT_DIM", 64),
        tdd_output_dim=config.get("TDD_OUTPUT_DIM", 64),
        tdd_lr=config.get("TDD_LR", 0.0004),
        tdd_batch_size=config.get("TDD_BATCH_SIZE", 512),
        tdd_update_epochs=config.get("TDD_UPDATE_EPOCHS", 144),
        tdd_discount=config.get("TDD_DISCOUNT", 0.9),
    )


def print_training_header_linear(config: Dict):
    print("\n" + "=" * 80)
    print("MAPPO Linear - r = r_ext + alpha * r_int")
    print("=" * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Seed: {config['SEED']}")
    print(f"Total Timesteps: {config['TOTAL_TIMESTEPS']:,.0f}")
    print(f"Num Envs: {config['NUM_ENVS']} | Num Steps: {config['NUM_STEPS']}")
    print(f"Update Epochs: {config['UPDATE_EPOCHS']} (NO MINIBATCH - full trajectory)")
    print(f"Linear: beta={config.get('INTRINSIC_REWARD_BETA', 0.1)}, scale={config.get('INTRINSIC_REWARD_SCALE', 1.0)}")
    print("=" * 80 + "\n")


def make_train(config):
    env_kwargs = config.get("ENV_KWARGS", {})
    env_name = config["ENV_NAME"]
    if "corridor" in env_name.lower():
        env_kwargs["sparse_reward"] = config.get("SPARSE_REWARD", False)
    env = jaxmarl.make(env_name, **env_kwargs)

    num_learning_agents = len(env.agents)
    config["NUM_ACTORS"] = num_learning_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]

    tc = create_trainer_config_linear(config, num_learning_agents)

    env = MPEWorldStateWrapper(env)
    env = MPELogWrapper(env)

    anneal_lr = config.get("ANNEAL_LR", False)
    actor_lr = config.get("ACTOR_LR") or config.get("LR", 0.001)
    critic_lr = config.get("CRITIC_LR") or config.get("LR", 0.001)
    if anneal_lr:
        actor_lr_schedule = make_linear_schedule(actor_lr, 1, tc.update_epochs, tc.num_updates)
        critic_lr_schedule = make_linear_schedule(critic_lr, 1, tc.update_epochs, tc.num_updates)
    else:
        actor_lr_schedule = None
        critic_lr_schedule = None

    def train(rng):
        rng, _rng_actor, _rng_critic, _rng_tdd = jax.random.split(rng, 4)
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        action_dim = env.action_space(env.agents[0]).n

        _, _, actor_networks, actor_train_states = create_actor_train_state(
            rng=_rng_actor, config=config, action_dim=action_dim, obs_dim=obs_dim,
            num_agents=tc.num_agents, lr_schedule=actor_lr_schedule, shared_params=False,
        )
        critic_network, critic_train_state = create_critic_train_state(
            rng=_rng_critic, config=config, world_state_size=env.world_state_size(), lr_schedule=critic_lr_schedule,
        )

        tdd_potential_states_raw, tdd_s_encoder_states_raw = create_tdd_train_states(
            rng=_rng_tdd, config=config, num_agents=tc.num_agents, input_dim=2,
        )
        tdd_potential_states = tuple(tdd_potential_states_raw)
        tdd_s_encoder_states = tuple(tdd_s_encoder_states_raw)

        print(f"[Linear] beta={tc.intrinsic_reward_beta}, scale={tc.intrinsic_reward_scale}")
        print(f"[Linear] Actor Update: Per-agent (NO MINIBATCH)")
        if tc.anneal_ent:
            print(f"[Linear] Entropy Annealing: {tc.ent_coef} -> {tc.ent_coef_end}")

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, tc.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config["GRU_HIDDEN_DIM"])

        raw_env = env
        while hasattr(raw_env, '_env'):
            raw_env = raw_env._env
        video_steps = compute_video_steps(tc.num_updates, tc.warmup_rollouts, config.get("NUM_INTERMEDIATE_VIDEOS", 6))
        callback_handler = CallbackHandler(
            config=config, env=raw_env, actor_network=actor_networks, critic_network=critic_network,
            shared_actor_params=False, num_agents=tc.num_agents, video_steps=video_steps,
        )

        agent_idx_array = jnp.repeat(jnp.arange(tc.num_agents), tc.num_envs)

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            current_ent_coef = jax.lax.cond(
                tc.anneal_ent,
                lambda: tc.ent_coef - (tc.ent_coef - tc.ent_coef_end) * (update_steps / tc.num_updates),
                lambda: jnp.float32(tc.ent_coef),
            )

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state
                actor_train_states_local, critic_train_state_local, _, _ = train_states
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, tc.num_actors)

                actor_template = actor_networks[0]
                stacked_actor_params = stack_params([s.params for s in actor_train_states_local])
                obs_reshaped = obs_batch.reshape(tc.num_agents, tc.num_envs, -1)
                done_reshaped = last_done.reshape(tc.num_agents, tc.num_envs)
                hstate_reshaped = hstates[0].reshape(tc.num_agents, tc.num_envs, -1)

                def _forward_single_actor(params, hidden, obs, done):
                    ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
                    new_hidden, _, logits = actor_template.apply(params, hidden, ac_in)
                    return new_hidden, logits.squeeze(0)

                ac_hstate_per_agent, action_logits_per_agent = jax.vmap(_forward_single_actor)(
                    stacked_actor_params, hstate_reshaped, obs_reshaped, done_reshaped
                )

                ac_hstate = ac_hstate_per_agent.reshape(tc.num_actors, -1)
                action_logits = action_logits_per_agent.reshape(tc.num_actors, -1)

                pi = distrax.Categorical(logits=action_logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                env_act = unbatchify(action, env.agents, tc.num_envs, tc.num_agents)

                world_state = last_obs["world_state"].swapaxes(0, 1).reshape((tc.num_actors, -1))
                cr_in = (world_state[None, :], last_done[np.newaxis, :])
                cr_hstate, value = critic_network.apply(critic_train_state_local.params, hstates[1], cr_in)
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, tc.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((tc.num_actors)), info)
                done_batch = batchify(done, env.agents, tc.num_actors).squeeze()

                agent_positions = obs_batch[:, 2:4]

                transition = Transition(
                    jnp.tile(done["__all__"], tc.num_agents), last_done, action.squeeze(), action_logits,
                    value.squeeze(), batchify(reward, env.agents, tc.num_actors).squeeze(), log_prob.squeeze(),
                    obs_batch, world_state, info,
                    z_i=jnp.zeros((tc.num_actors,)),
                    intrinsic_reward=None,
                    agent_positions=agent_positions,
                    agent_local_features_curr=agent_positions, agent_local_features_next=None,
                    agent_idx=agent_idx_array,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, tc.num_steps)

            train_states, env_state, last_obs, last_done, hstates, rng = runner_state
            actor_train_states, critic_train_state, tdd_potential_states, tdd_s_encoder_states = train_states

            obs_all = traj_batch.obs
            obs_reshaped = obs_all.reshape(tc.num_steps, tc.num_agents, tc.num_envs, -1)
            agent_features_curr_all = obs_reshaped[..., 2:4].transpose(1, 0, 2, 3)
            agent_features_next_all = jnp.concatenate([agent_features_curr_all[:, 1:, :, :], agent_features_curr_all[:, -1:, :, :]], axis=1)
            rng, _rng_tdd = jax.random.split(rng)

            def _tdd_update_step(tdd_states_arg, rng_arg):
                potential_states_inner, s_encoder_states_inner = tdd_states_arg
                batch_size = tc.tdd_batch_size
                I = jnp.eye(batch_size)

                agent_rngs = jax.random.split(rng_arg, tc.num_agents)
                potential_apply_fn = potential_states_inner[0].apply_fn
                s_encoder_apply_fn = s_encoder_states_inner[0].apply_fn
                stacked_potential_params = stack_params([s.params for s in potential_states_inner])
                stacked_s_encoder_params = stack_params([s.params for s in s_encoder_states_inner])

                def _tdd_loss_single_agent(potential_params, s_encoder_params, agent_features, agent_rng):
                    rng_t, rng_env, rng_goal = jax.random.split(agent_rng, 3)
                    t_indices = jax.random.randint(rng_t, (batch_size,), 0, tc.num_steps - 1)
                    env_indices = jax.random.randint(rng_env, (batch_size,), 0, tc.num_envs)
                    obs = agent_features[t_indices, env_indices, :]
                    ranges = (tc.num_steps - 1 - t_indices).astype(jnp.int32)
                    ranges = jnp.maximum(ranges, 1)
                    intervals = discounted_sampling(ranges, tc.tdd_discount, rng_goal)
                    goal_t = t_indices + intervals + 1
                    goal_t = jnp.clip(goal_t, 0, tc.num_steps - 1)
                    goal = agent_features[goal_t, env_indices, :]

                    valid_mask = goal_t > t_indices
                    valid_mask_float = valid_mask.astype(jnp.float32)
                    num_valid = jnp.sum(valid_mask_float)

                    c_g = potential_apply_fn(potential_params, goal)
                    phi_s = s_encoder_apply_fn(s_encoder_params, obs)
                    phi_g = s_encoder_apply_fn(s_encoder_params, goal)

                    phi_s_expanded = phi_s[:, None, :]
                    phi_g_expanded = phi_g[None, :, :]
                    mrn_dists = mrn_distance(phi_s_expanded, phi_g_expanded)

                    c_g_squeezed = c_g.squeeze(-1)
                    logits = c_g_squeezed[None, :] - mrn_dists

                    log_softmax_logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
                    loss1_per_sample = -(log_softmax_logits * I).sum(axis=-1)
                    loss1 = jnp.sum(loss1_per_sample * valid_mask_float) / jnp.maximum(num_valid, 1)

                    log_softmax_logits_T = logits.T - jax.scipy.special.logsumexp(logits.T, axis=-1, keepdims=True)
                    loss2_per_sample = -(log_softmax_logits_T * I).sum(axis=-1)
                    loss2 = jnp.sum(loss2_per_sample * valid_mask_float) / jnp.maximum(num_valid, 1)

                    return (loss1 + loss2) / 2

                def _compute_loss_and_grad(pot_params, enc_params, features, rng):
                    loss, (pot_grad, enc_grad) = jax.value_and_grad(
                        _tdd_loss_single_agent, argnums=(0, 1)
                    )(pot_params, enc_params, features, rng)
                    return loss, pot_grad, enc_grad

                losses, potential_grads, s_encoder_grads = jax.vmap(_compute_loss_and_grad)(
                    stacked_potential_params, stacked_s_encoder_params,
                    agent_features_curr_all, agent_rngs
                )

                new_potential_states_list = vectorized_apply_gradients(list(potential_states_inner), potential_grads)
                new_s_encoder_states_list = vectorized_apply_gradients(list(s_encoder_states_inner), s_encoder_grads)

                total_loss = losses.mean()
                return (tuple(new_potential_states_list), tuple(new_s_encoder_states_list)), total_loss

            def _tdd_update_epoch(carry, unused):
                tdd_states_inner, rng_inner = carry
                rng_inner, _rng_step = jax.random.split(rng_inner)
                new_tdd_states_inner, loss = _tdd_update_step(tdd_states_inner, _rng_step)
                return (new_tdd_states_inner, rng_inner), loss

            init_tdd_states = (tdd_potential_states, tdd_s_encoder_states)

            (new_tdd_states, _), tdd_losses = jax.lax.scan(
                _tdd_update_epoch, (init_tdd_states, _rng_tdd), None, tc.tdd_update_epochs
            )
            tdd_loss = tdd_losses.mean()

            tdd_potential_states = tuple(new_tdd_states[0])
            tdd_s_encoder_states = tuple(new_tdd_states[1])

            intrinsic_reward_per_agent = compute_intrinsic_reward_sd(
                agent_features_curr_all, agent_features_next_all, tdd_s_encoder_states, tdd_shared=False,
                num_agents=tc.num_agents, num_envs=tc.num_envs, num_steps=tc.num_steps,
            )
            intrinsic_reward_all = intrinsic_reward_per_agent.transpose(1, 0, 2).reshape(tc.num_steps, tc.num_actors)
            intrinsic_reward_raw = intrinsic_reward_all

            r_int_scaled = intrinsic_reward_all * tc.intrinsic_reward_scale

            use_intrinsic = update_steps >= tc.warmup_rollouts
            r_ext = traj_batch.reward.astype(jnp.float32)
            r_combined = jax.lax.cond(
                use_intrinsic,
                lambda: r_ext + tc.intrinsic_reward_beta * r_int_scaled,
                lambda: r_ext,
            )

            traj_batch = traj_batch._replace(intrinsic_reward=r_int_scaled)

            last_world_state = last_obs["world_state"].swapaxes(0, 1).reshape((tc.num_actors, -1))
            cr_in = (last_world_state[None, :], last_done[np.newaxis, :])
            _, last_val = critic_network.apply(critic_train_state.params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch_arg, last_val_arg, combined_rewards):
                def _get_advantages(gae_and_next_value, inputs):
                    gae, next_value = gae_and_next_value
                    done, value, reward = inputs
                    next_value, value = next_value.reshape(-1), value.reshape(-1)
                    delta = reward + tc.gamma * next_value * (1 - done) - value
                    gae = delta + tc.gamma * tc.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                inputs = (traj_batch_arg.global_done, traj_batch_arg.value, combined_rewards)
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val_arg), last_val_arg), inputs, reverse=True, unroll=16)
                return advantages, advantages + traj_batch_arg.value

            advantages, targets = _calculate_gae(traj_batch, last_val, r_combined)

            def _update_epoch(update_state, unused):
                (train_states_upd, init_hstates, traj_batch_upd,
                 advantages_upd, targets_upd, rng_upd) = update_state
                rng_upd, _rng = jax.random.split(rng_upd)

                actor_train_states_upd, critic_train_state_upd = train_states_upd

                def to_per_agent_3d(x):
                    return x.reshape(tc.num_steps, tc.num_agents, tc.num_envs).transpose(1, 0, 2)

                def to_per_agent_4d(x):
                    return x.reshape(tc.num_steps, tc.num_agents, tc.num_envs, -1).transpose(1, 0, 2, 3)

                agent_obs = to_per_agent_4d(traj_batch_upd.obs)
                agent_done = to_per_agent_3d(traj_batch_upd.done)
                agent_action = to_per_agent_3d(traj_batch_upd.action)
                agent_log_prob = to_per_agent_3d(traj_batch_upd.log_prob)
                agent_gae = to_per_agent_3d(advantages_upd)
                agent_init_hstate = init_hstates[0].reshape(tc.num_agents, tc.num_envs, -1)

                actor_template = actor_networks[0]
                stacked_actor_params = stack_params([s.params for s in actor_train_states_upd])

                def _actor_loss_single_agent(actor_params, obs, done, action, log_prob, gae, init_hstate, agent_rng):
                    perm = jax.random.permutation(agent_rng, tc.num_envs)
                    obs_shuffled = jnp.take(obs, perm, axis=1)
                    done_shuffled = jnp.take(done, perm, axis=1)
                    action_shuffled = jnp.take(action, perm, axis=1)
                    log_prob_shuffled = jnp.take(log_prob, perm, axis=1)
                    gae_shuffled = jnp.take(gae, perm, axis=1)
                    hstate_shuffled = jnp.take(init_hstate, perm, axis=0)

                    _, pi, _ = actor_template.apply(actor_params, hstate_shuffled, (obs_shuffled, done_shuffled))
                    new_log_prob = pi.log_prob(action_shuffled)
                    entropy = pi.entropy()

                    logratio = new_log_prob - log_prob_shuffled
                    ratio = jnp.exp(logratio)
                    gae_normalized = (gae_shuffled - gae_shuffled.mean()) / (gae_shuffled.std() + 1e-6)
                    loss_actor1 = ratio * gae_normalized
                    loss_actor2 = jnp.clip(ratio, 1.0 - tc.clip_eps, 1.0 + tc.clip_eps) * gae_normalized
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                    entropy_mean = entropy.mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_frac = jnp.mean(jnp.abs(ratio - 1) > tc.clip_eps)

                    actor_loss = loss_actor - current_ent_coef * entropy_mean
                    return actor_loss, (loss_actor, entropy_mean, ratio, approx_kl, clip_frac)

                def _compute_actor_grad(params, obs, done, action, log_prob, gae, init_hstate, rng):
                    (loss, aux), grad = jax.value_and_grad(_actor_loss_single_agent, has_aux=True)(
                        params, obs, done, action, log_prob, gae, init_hstate, rng
                    )
                    return loss, aux, grad

                agent_rngs = jax.random.split(_rng, tc.num_agents)
                actor_losses, actor_aux, actor_grads = jax.vmap(_compute_actor_grad)(
                    stacked_actor_params, agent_obs, agent_done, agent_action,
                    agent_log_prob, agent_gae, agent_init_hstate, agent_rngs
                )

                new_actor_states = vectorized_apply_gradients(actor_train_states_upd, actor_grads)

                def _critic_loss_fn(critic_params, init_hstate, traj_batch_inner, targets_inner):
                    _, value = critic_network.apply(critic_params, init_hstate, (traj_batch_inner.world_state, traj_batch_inner.done))
                    value_pred_clipped = traj_batch_inner.value + (value - traj_batch_inner.value).clip(-tc.clip_eps, tc.clip_eps)
                    value_losses = jnp.square(value - targets_inner)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets_inner)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    return value_loss, value_loss

                cr_init_hstate = init_hstates[1]
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                critic_loss, critic_grads = critic_grad_fn(
                    critic_train_state_upd.params, cr_init_hstate, traj_batch_upd, targets_upd
                )
                critic_train_state_upd = critic_train_state_upd.apply_gradients(grads=critic_grads)

                actor_loss_mean = actor_losses.mean()
                entropy_mean = actor_aux[1].mean()
                ratio_mean = actor_aux[2].mean()
                approx_kl_mean = actor_aux[3].mean()
                clip_frac_mean = actor_aux[4].mean()

                total_loss = actor_loss_mean + critic_loss[0]

                actor_grad_norms = jax.vmap(optax.global_norm)(actor_grads)
                critic_grad_norm = optax.global_norm(critic_grads)

                loss_info = {
                    "total_loss": total_loss, "actor_loss": actor_loss_mean, "value_loss": critic_loss[0],
                    "entropy": entropy_mean, "ratio": ratio_mean,
                    "approx_kl": approx_kl_mean, "clip_frac": clip_frac_mean,
                    "actor_grad_norm": actor_grad_norms.mean(),
                    "critic_grad_norm": critic_grad_norm,
                }

                train_states_upd = (new_actor_states, critic_train_state_upd)
                update_state = (train_states_upd, init_hstates, traj_batch_upd, advantages_upd, targets_upd, rng_upd)
                return update_state, loss_info

            train_states = (actor_train_states, critic_train_state)
            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, tc.update_epochs)

            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            actor_train_states, critic_train_state = train_states

            metric = traj_batch.info
            metric["loss"] = loss_info
            metric["tdd_loss"] = tdd_loss
            metric["linear"] = {
                "beta": tc.intrinsic_reward_beta,
                "r_ext_mean": jnp.mean(r_ext),
                "r_int_mean": jnp.mean(r_int_scaled),
                "r_combined_mean": jnp.mean(r_combined),
            }
            metric["rewards"] = {"extrinsic": jnp.mean(r_ext), "intrinsic_scaled": jnp.mean(r_int_scaled), "combined": jnp.mean(r_combined)}
            metric["ent_coef"] = current_ent_coef
            metric["action_logits"] = traj_batch.action_logits
            metric["intrinsic_reward_scaled"] = traj_batch.intrinsic_reward
            metric["intrinsic_reward_raw"] = intrinsic_reward_raw
            metric["agent_positions"] = traj_batch.agent_positions
            metric["num_agents"], metric["num_envs"] = tc.num_agents, tc.num_envs
            metric["intrinsic_reward_scale"] = tc.intrinsic_reward_scale
            metric["actor_params"] = [s.params for s in actor_train_states]
            metric["critic_params"] = critic_train_state.params
            metric["s_encoder_params"] = tdd_s_encoder_states[0].params
            metric["tdd_potential_params"] = [s.params for s in tdd_potential_states]
            metric["tdd_s_encoder_params"] = [s.params for s in tdd_s_encoder_states]
            rng = update_state[-1]
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback_handler, None, metric, ordered=True)
            update_steps = update_steps + 1
            train_states = (actor_train_states, critic_train_state, tdd_potential_states, tdd_s_encoder_states)
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)

            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        train_states = (actor_train_states, critic_train_state, tdd_potential_states, tdd_s_encoder_states)
        runner_state = (train_states, env_state, obsv, jnp.zeros((tc.num_actors), dtype=bool), (ac_init_hstate, cr_init_hstate), _rng)

        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, tc.num_updates
        )

        return {"runner_state": runner_state}

    return train


def convert_wandb_args_to_hydra():
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith('--') and '=' in arg:
            new_arg = arg[2:]
            new_argv.append(new_arg)
        else:
            new_argv.append(arg)
    sys.argv = new_argv

convert_wandb_args_to_hydra()


@hydra.main(version_base=None, config_path="config", config_name="mappo_linear_corridor")
def main(config):
    config = OmegaConf.to_container(config)
    print_training_header_linear(config)

    wandb_project = config.get("PROJECT", "mappo")
    env_display_name = config["ENV_NAME"].replace("/", "-").replace("_", "-")
    run_name = config.get("RUN_NAME") or f"mappo-linear-{env_display_name}"
    run = wandb.init(
        project=wandb_project, name=run_name, 
        tags=["mappo", "linear", config["ENV_NAME"]],
        config=config, mode=config["WANDB_MODE"]
    )

    rng = jax.random.PRNGKey(config["SEED"])
    print("Starting training...")
    try:
        with jax.disable_jit(False):
            train_jit = jax.jit(make_train(config))
            start_time = time()
            out = train_jit(rng)
            elapsed_time = time() - start_time
            print(f"\n{'=' * 80}\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n{'=' * 80}")
    except KeyboardInterrupt:
        print("\n> Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n> Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    from core.efxplorer_trainer_Kreturn import save_training_checkpoints
    save_training_checkpoints(run, config, out, save_params)


if __name__ == "__main__":
    main()
