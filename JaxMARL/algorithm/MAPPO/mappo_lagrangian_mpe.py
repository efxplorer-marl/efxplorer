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
from flax.training.train_state import TrainState

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

from core.networks_z import IntrinsicCriticRNN, create_intrinsic_critic_train_state


class TrainerConfigPureLagrangian(NamedTuple):
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

    lambda_init: float
    lambda_lr: float
    lambda_max: float
    adv_clip_abs: float
    normalize_lagrangian_adv: bool
    warmup_rollouts: int
    seed: int

    tdd_latent_dim: int
    tdd_output_dim: int
    tdd_lr: float
    tdd_batch_size: int
    tdd_update_epochs: int
    tdd_discount: float

    intrinsic_reward_scale: float

    v_int_gae_lambda: float


def create_trainer_config_pure_lagrangian(config: Dict, num_agents: int) -> TrainerConfigPureLagrangian:
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    return TrainerConfigPureLagrangian(
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

        lambda_init=config.get("LAMBDA_INIT", 1.0),
        lambda_lr=config.get("LAMBDA_LR", 0.005),
        lambda_max=config.get("LAMBDA_MAX", 10.0),
        adv_clip_abs=config.get("ADV_CLIP_ABS", 100.0),
        normalize_lagrangian_adv=config.get("NORMALIZE_LAGRANGIAN_ADV", False),
        warmup_rollouts=config.get("WARMUP_ROLLOUTS", 0),
        seed=config["SEED"],

        tdd_latent_dim=config.get("TDD_LATENT_DIM", 64),
        tdd_output_dim=config.get("TDD_OUTPUT_DIM", 64),
        tdd_lr=config.get("TDD_LR", 0.0004),
        tdd_batch_size=config.get("TDD_BATCH_SIZE", 512),
        tdd_update_epochs=config.get("TDD_UPDATE_EPOCHS", 144),
        tdd_discount=config.get("TDD_DISCOUNT", 0.9),

        intrinsic_reward_scale=config.get("INTRINSIC_REWARD_SCALE", 1.0),

        v_int_gae_lambda=config.get("V_INT_GAE_LAMBDA", gae_lambda),
    )


def print_training_header_pure_lagrangian(config: Dict):
    print("\n" + "=" * 80)
    print("MAPPO Lagrangian - NO Z-AUGMENTED STATE")
    print("=" * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Seed: {config['SEED']}")
    print(f"Total Timesteps: {config['TOTAL_TIMESTEPS']:,.0f}")
    print(f"Num Envs: {config['NUM_ENVS']} | Num Steps: {config['NUM_STEPS']}")
    print(f"Update Epochs: {config['UPDATE_EPOCHS']} (NO MINIBATCH - full trajectory)")
    print(f"Lagrangian: lambda_init={config.get('LAMBDA_INIT', 1.0)}, lambda_lr={config.get('LAMBDA_LR', 0.005)}")
    print("V^int(s) - State-only critic (NO z-augmented state)")
    print("=" * 80 + "\n")


def compute_intrinsic_gae_pure(r_int, v_int_all, last_v_int, gamma, gae_lambda, dones):
    def _get_advantages(gae_and_next_value, transition_data):
        gae, next_value = gae_and_next_value
        done, r, v = transition_data
        delta = r + gamma * next_value * (1 - done) - v
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, v), gae

    v_int_flat = v_int_all.squeeze() if v_int_all.ndim > 2 else v_int_all
    _, advantages_int = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_v_int), last_v_int),
        (dones, r_int, v_int_flat),
        reverse=True,
        unroll=16
    )
    targets_v_int = advantages_int + v_int_flat
    return advantages_int, targets_v_int


def compute_lagrangian_advantage_pure(advantages_ext, advantages_int, lambda_t, adv_clip_abs, normalize_advantages):
    adv_ext_clipped = jnp.clip(advantages_ext, -adv_clip_abs, adv_clip_abs)
    adv_int_clipped = jnp.clip(advantages_int, -adv_clip_abs, adv_clip_abs)

    combined = adv_int_clipped + lambda_t * adv_ext_clipped

    if normalize_advantages:
        combined = (combined - combined.mean()) / (combined.std() + 1e-8)

    stats = {
        "lambda": lambda_t,
        "A_ext_mean": advantages_ext.mean(),
        "A_ext_std": advantages_ext.std(),
        "A_int_mean": advantages_int.mean(),
        "A_int_std": advantages_int.std(),
        "A_combined_mean": combined.mean(),
        "A_combined_std": combined.std(),
    }
    return combined, stats


def update_lambda_pure(lambda_t, mean_adv_ext, lambda_lr, lambda_max):
    lambda_new = lambda_t - lambda_lr * mean_adv_ext
    lambda_new = jnp.clip(lambda_new, 0.0, lambda_max)
    return lambda_new


def make_train(config):
    env_kwargs = config.get("ENV_KWARGS", {})
    env_name = config["ENV_NAME"]
    if "corridor" in env_name.lower():
        env_kwargs["sparse_reward"] = config.get("SPARSE_REWARD", False)
    env = jaxmarl.make(env_name, **env_kwargs)

    num_learning_agents = len(env.agents)
    config["NUM_ACTORS"] = num_learning_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]

    tc = create_trainer_config_pure_lagrangian(config, num_learning_agents)

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
        rng, _rng_actor, _rng_critic, _rng_tdd, _rng_v_int = jax.random.split(rng, 5)
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

        v_int_network = IntrinsicCriticRNN(config=config)
        v_int_init_x = (
            jnp.zeros((1, tc.num_actors, tc.tdd_output_dim)),
            jnp.zeros((1, tc.num_actors)),
        )
        v_int_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config["GRU_HIDDEN_DIM"])
        v_int_params = v_int_network.init(_rng_v_int, v_int_init_hstate, v_int_init_x)

        v_int_lr = config.get("V_INT_LR") or config.get("LR") or critic_lr
        v_int_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(v_int_lr, eps=1e-5),
        )
        v_int_train_state = TrainState.create(
            apply_fn=v_int_network.apply,
            params=v_int_params,
            tx=v_int_tx,
        )
        v_int_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config["GRU_HIDDEN_DIM"])

        print(f"[Pure-Lagrangian] lambda_init={tc.lambda_init}, lambda_lr={tc.lambda_lr}, lambda_max={tc.lambda_max}")
        print(f"[Pure-Lagrangian] V^int: State-only critic (NO z-augmented state)")
        if tc.anneal_ent:
            print(f"[Pure-Lagrangian] Entropy Annealing: {tc.ent_coef} -> {tc.ent_coef_end}")

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
            runner_state, update_steps, lambda_t = update_runner_state

            current_ent_coef = jax.lax.cond(
                tc.anneal_ent,
                lambda: tc.ent_coef - (tc.ent_coef - tc.ent_coef_end) * (update_steps / tc.num_updates),
                lambda: jnp.float32(tc.ent_coef),
            )

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state
                actor_train_states_local, critic_train_state_local, _, _, _ = train_states
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
            actor_train_states, critic_train_state, tdd_potential_states, tdd_s_encoder_states, v_int_train_state = train_states
            last_world_state = last_obs["world_state"].swapaxes(0, 1).reshape((tc.num_actors, -1))
            cr_in = (last_world_state[None, :], last_done[np.newaxis, :])
            _, last_val = critic_network.apply(critic_train_state.params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            last_obs_batch = batchify(last_obs, env.agents, tc.num_actors)
            last_agent_positions = last_obs_batch[:, 2:4]

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.global_done, transition.value, transition.reward
                    next_value, value = next_value.reshape(-1), value.reshape(-1)
                    delta = reward + tc.gamma * next_value * (1 - done) - value
                    gae = delta + tc.gamma * tc.gae_lambda * (1 - done) * gae
                    return (gae, value), gae
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            advantages_ext, targets_ext = _calculate_gae(traj_batch, last_val)

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

            r_sd = intrinsic_reward_all * tc.intrinsic_reward_scale
            intrinsic_reward_all = jax.lax.stop_gradient(r_sd)

            traj_batch = traj_batch._replace(intrinsic_reward=intrinsic_reward_all)

            use_lagrangian = update_steps >= tc.warmup_rollouts

            def _compute_lagrangian_gae(args):
                traj_batch_arg, advantages_ext_arg, last_agent_positions_arg, lambda_current = args
                r_int = traj_batch_arg.intrinsic_reward

                agent_positions_traj = traj_batch_arg.agent_positions
                s_encoder_apply = tdd_s_encoder_states[0].apply_fn
                stacked_s_encoder_params = stack_params([s.params for s in tdd_s_encoder_states])
                agent_positions_reshaped = agent_positions_traj.reshape(tc.num_steps, tc.num_agents, tc.num_envs, 2)

                def _encode_single_agent(agent_params, agent_pos_seq):
                    flat_pos = agent_pos_seq.reshape(-1, 2)
                    embeddings = jax.vmap(lambda pos: s_encoder_apply(agent_params, pos))(flat_pos)
                    return embeddings.reshape(tc.num_steps, tc.num_envs, -1)

                s_enc_per_agent = jax.vmap(_encode_single_agent)(stacked_s_encoder_params, agent_positions_reshaped.transpose(1, 0, 2, 3))
                s_enc_embeddings = s_enc_per_agent.transpose(1, 0, 2, 3).reshape(tc.num_steps, tc.num_actors, -1)
                s_enc_embeddings = jax.lax.stop_gradient(s_enc_embeddings)

                v_int_in = (s_enc_embeddings, traj_batch_arg.global_done)
                _, v_int_all = v_int_network.apply(v_int_train_state.params, v_int_init_hstate, v_int_in)

                last_pos_reshaped = last_agent_positions_arg.reshape(tc.num_agents, tc.num_envs, 2)
                def _encode_last_pos(agent_params, agent_pos):
                    return jax.vmap(lambda pos: s_encoder_apply(agent_params, pos))(agent_pos)
                last_enc_per_agent = jax.vmap(_encode_last_pos)(stacked_s_encoder_params, last_pos_reshaped)
                last_s_enc = jax.lax.stop_gradient(last_enc_per_agent.reshape(tc.num_actors, -1))

                last_v_int_in = (last_s_enc[None, :], traj_batch_arg.global_done[-1:])
                _, last_v_int = v_int_network.apply(v_int_train_state.params, v_int_init_hstate, last_v_int_in)
                last_v_int = last_v_int.squeeze(0)

                advantages_int, targets_v_int = compute_intrinsic_gae_pure(
                    r_int, v_int_all, last_v_int, tc.gamma, tc.v_int_gae_lambda, traj_batch_arg.global_done
                )

                combined_advantages, lagrangian_stats = compute_lagrangian_advantage_pure(
                    advantages_ext=advantages_ext_arg,
                    advantages_int=advantages_int,
                    lambda_t=lambda_current,
                    adv_clip_abs=tc.adv_clip_abs,
                    normalize_advantages=tc.normalize_lagrangian_adv,
                )

                mean_adv_ext = advantages_ext_arg.mean()
                return combined_advantages, lagrangian_stats, targets_v_int, v_int_all, mean_adv_ext

            def _compute_standard_gae_fallback(args):
                traj_batch_arg, advantages_ext_arg, last_agent_positions_arg, lambda_current = args
                dummy_stats = {
                    "lambda": lambda_current,
                    "A_ext_mean": jnp.array(0.0),
                    "A_ext_std": jnp.array(0.0),
                    "A_int_mean": jnp.array(0.0),
                    "A_int_std": jnp.array(0.0),
                    "A_combined_mean": jnp.array(0.0),
                    "A_combined_std": jnp.array(0.0),
                }
                mean_adv_ext = advantages_ext_arg.mean()
                return advantages_ext_arg, dummy_stats, jnp.zeros_like(advantages_ext_arg), jnp.zeros_like(advantages_ext_arg), mean_adv_ext

            advantages, lagrangian_stats, targets_v_int, v_int_old, mean_adv_ext = jax.lax.cond(
                use_lagrangian,
                _compute_lagrangian_gae,
                _compute_standard_gae_fallback,
                (traj_batch, advantages_ext, last_agent_positions, lambda_t),
            )

            lambda_new = jax.lax.cond(
                use_lagrangian,
                lambda: update_lambda_pure(lambda_t, mean_adv_ext, tc.lambda_lr, tc.lambda_max),
                lambda: lambda_t,
            )

            def _update_epoch(update_state, unused):
                (train_states_upd, init_hstates, v_int_hstate_upd, traj_batch_upd,
                 advantages_upd, targets_upd, targets_v_int_upd, v_int_old_upd, rng_upd) = update_state
                rng_upd, _rng = jax.random.split(rng_upd)

                actor_train_states_upd, critic_train_state_upd, v_int_train_state_upd = train_states_upd

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

                def _v_int_loss_fn(v_int_params, v_int_hstate, agent_positions_inner, agent_idx_inner, dones_inner, targets_inner, v_int_old_inner):
                    num_steps_inner = agent_positions_inner.shape[0]
                    num_actors_inner = agent_positions_inner.shape[1]
                    s_encoder_apply = tdd_s_encoder_states[0].apply_fn
                    stacked_enc_params = stack_params([s.params for s in tdd_s_encoder_states])

                    def _encode_with_agent_idx(pos, agent_id):
                        return s_encoder_apply(jax.tree.map(lambda x: x[agent_id], stacked_enc_params), pos)

                    flat_positions = agent_positions_inner.reshape(-1, 2)
                    flat_agent_idx = agent_idx_inner.reshape(-1).astype(jnp.int32)
                    s_enc_embeddings_inner = jax.vmap(_encode_with_agent_idx)(flat_positions, flat_agent_idx)
                    s_enc_embeddings_inner = jax.lax.stop_gradient(s_enc_embeddings_inner.reshape(num_steps_inner, num_actors_inner, -1))

                    v_int_in = (s_enc_embeddings_inner, dones_inner)
                    _, v_int_pred = v_int_network.apply(v_int_params, v_int_hstate, v_int_in)
                    v_int_clipped = v_int_old_inner + (v_int_pred - v_int_old_inner).clip(-tc.clip_eps, tc.clip_eps)
                    v_int_losses = jnp.square(v_int_pred - targets_inner)
                    v_int_losses_clipped = jnp.square(v_int_clipped - targets_inner)
                    v_int_loss_raw = 0.5 * jnp.maximum(v_int_losses, v_int_losses_clipped).mean()
                    return v_int_loss_raw, v_int_loss_raw

                v_int_grad_fn = jax.value_and_grad(_v_int_loss_fn, has_aux=True)
                (v_int_loss_total, v_int_loss_raw), v_int_grads = v_int_grad_fn(
                    v_int_train_state_upd.params, v_int_hstate_upd,
                    traj_batch_upd.agent_positions, traj_batch_upd.agent_idx,
                    traj_batch_upd.global_done,
                    targets_v_int_upd, v_int_old_upd,
                )
                v_int_train_state_upd = v_int_train_state_upd.apply_gradients(grads=v_int_grads)

                actor_loss_mean = actor_losses.mean()
                entropy_mean = actor_aux[1].mean()
                ratio_mean = actor_aux[2].mean()
                approx_kl_mean = actor_aux[3].mean()
                clip_frac_mean = actor_aux[4].mean()

                total_loss = actor_loss_mean + critic_loss[0] + v_int_loss_total

                actor_grad_norms = jax.vmap(optax.global_norm)(actor_grads)
                critic_grad_norm = optax.global_norm(critic_grads)
                v_int_grad_norm = optax.global_norm(v_int_grads)

                loss_info = {
                    "total_loss": total_loss, "actor_loss": actor_loss_mean, "value_loss": critic_loss[0],
                    "v_int_loss": v_int_loss_raw, "entropy": entropy_mean, "ratio": ratio_mean,
                    "approx_kl": approx_kl_mean, "clip_frac": clip_frac_mean,
                    "actor_grad_norm": actor_grad_norms.mean(),
                    "critic_grad_norm": critic_grad_norm, "v_int_grad_norm": v_int_grad_norm,
                }

                train_states_upd = (new_actor_states, critic_train_state_upd, v_int_train_state_upd)
                update_state = (train_states_upd, init_hstates, v_int_hstate_upd,
                    traj_batch_upd, advantages_upd, targets_upd, targets_v_int_upd, v_int_old_upd, rng_upd)
                return update_state, loss_info

            train_states = (actor_train_states, critic_train_state, v_int_train_state)
            update_state = (train_states, initial_hstates, v_int_init_hstate, traj_batch, advantages, targets_ext, targets_v_int, v_int_old, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, tc.update_epochs)

            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            actor_train_states, critic_train_state, v_int_train_state = train_states

            metric = traj_batch.info
            metric["loss"] = loss_info
            metric["tdd_loss"] = tdd_loss
            metric["lagrangian"] = lagrangian_stats
            metric["rewards"] = {"extrinsic": jnp.mean(traj_batch.reward), "intrinsic_scaled": jnp.mean(traj_batch.intrinsic_reward)}
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
            metric["v_int_params"] = v_int_train_state.params
            rng = update_state[-1]
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback_handler, None, metric, ordered=True)
            update_steps = update_steps + 1
            train_states = (actor_train_states, critic_train_state, tdd_potential_states, tdd_s_encoder_states, v_int_train_state)
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)

            return (runner_state, update_steps, lambda_new), metric

        rng, _rng = jax.random.split(rng)
        train_states = (actor_train_states, critic_train_state, tdd_potential_states, tdd_s_encoder_states, v_int_train_state)
        runner_state = (train_states, env_state, obsv, jnp.zeros((tc.num_actors), dtype=bool), (ac_init_hstate, cr_init_hstate), _rng)
        lambda_init = jnp.array(tc.lambda_init)

        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0, lambda_init), None, tc.num_updates
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


@hydra.main(version_base=None, config_path="config", config_name="mappo_lagrangian_corridor")
def main(config):
    config = OmegaConf.to_container(config)
    print_training_header_pure_lagrangian(config)

    wandb_project = config.get("PROJECT", "mappo")
    env_display_name = config["ENV_NAME"].replace("/", "-").replace("_", "-")
    run_name = config.get("RUN_NAME") or f"mappo-lagrangian-{env_display_name}"
    run = wandb.init(
        project=wandb_project, name=run_name, 
        tags=["mappo", "lagrangian", "no-z", config["ENV_NAME"]],
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
