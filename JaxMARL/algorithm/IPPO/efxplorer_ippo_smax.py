import os
import sys
import logging

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import hydra
from omegaconf import OmegaConf
from typing import NamedTuple, Dict
from time import time

from jaxmarl.wrappers.baselines import SMAXLogWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

import wandb

from core import (
    ScannedRNN,
    create_intrinsic_critic_z_aug_train_state,
    create_tdd_train_states,
    batchify,
    unbatchify,
    make_linear_schedule,
    discounted_sampling,
    compute_delta_int_with_v_int,
    stack_params,
    save_params,
    roll_z_trajectory_forward,
    compute_intrinsic_reward_sd,
    mrn_distance,
    vectorized_apply_gradients,
)
from core.networks import create_actor_critic_train_state
from core.kstep_advantage import compute_joint_advantage_kstep
from core.z0_update import update_z0_from_episode_data, collect_episode_data_simple
from core.efxplorer_trainer_Kreturn import (
    create_kstep_gae_metrics,
    compute_v_int_gae_targets,
    create_z_init_metrics_kreturn,
)



class TransitionSMAX(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    action_logits: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    z_i: jnp.ndarray
    intrinsic_reward: jnp.ndarray
    agent_positions: jnp.ndarray
    agent_local_features_curr: jnp.ndarray 
    agent_idx: jnp.ndarray


class TrainerConfigIPPOSMAX(NamedTuple):
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
    vf_coef: float
    v_int_coef: float
    max_grad_norm: float
    num_updates: int
    update_epochs: int

    gamma_int: float
    z_clip_min: float
    z_clip_max: float
    z_budget_reset_on_done: bool
    z_init_budget_initial: float
    z_reset_interval: int

    joint_gae_lambda: float
    kstep_adv_clip_abs: float

    use_adaptive_z_init: bool
    use_episode_level_z0: bool
    alpha_z: float
    z0_best_reward_decay: float
    z_failure_factor: float
    seed: int

    tdd_latent_dim: int
    tdd_output_dim: int
    tdd_lr: float
    tdd_batch_size: int
    tdd_update_epochs: int
    tdd_discount: float

    intrinsic_reward_scale: float
    ext_reward_scale: float
    tdd_feature_dim: int

    v_int_gae_lambda: float

    map_width: float
    map_height: float


def create_trainer_config_ippo_smax(
    config: Dict, num_agents: int, map_width: float, map_height: float, tdd_feature_dim: int
) -> TrainerConfigIPPOSMAX:
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    return TrainerConfigIPPOSMAX(
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
        vf_coef=config.get("VF_COEF", 0.5),
        v_int_coef=config.get("V_INT_COEF", 0.5),
        max_grad_norm=config["MAX_GRAD_NORM"],
        num_updates=config["NUM_UPDATES"],
        update_epochs=config["UPDATE_EPOCHS"],
        gamma_int=gamma,
        z_clip_min=config.get("Z_CLIP_MIN", -100.0),
        z_clip_max=config.get("Z_CLIP_MAX", 100.0),
        z_budget_reset_on_done=config.get("Z_BUDGET_RESET_ON_DONE", True),
        z_init_budget_initial=config.get("Z_INIT_BUDGET", 10.0),
        z_reset_interval=config.get("Z_RESET_INTERVAL", 0),
        joint_gae_lambda=config.get("JOINT_GAE_LAMBDA", gae_lambda),
        kstep_adv_clip_abs=config.get("KSTEP_ADV_CLIP_ABS", 20.0),
        use_adaptive_z_init=config.get("USE_ADAPTIVE_Z_INIT", False),
        use_episode_level_z0=config.get("USE_EPISODE_LEVEL_Z0", True),
        alpha_z=config.get("ALPHA_Z", 0.1),
        z0_best_reward_decay=config.get("Z0_BEST_REWARD_DECAY", 0.99),
        z_failure_factor=config.get(
            "Z_FAILURE_FACTOR", 0.0 if config.get("Z_RESET_TO_ZERO_ON_FAILURE", False) else 1.0
        ),
        seed=config["SEED"],
        tdd_latent_dim=config.get("TDD_LATENT_DIM", 64),
        tdd_output_dim=config.get("TDD_OUTPUT_DIM", 32),
        tdd_lr=config.get("TDD_LR", 1e-3),
        tdd_batch_size=config.get("TDD_BATCH_SIZE", 512),
        tdd_update_epochs=config.get("TDD_UPDATE_EPOCHS", 144),
        tdd_discount=config.get("TDD_DISCOUNT", 0.95),
        intrinsic_reward_scale=config.get("INTRINSIC_REWARD_SCALE", 1.0),
        ext_reward_scale=config.get("EXT_REWARD_SCALE", 100.0),
        tdd_feature_dim=tdd_feature_dim,
        v_int_gae_lambda=config.get("V_INT_GAE_LAMBDA", gae_lambda),
        map_width=map_width,
        map_height=map_height,
    )


def print_training_header_ippo_smax(config: Dict):
    print("\n" + "=" * 80)
    print("EFXPLORER K-Return IPPO - SMAX (shared actor-critic backbone, local obs)")
    print("=" * 80)
    print(f"Map: {config['MAP_NAME']}")
    print(f"Seed: {config['SEED']}")
    print(f"Total Timesteps: {config['TOTAL_TIMESTEPS']:,.0f}")
    print(f"Num Envs: {config['NUM_ENVS']} | Num Steps: {config['NUM_STEPS']}")
    print(f"Update Epochs: {config['UPDATE_EPOCHS']}")
    print("=" * 80 + "\n")


class SMAXCallbackHandler:
    def __init__(self, config: Dict, num_agents: int):
        self.config = config
        self.num_agents = num_agents
        self.num_envs = config["NUM_ENVS"]
        self.num_steps = config["NUM_STEPS"]
        self.wandb_enabled = config.get("WANDB_MODE", "disabled") != "disabled"

    def __call__(self, metric):
        update_steps = int(np.array(metric["update_steps"]))
        env_step = update_steps * self.num_envs * self.num_steps

        returned_episode = metric["returned_episode"][:, :, 0]
        returns = metric["returned_episode_returns"][:, :, 0][returned_episode].mean()
        win_rate = metric["returned_won_episode"][:, :, 0][returned_episode].mean()

        loss_info = metric.get("loss", {})
        reward_info = metric.get("rewards", {})
        z_init_info = metric.get("z_init", {})
        kstep_info = metric.get("kstep", {})

        if self.wandb_enabled:
            log_dict = {
                "returns": float(np.array(returns)),
                "win_rate": float(np.array(win_rate)),
                "env_step": env_step,
                "reward/extrinsic": float(np.array(reward_info.get("extrinsic", 0.0))),
                "reward/intrinsic_scaled": float(np.array(reward_info.get("intrinsic_scaled", 0.0))),
                "loss/total_loss": float(np.array(loss_info.get("total_loss", 0.0))),
                "loss/actor_loss": float(np.array(loss_info.get("actor_loss", 0.0))),
                "loss/value_loss": float(np.array(loss_info.get("value_loss", 0.0))),
                "loss/v_int_loss": float(np.array(loss_info.get("v_int_loss", 0.0))),
                "loss/entropy": float(np.array(loss_info.get("entropy", 0.0))),
                "loss/ratio": float(np.array(loss_info.get("ratio", 1.0))),
                "loss/approx_kl": float(np.array(loss_info.get("approx_kl", 0.0))),
                "loss/clip_frac": float(np.array(loss_info.get("clip_frac", 0.0))),
                "loss/ac_grad_norm": float(np.array(loss_info.get("ac_grad_norm", 0.0))),
                "loss/v_int_grad_norm": float(np.array(loss_info.get("v_int_grad_norm", 0.0))),
                "tdd/loss": float(np.array(metric.get("tdd_loss", 0.0))),
            }
            for k, v in z_init_info.items():
                log_dict[f"z_init/{k}"] = float(np.array(v).mean()) if isinstance(v, (jnp.ndarray, np.ndarray)) else float(v)
            for k, v in kstep_info.items():
                log_dict[f"kstep/{k}"] = float(np.array(v).mean()) if isinstance(v, (jnp.ndarray, np.ndarray)) else float(v)
            try:
                wandb.log(log_dict, step=update_steps)
            except Exception:
                pass

        nu = self.config["TOTAL_TIMESTEPS"] // self.num_steps // self.num_envs
        print(
            f"[EFXPLORER-IPPO-SMAX] {update_steps:4d}/{nu} | step {env_step:8d} | "
            f"ret {float(np.array(returns)):8.2f} | win {float(np.array(win_rate)):5.1%}"
        )


def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config.get("ENV_KWARGS", {}))

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    if config.get("SCALE_CLIP_EPS", False):
        config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents

    own_features_dim = len(env.own_features)
    action_dim_pre = env.action_space(env.agents[0]).n
    tdd_feature_dim = own_features_dim + action_dim_pre
    tc = create_trainer_config_ippo_smax(
        config, env.num_agents, float(env.map_width), float(env.map_height), tdd_feature_dim
    )
    env = SMAXLogWrapper(env)

    lr_base = config.get("LR") or config.get("ACTOR_LR", 0.001)
    anneal_lr = config.get("ANNEAL_LR", False)
    ac_lr_schedule = (
        make_linear_schedule(lr_base, 1, tc.update_epochs, tc.num_updates) if anneal_lr else None
    )

    def train(rng):
        rng, _rng_ac, _rng_tdd, _rng_v = jax.random.split(rng, 4)
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        action_dim = env.action_space(env.agents[0]).n

        ac_network, ac_train_state = create_actor_critic_train_state(
            rng=_rng_ac,
            config=config,
            action_dim=action_dim,
            obs_dim=obs_dim,
            lr_schedule=ac_lr_schedule,
        )

        tdd_potential_states_raw, tdd_s_encoder_states_raw = create_tdd_train_states(
            rng=_rng_tdd, config=config, num_agents=tc.num_agents, input_dim=tc.tdd_feature_dim,
        )
        tdd_potential_states = tuple(tdd_potential_states_raw)
        tdd_s_encoder_states = tuple(tdd_s_encoder_states_raw)

        v_int_network, v_int_train_state = create_intrinsic_critic_z_aug_train_state(
            rng=_rng_v,
            config=config,
            s_encoder_output_dim=tc.tdd_output_dim,
            num_actors=tc.num_actors,
            lr_schedule=None,
        )
        v_int_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config["GRU_HIDDEN_DIM"])

        own_features_dim = tc.tdd_feature_dim - action_dim
        print(
            f"K-step (lambda={tc.joint_gae_lambda}, clip={tc.kstep_adv_clip_abs}) | fused ActorCritic | "
            f"VF_COEF={tc.vf_coef} | TDD features own_obs+action (dim={tc.tdd_feature_dim})"
        )
        if tc.use_episode_level_z0:
            print(f"z_init episode-level decay={tc.z0_best_reward_decay}")
        elif tc.use_adaptive_z_init:
            print(f"z_init adaptive alpha={tc.alpha_z}")
        else:
            print(f"z_init fixed {tc.z_init_budget_initial}")

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, tc.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config["GRU_HIDDEN_DIM"])

        callback_handler = SMAXCallbackHandler(config=config, num_agents=tc.num_agents)
        agent_idx_array = jnp.repeat(jnp.arange(tc.num_agents), tc.num_envs)

        def _update_step(update_runner_state, unused):
            runner_state, update_steps, z_prev, z_init_budget_current, best_ext_reward = update_runner_state

            current_ent_coef = jax.lax.cond(
                tc.anneal_ent,
                lambda: tc.ent_coef - (tc.ent_coef - tc.ent_coef_end) * (update_steps / tc.num_updates),
                lambda: jnp.float32(tc.ent_coef),
            )
            intrinsic_scale = jnp.float32(tc.intrinsic_reward_scale)

            reset_z = jnp.logical_and(
                tc.z_reset_interval > 0,
                jnp.logical_and(update_steps > 0, jnp.equal(jnp.mod(update_steps, tc.z_reset_interval), 0)),
            )
            z_init_budget_current = jax.lax.cond(
                reset_z, lambda: jnp.zeros_like(z_init_budget_current), lambda: z_init_budget_current
            )

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstate, rng = runner_state
                ac_state, _, _, _ = train_states
                rng, _rng = jax.random.split(rng)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions, env.agents, tc.num_actors))
                obs_batch = batchify(last_obs, env.agents, tc.num_actors)

                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                new_h, action_logits, value = ac_network.apply(ac_state.params, hstate, ac_in)
                action_logits = action_logits.squeeze(0)
                value = value.squeeze(0)

                masked_logits = action_logits - (1 - avail_actions) * 1e10
                pi = distrax.Categorical(logits=masked_logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                env_act = unbatchify(action, env.agents, tc.num_envs, tc.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, tc.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree.map(lambda x: x.reshape((tc.num_actors)), info)
                done_batch = batchify(done, env.agents, tc.num_actors).squeeze()

                smax_state = env_state.env_state.state
                ally_pos = smax_state.unit_positions[:, :tc.num_agents, :]
                ally_norm = ally_pos / jnp.array([tc.map_width, tc.map_height])
                agent_positions = ally_norm.swapaxes(0, 1).reshape(tc.num_actors, 2)

                own_feats = obs_batch[:, -own_features_dim:]
                action_oh = jax.nn.one_hot(action.squeeze(), num_classes=action_dim)
                tdd_features = jnp.concatenate([own_feats, action_oh], axis=-1)

                z_ph = jnp.zeros((tc.num_actors,), dtype=jnp.float32)
                transition = TransitionSMAX(
                    jnp.tile(done["__all__"], tc.num_agents),
                    last_done,
                    action.squeeze(),
                    masked_logits,
                    value,
                    batchify(reward, env.agents, tc.num_actors).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                    z_ph,
                    z_ph,
                    agent_positions,
                    tdd_features,
                    agent_idx_array,
                )
                runner_state = (train_states, env_state, obsv, done_batch, new_h, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, tc.num_steps)

            train_states, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_train_state, tdd_potential_states, tdd_s_encoder_states, v_int_train_state = train_states

            last_obs_batch = batchify(last_obs, env.agents, tc.num_actors)
            _, _, last_val = ac_network.apply(
                ac_train_state.params, hstate, (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :])
            )
            last_val = last_val.squeeze()

            last_ally = env_state.env_state.state.unit_positions[:, :tc.num_agents, :]
            last_norm = last_ally / jnp.array([tc.map_width, tc.map_height])
            last_agent_positions = last_norm.swapaxes(0, 1).reshape(tc.num_actors, 2)

            tdd_features_traj = traj_batch.agent_local_features_curr
            agent_features_curr_all = tdd_features_traj.reshape(
                tc.num_steps, tc.num_agents, tc.num_envs, tc.tdd_feature_dim
            ).transpose(1, 0, 2, 3)
            agent_features_next_all = jnp.concatenate(
                [agent_features_curr_all[:, 1:, :, :], agent_features_curr_all[:, -1:, :, :]], axis=1
            )
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
                    mrn_dists = mrn_distance(phi_s[:, None, :], phi_g[None, :, :])
                    logits = c_g.squeeze(-1)[None, :] - mrn_dists
                    log_sm = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
                    loss1 = jnp.sum(-(log_sm * I).sum(axis=-1) * valid_mask_float) / jnp.maximum(num_valid, 1)
                    log_sm_T = logits.T - jax.scipy.special.logsumexp(logits.T, axis=-1, keepdims=True)
                    loss2 = jnp.sum(-(log_sm_T * I).sum(axis=-1) * valid_mask_float) / jnp.maximum(num_valid, 1)
                    return (loss1 + loss2) / 2

                def _loss_grad(pot_params, enc_params, feats, rng_a):
                    loss, (pg, eg) = jax.value_and_grad(_tdd_loss_single_agent, argnums=(0, 1))(
                        pot_params, enc_params, feats, rng_a
                    )
                    return loss, pg, eg

                losses, pg, eg = jax.vmap(_loss_grad)(
                    stacked_potential_params, stacked_s_encoder_params, agent_features_curr_all, agent_rngs
                )
                new_pot = vectorized_apply_gradients(list(potential_states_inner), pg)
                new_enc = vectorized_apply_gradients(list(s_encoder_states_inner), eg)
                return (tuple(new_pot), tuple(new_enc)), losses.mean()

            def _tdd_epoch(carry, _):
                ts, rng_e = carry
                rng_e, rs = jax.random.split(rng_e)
                nts, loss = _tdd_update_step(ts, rs)
                return (nts, rng_e), loss

            (new_tdd, _), tdd_losses = jax.lax.scan(
                _tdd_epoch, ((tdd_potential_states, tdd_s_encoder_states), _rng_tdd), None, tc.tdd_update_epochs
            )
            tdd_loss = tdd_losses.mean()
            tdd_potential_states, tdd_s_encoder_states = new_tdd[0], new_tdd[1]

            intrinsic_reward_per_agent = compute_intrinsic_reward_sd(
                agent_features_curr_all,
                agent_features_next_all,
                tdd_s_encoder_states,
                tdd_shared=False,
                num_agents=tc.num_agents,
                num_envs=tc.num_envs,
                num_steps=tc.num_steps,
            )
            intrinsic_reward_all = intrinsic_reward_per_agent.transpose(1, 0, 2).reshape(tc.num_steps, tc.num_actors)
            intrinsic_reward_raw = intrinsic_reward_all
            intrinsic_reward_all = jax.lax.stop_gradient(intrinsic_reward_all * intrinsic_scale)

            dones_all = traj_batch.global_done
            z_traj = roll_z_trajectory_forward(
                intrinsic_reward_all,
                dones_all,
                z_init_budget_current,
                tc.num_steps,
                tc.num_agents,
                tc.num_envs,
                tc.gamma_int,
                tc.z_clip_min,
                tc.z_clip_max,
                tc.z_budget_reset_on_done,
            )
            z_flat = z_traj.reshape(tc.num_steps, tc.num_agents * tc.num_envs)
            traj_batch = traj_batch._replace(z_i=z_flat, intrinsic_reward=intrinsic_reward_all)
            z_prev_new = z_flat

            if tc.use_episode_level_z0:
                r_int_r = intrinsic_reward_all.reshape(tc.num_steps, tc.num_agents, tc.num_envs)
                r_ext_r = (traj_batch.reward * tc.ext_reward_scale).reshape(tc.num_steps, tc.num_agents, tc.num_envs)
                ext_sums, z0_sums = collect_episode_data_simple(
                    intrinsic_rewards=r_int_r, extrinsic_rewards=r_ext_r, gamma=tc.gamma_int
                )
                z_init_budget_new, best_ext_reward_new, z0_update_info = update_z0_from_episode_data(
                    ext_episode_sums=ext_sums,
                    z0_episode_sums=z0_sums,
                    z_0_ema=z_init_budget_current,
                    best_ext_reward=best_ext_reward,
                    z_ema_alpha=tc.alpha_z,
                    z0_best_reward_decay=tc.z0_best_reward_decay,
                    z_failure_factor=tc.z_failure_factor,
                )
                z_init_budget_new = jnp.clip(z_init_budget_new, 0.1, 1000.0)
            elif tc.use_adaptive_z_init:
                r_int_r = intrinsic_reward_all.reshape(tc.num_steps, tc.num_agents, tc.num_envs)
                g_int = jnp.sum(r_int_r, axis=(0, 2)) / tc.num_envs
                z_init_budget_new = jnp.where(
                    update_steps == 0,
                    g_int,
                    tc.alpha_z * g_int + (1 - tc.alpha_z) * z_init_budget_current,
                )
                z_init_budget_new = jnp.clip(z_init_budget_new, 0.1, 1000.0)
                best_ext_reward_new = best_ext_reward
                z0_update_info = {
                    "passed_ratio": jnp.array(1.0, dtype=jnp.float32),
                    "update_delta_mean": jnp.mean(jnp.abs(z_init_budget_new - z_init_budget_current)),
                }
            else:
                z_init_budget_new = z_init_budget_current
                best_ext_reward_new = best_ext_reward
                z0_update_info = {
                    "passed_ratio": jnp.array(0.0, dtype=jnp.float32),
                    "update_delta_mean": jnp.array(0.0, dtype=jnp.float32),
                }

            r_int, z_i = traj_batch.intrinsic_reward, traj_batch.z_i
            s_enc_apply = tdd_s_encoder_states[0].apply_fn
            stacked_enc = stack_params([s.params for s in tdd_s_encoder_states])
            tdd_features_reshaped = tdd_features_traj.reshape(
                tc.num_steps, tc.num_agents, tc.num_envs, tc.tdd_feature_dim
            )

            def _enc1(p, seq):
                flat = seq.reshape(-1, tc.tdd_feature_dim)
                emb = jax.vmap(lambda feat: s_enc_apply(p, feat))(flat)
                return emb.reshape(tc.num_steps, tc.num_envs, -1)

            s_enc_pa = jax.vmap(_enc1)(stacked_enc, tdd_features_reshaped.transpose(1, 0, 2, 3))
            s_enc = jax.lax.stop_gradient(s_enc_pa.transpose(1, 0, 2, 3).reshape(tc.num_steps, tc.num_actors, -1))

            _, v_int_all = v_int_network.apply(v_int_train_state.params, v_int_init_hstate, (s_enc, z_i, traj_batch.global_done))

            last_tdd_features = tdd_features_traj[-1]
            last_feat_reshaped = last_tdd_features.reshape(tc.num_agents, tc.num_envs, tc.tdd_feature_dim)

            def _enc_last(p, ap):
                return jax.vmap(lambda feat: s_enc_apply(p, feat))(ap)

            last_enc_pa = jax.vmap(_enc_last)(stacked_enc, last_feat_reshaped)
            last_s_enc = jax.lax.stop_gradient(last_enc_pa.reshape(tc.num_actors, -1))

            last_z, last_r, last_d = z_i[-1], r_int[-1], traj_batch.global_done[-1]
            z_next_last = jnp.clip((last_z - last_r) / tc.gamma_int, tc.z_clip_min, tc.z_clip_max)
            z_init_flat = jnp.broadcast_to(z_init_budget_current[:, None], (tc.num_agents, tc.num_envs)).reshape(-1)
            z_next_last = jnp.where(last_d & tc.z_budget_reset_on_done, z_init_flat, z_next_last)

            _, last_v_int = v_int_network.apply(
                v_int_train_state.params,
                v_int_init_hstate,
                (last_s_enc[None, :], z_next_last[None, :], traj_batch.global_done[-1:]),
            )
            last_v_int = last_v_int.squeeze(0)

            z_next = jnp.concatenate([z_i[1:], z_next_last[None]], axis=0)
            v_int_next_in = (
                jnp.concatenate([s_enc[1:], last_s_enc[None, :]], axis=0),
                z_next,
                jnp.concatenate([traj_batch.global_done[1:], traj_batch.global_done[-1:]], axis=0),
            )
            _, v_int_next = v_int_network.apply(v_int_train_state.params, v_int_init_hstate, v_int_next_in)
            delta_int = compute_delta_int_with_v_int(
                r_int, v_int_all, v_int_next, tc.gamma_int, traj_batch.global_done
            )
            _, targets_v_int = compute_v_int_gae_targets(
                r_int, v_int_all, last_v_int, tc.gamma_int, tc.v_int_gae_lambda, traj_batch.global_done
            )

            def _compute_delta_ext(carry, inputs):
                done, value, reward = inputs.global_done, inputs.value, inputs.reward * tc.ext_reward_scale
                delta = reward + tc.gamma * carry * (1 - done) - value
                return value, delta

            _, delta_ext_all = jax.lax.scan(_compute_delta_ext, last_val, traj_batch, reverse=True)

            advantages, kstep_stats = compute_joint_advantage_kstep(
                value_t=traj_batch.value,
                V_int_t=v_int_all,
                delta_ext_t=delta_ext_all,
                delta_int_t=delta_int,
                z_i_t=z_i,
                done_t=traj_batch.global_done,
                gamma=tc.gamma,
                lmbda=tc.joint_gae_lambda,
                adv_clip_abs=tc.kstep_adv_clip_abs,
            )
            targets = advantages + traj_batch.value

            kstep_metrics = create_kstep_gae_metrics(
                kstep_stats=kstep_stats,
                advantages=advantages,
                z_i=z_i,
                r_int=r_int,
                z_prev=z_prev,
                v_int_all=v_int_all,
                targets_v_int=targets_v_int,
                gamma_int=tc.gamma_int,
                delta_ext_all=delta_ext_all,
                delta_int_all=delta_int,
                num_agents=tc.num_agents,
            )
            v_int_old = v_int_all

            def _update_epoch(update_state, unused):
                (train_states_upd, init_hstate, v_int_hstate_upd, traj_batch_upd,
                 advantages_upd, targets_upd, targets_v_int_upd, v_int_old_upd, rng_upd) = update_state
                rng_upd, _rng = jax.random.split(rng_upd)
                ac_state_upd, v_int_train_state_upd = train_states_upd

                def _ac_loss_fn(params, init_hstate_inner, traj_batch_inner, gae, targets_inner):
                    _, action_logits, value = ac_network.apply(
                        params, init_hstate_inner.squeeze(),
                        (traj_batch_inner.obs, traj_batch_inner.done),
                    )
                    masked_logits = action_logits - (1 - traj_batch_inner.avail_actions) * 1e10
                    pi = distrax.Categorical(logits=masked_logits)
                    log_prob = pi.log_prob(traj_batch_inner.action)
                    value_pred_clipped = traj_batch_inner.value + (
                        value - traj_batch_inner.value
                    ).clip(-tc.clip_eps, tc.clip_eps)
                    value_losses = jnp.square(value - targets_inner)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets_inner)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    logratio = log_prob - traj_batch_inner.log_prob
                    ratio = jnp.exp(logratio)
                    gae_normalized = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae_normalized
                    loss_actor2 = jnp.clip(ratio, 1.0 - tc.clip_eps, 1.0 + tc.clip_eps) * gae_normalized
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = pi.entropy().mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_frac = jnp.mean(jnp.abs(ratio - 1) > tc.clip_eps)
                    total = loss_actor + tc.vf_coef * value_loss - current_ent_coef * entropy
                    return total, (loss_actor, value_loss, entropy, ratio, approx_kl, clip_frac)

                init_hstate_reshaped = jax.tree.map(
                    lambda x: jnp.reshape(x, (1, tc.num_actors, -1)), init_hstate
                )
                ac_grad_fn = jax.value_and_grad(_ac_loss_fn, has_aux=True)
                (ac_total_loss, ac_aux), ac_grads = ac_grad_fn(
                    ac_state_upd.params, init_hstate_reshaped,
                    traj_batch_upd, advantages_upd, targets_upd,
                )
                ac_state_upd = ac_state_upd.apply_gradients(grads=ac_grads)
                actor_loss_val, value_loss_val, entropy_mean, ratio_arr, approx_kl_mean, clip_frac_mean = ac_aux
                ratio_mean = ratio_arr.mean()
                ac_grad_norm = optax.global_norm(ac_grads)

                def _v_int_loss_fn(v_int_params, v_int_hstate, tdd_feat_inner, agent_idx_inner, z_inner, dones_inner, targets_inner_vi, v_int_old_inner):
                    num_steps_inner = tdd_feat_inner.shape[0]
                    num_actors_inner = tdd_feat_inner.shape[1]
                    s_encoder_apply = tdd_s_encoder_states[0].apply_fn
                    stacked_enc_params = stack_params([s.params for s in tdd_s_encoder_states])

                    def _encode_with_agent_idx(feat, agent_id):
                        return s_encoder_apply(jax.tree.map(lambda x: x[agent_id], stacked_enc_params), feat)

                    flat_feat = tdd_feat_inner.reshape(-1, tc.tdd_feature_dim)
                    flat_agent_idx = agent_idx_inner.reshape(-1).astype(jnp.int32)
                    s_enc_embeddings_inner = jax.vmap(_encode_with_agent_idx)(flat_feat, flat_agent_idx)
                    s_enc_embeddings_inner = jax.lax.stop_gradient(
                        s_enc_embeddings_inner.reshape(num_steps_inner, num_actors_inner, -1)
                    )
                    v_int_in_loc = (s_enc_embeddings_inner, z_inner, dones_inner)
                    _, v_int_pred = v_int_network.apply(v_int_params, v_int_hstate, v_int_in_loc)
                    v_int_clipped = v_int_old_inner + (v_int_pred - v_int_old_inner).clip(-tc.clip_eps, tc.clip_eps)
                    v_int_losses = jnp.square(v_int_pred - targets_inner_vi)
                    v_int_losses_clipped = jnp.square(v_int_clipped - targets_inner_vi)
                    v_int_loss_raw = 0.5 * jnp.maximum(v_int_losses, v_int_losses_clipped).mean()
                    return tc.v_int_coef * v_int_loss_raw, v_int_loss_raw

                v_int_grad_fn = jax.value_and_grad(_v_int_loss_fn, has_aux=True)
                (v_int_loss_total, v_int_loss_raw), v_int_grads = v_int_grad_fn(
                    v_int_train_state_upd.params, v_int_hstate_upd,
                    traj_batch_upd.agent_local_features_curr, traj_batch_upd.agent_idx,
                    traj_batch_upd.z_i, traj_batch_upd.global_done,
                    targets_v_int_upd, v_int_old_upd,
                )
                v_int_train_state_upd = v_int_train_state_upd.apply_gradients(grads=v_int_grads)
                v_int_grad_norm = optax.global_norm(v_int_grads)

                total_loss = ac_total_loss + v_int_loss_total
                loss_info = {
                    "total_loss": total_loss,
                    "actor_loss": actor_loss_val,
                    "value_loss": value_loss_val,
                    "v_int_loss": v_int_loss_raw,
                    "entropy": entropy_mean,
                    "ratio": ratio_mean,
                    "approx_kl": approx_kl_mean,
                    "clip_frac": clip_frac_mean,
                    "ac_grad_norm": ac_grad_norm,
                    "v_int_grad_norm": v_int_grad_norm,
                }
                train_states_upd = (ac_state_upd, v_int_train_state_upd)
                update_state_out = (
                    train_states_upd, init_hstate, v_int_hstate_upd,
                    traj_batch_upd, advantages_upd, targets_upd, targets_v_int_upd, v_int_old_upd, rng_upd,
                )
                return update_state_out, loss_info

            train_states_inner = (ac_train_state, v_int_train_state)
            update_state = (
                train_states_inner, initial_hstate, v_int_init_hstate, traj_batch,
                advantages, targets, targets_v_int, v_int_old, rng,
            )
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, tc.update_epochs)

            loss_info["ratio_0"] = loss_info["ratio"].at[0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states_inner = update_state[0]
            ac_train_state, v_int_train_state = train_states_inner

            metric = jax.tree.map(
                lambda x: x.reshape((tc.num_steps, tc.num_envs, tc.num_agents)),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            metric["tdd_loss"] = tdd_loss
            metric["kstep"] = kstep_metrics
            r_ext = traj_batch.reward
            r_int_scaled = traj_batch.intrinsic_reward
            r_int_raw = intrinsic_reward_raw
            r_ext_mean = jnp.mean(r_ext)
            r_int_scaled_mean = jnp.mean(r_int_scaled)
            r_int_raw_mean = jnp.mean(r_int_raw)
            r_ext_abs_mean = jnp.maximum(jnp.abs(r_ext_mean), 1e-8)
            metric["rewards"] = {
                "extrinsic": r_ext_mean,
                "extrinsic_min": jnp.min(r_ext),
                "extrinsic_max": jnp.max(r_ext),
                "extrinsic_std": jnp.std(r_ext),
                "intrinsic_scaled": r_int_scaled_mean,
                "intrinsic_scaled_min": jnp.min(r_int_scaled),
                "intrinsic_scaled_max": jnp.max(r_int_scaled),
                "intrinsic_scaled_std": jnp.std(r_int_scaled),
                "intrinsic_raw": r_int_raw_mean,
                "intrinsic_raw_min": jnp.min(r_int_raw),
                "intrinsic_raw_max": jnp.max(r_int_raw),
                "intrinsic_raw_std": jnp.std(r_int_raw),
                "int_ext_ratio_scaled": r_int_scaled_mean / r_ext_abs_mean,
                "int_ext_ratio_raw": r_int_raw_mean / r_ext_abs_mean,
            }
            metric["ent_coef"] = current_ent_coef
            metric["intrinsic_reward_scale"] = intrinsic_scale
            metric["z_init"] = create_z_init_metrics_kreturn(
                z_init_budget_new, best_ext_reward_new, z0_update_info
            )
            rng = update_state[-1]
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback_handler, None, metric, ordered=True)
            update_steps = update_steps + 1
            train_states = (
                ac_train_state, tdd_potential_states, tdd_s_encoder_states, v_int_train_state
            )
            runner_state = (train_states, env_state, last_obs, last_done, hstate, rng)

            return (
                runner_state, update_steps, z_prev_new, z_init_budget_new, best_ext_reward_new
            ), metric

        rng, _rng = jax.random.split(rng)
        train_states = (
            ac_train_state, tdd_potential_states, tdd_s_encoder_states, v_int_train_state
        )
        runner_state = (
            train_states, env_state, obsv, jnp.zeros((tc.num_actors), dtype=bool), ac_init_hstate, _rng
        )
        z_prev_init = jnp.zeros((tc.num_steps, tc.num_actors))
        z_init_budget_init = jnp.full((tc.num_agents,), tc.z_init_budget_initial)
        best_ext_reward_init = jnp.array(-jnp.inf)

        runner_state, metric = jax.lax.scan(
            _update_step,
            (runner_state, 0, z_prev_init, z_init_budget_init, best_ext_reward_init),
            None,
            tc.num_updates,
        )

        return {"runner_state": runner_state}

    return train


def save_ippo_smax_checkpoints(run, config: Dict, out: Dict) -> None:
    import os
    import yaml

    if config.get("WANDB_MODE") == "disabled" or not hasattr(run, "dir"):
        return
    save_dir = os.path.join(run.dir, "params")
    os.makedirs(save_dir, exist_ok=True)
    carry = out["runner_state"]
    inner = carry[0]
    train_states = inner[0]
    ac_train_state, tdd_potential_states, tdd_s_encoder_states, v_int_train_state = train_states
    save_params(ac_train_state.params, os.path.join(save_dir, "actor_critic_params.safetensors"))
    all_potential_params = [s.params for s in tdd_potential_states]
    stacked_potential_params = jax.tree.map(lambda *xs: jnp.stack(xs), *all_potential_params)
    save_params(stacked_potential_params, os.path.join(save_dir, "tdd_potential_params.safetensors"))
    all_s_encoder_params = [s.params for s in tdd_s_encoder_states]
    stacked_s_encoder_params = jax.tree.map(lambda *xs: jnp.stack(xs), *all_s_encoder_params)
    save_params(stacked_s_encoder_params, os.path.join(save_dir, "tdd_s_encoder_params.safetensors"))
    save_params(v_int_train_state.params, os.path.join(save_dir, "v_int_params.safetensors"))
    metadata = {"fused_actor_critic": True, "num_agents": len(tdd_potential_states), "has_v_int": True}
    metadata_path = os.path.join(save_dir, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Checkpoints saved to {save_dir}")
    if hasattr(run, "log_artifact"):
        artifact = wandb.Artifact(f"{run.name}-checkpoint", type="checkpoint")
        for fn in (
            "actor_critic_params.safetensors",
            "tdd_potential_params.safetensors",
            "tdd_s_encoder_params.safetensors",
            "v_int_params.safetensors",
        ):
            artifact.add_file(os.path.join(save_dir, fn))
        artifact.add_file(metadata_path)
        artifact.add_file(config_path)
        run.log_artifact(artifact)


def convert_wandb_args_to_hydra():
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith("--") and "=" in arg:
            new_argv.append(arg[2:])
        else:
            new_argv.append(arg)
    sys.argv = new_argv


convert_wandb_args_to_hydra()


@hydra.main(version_base=None, config_path="config", config_name="efxplorer_ippo_smax")
def main(config):
    config = OmegaConf.to_container(config)
    print_training_header_ippo_smax(config)

    wandb_entity = config.get("ENTITY") or None
    wandb_project = config.get("PROJECT", "efxplorer-ippo-smax")
    map_name = config["MAP_NAME"]
    run_name = config.get("RUN_NAME") or f"efxplorer-ippo-smax-{map_name}"
    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=run_name,
        tags=["efxplorer", "IPPO", "Kreturn", "SMAX", "fused-actor-critic", config["ENV_NAME"]],
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
            print(
                f"\n{'=' * 80}\nTraining completed in {elapsed_time:.2f} seconds "
                f"({elapsed_time / 60:.2f} minutes)\n{'=' * 80}"
            )
    except KeyboardInterrupt:
        print("\n> Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n> Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise

    save_ippo_smax_checkpoints(run, config, out)


if __name__ == "__main__":
    main()
