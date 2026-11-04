from typing import NamedTuple, Tuple, Any, Dict, Optional
import os
import yaml
import jax
import jax.numpy as jnp
import distrax
from functools import partial

from .intrinsic_td import compute_delta_int_with_v_int
from .z_dynamics import roll_z_trajectory_forward, ZDynamicsMetrics
from .data_utils import Transition, discounted_sampling
from .tdd_networks import mrn_distance
from .utils import stack_params


class TrainerConfigKreturn(NamedTuple):
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
    num_minibatches: int

    gamma_int: float
    z_clip_min: float
    z_clip_max: float
    z_budget_reset_on_done: bool
    z_init_budget_initial: float

    joint_gae_lambda: float
    kstep_adv_clip_abs: float

    use_adaptive_z_init: bool
    use_episode_level_z0: bool
    alpha_z: float
    z0_best_reward_decay: float
    warmup_rollouts: int
    seed: int

    tdd_shared: bool
    tdd_latent_dim: int
    tdd_output_dim: int
    tdd_lr: float
    tdd_batch_size: int
    tdd_update_epochs: int
    tdd_discount: float

    intrinsic_reward_scale: float

    v_int_gae_lambda: float


class TrainerStateKreturn(NamedTuple):
    actor_train_states: Any
    critic_train_state: Any
    v_int_train_state: Any
    tdd_potential_states: Any
    tdd_s_encoder_states: Any

    z_init_budget: jnp.ndarray
    best_ext_reward: jnp.ndarray
    z_prev: jnp.ndarray


def create_trainer_config_kreturn(config: Dict, num_agents: int) -> TrainerConfigKreturn:
    gamma = config["GAMMA"]
    gae_lambda = config["GAE_LAMBDA"]
    return TrainerConfigKreturn(
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
        vf_coef=config["VF_COEF"],
        v_int_coef=config.get("V_INT_COEF", 0.5),
        max_grad_norm=config["MAX_GRAD_NORM"],
        num_updates=config["NUM_UPDATES"],
        update_epochs=config["UPDATE_EPOCHS"],
        num_minibatches=config["NUM_MINIBATCHES"],

        gamma_int=gamma,
        z_clip_min=config.get("Z_CLIP_MIN", -100.0),
        z_clip_max=config.get("Z_CLIP_MAX", 100.0),
        z_budget_reset_on_done=config.get("Z_BUDGET_RESET_ON_DONE", True),
        z_init_budget_initial=config.get("Z_INIT_BUDGET", 10.0),

        joint_gae_lambda=config.get("JOINT_GAE_LAMBDA", gae_lambda),
        kstep_adv_clip_abs=config.get("KSTEP_ADV_CLIP_ABS", 20.0),

        use_adaptive_z_init=config.get("USE_ADAPTIVE_Z_INIT", True),
        use_episode_level_z0=config.get("USE_EPISODE_LEVEL_Z0", True),
        alpha_z=config.get("ALPHA_Z", 0.1),
        z0_best_reward_decay=config.get("Z0_BEST_REWARD_DECAY", 0.99),
        warmup_rollouts=config.get("WARMUP_ROLLOUTS", 3),
        seed=config["SEED"],

        tdd_shared=config.get("TDD_SHARED_PARAMS", True),
        tdd_latent_dim=config.get("TDD_LATENT_DIM", 64),
        tdd_output_dim=config.get("TDD_OUTPUT_DIM", 32),
        tdd_lr=config.get("TDD_LR", 1e-3),
        tdd_batch_size=config.get("TDD_BATCH_SIZE", 512),
        tdd_update_epochs=config.get("TDD_UPDATE_EPOCHS", 144),
        tdd_discount=config.get("TDD_DISCOUNT", 0.967),

        intrinsic_reward_scale=config.get("INTRINSIC_REWARD_SCALE", 1.0),

        v_int_gae_lambda=config.get("V_INT_GAE_LAMBDA", gae_lambda),
    )


def compute_tdd_loss_shared(
    potential_apply_fn, s_encoder_apply_fn,
    potential_params, s_encoder_params,
    agent_features, agent_rng,
    batch_size: int, num_steps: int, num_envs: int,
    tdd_discount: float,
):
    I = jnp.eye(batch_size)
    rng_t, rng_env, rng_goal = jax.random.split(agent_rng, 3)
    t_indices = jax.random.randint(rng_t, (batch_size,), 0, num_steps - 1)
    env_indices = jax.random.randint(rng_env, (batch_size,), 0, num_envs)
    obs = agent_features[t_indices, env_indices, :]
    ranges = (num_steps - 1 - t_indices).astype(jnp.int32)
    ranges = jnp.maximum(ranges, 1)
    intervals = discounted_sampling(ranges, tdd_discount, rng_goal)
    goal_t = t_indices + intervals + 1
    goal_t = jnp.clip(goal_t, 0, num_steps - 1)
    goal = agent_features[goal_t, env_indices, :]

    c_g = potential_apply_fn(potential_params, goal)
    phi_s = s_encoder_apply_fn(s_encoder_params, obs)
    phi_g = s_encoder_apply_fn(s_encoder_params, goal)

    phi_s_expanded = phi_s[:, None, :]
    phi_g_expanded = phi_g[None, :, :]
    mrn_dists = mrn_distance(phi_s_expanded, phi_g_expanded)

    c_g_squeezed = c_g.squeeze(-1)
    logits = c_g_squeezed[None, :] - mrn_dists

    log_softmax_logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    loss1 = -(log_softmax_logits * I).sum(axis=-1).mean()

    log_softmax_logits_T = logits.T - jax.scipy.special.logsumexp(logits.T, axis=-1, keepdims=True)
    loss2 = -(log_softmax_logits_T * I).sum(axis=-1).mean()

    return (loss1 + loss2) / 2


def compute_tdd_loss_independent(
    potential_apply_fn, s_encoder_apply_fn,
    potential_params, s_encoder_params,
    agent_features, agent_rng,
    batch_size: int, num_steps: int, num_envs: int,
    tdd_discount: float,
):
    I = jnp.eye(batch_size)
    rng_t, rng_env, rng_goal = jax.random.split(agent_rng, 3)
    t_indices = jax.random.randint(rng_t, (batch_size,), 0, num_steps - 1)
    env_indices = jax.random.randint(rng_env, (batch_size,), 0, num_envs)
    obs = agent_features[t_indices, env_indices, :]
    ranges = (num_steps - 1 - t_indices).astype(jnp.int32)
    ranges = jnp.maximum(ranges, 1)
    intervals = discounted_sampling(ranges, tdd_discount, rng_goal)
    goal_t = t_indices + intervals + 1
    goal_t = jnp.clip(goal_t, 0, num_steps - 1)
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


def create_kstep_gae_metrics(
    kstep_stats: Dict,
    advantages: jnp.ndarray,
    z_i: jnp.ndarray,
    r_int: jnp.ndarray,
    z_prev: jnp.ndarray,
    v_int_all: jnp.ndarray,
    targets_v_int: jnp.ndarray,
    gamma_int: float,
    delta_ext_all: jnp.ndarray,
    delta_int_all: jnp.ndarray,
    num_agents: int,
) -> Dict:
    ext_rate = kstep_stats["ext_selection_rate"]
    int_rate = kstep_stats["int_selection_rate"]

    scale_ratio_abs = kstep_stats.get("scale_ratio_abs", jnp.array(0.0))
    scale_ratio_std = kstep_stats.get("scale_ratio_std", jnp.array(0.0))
    A_ext_abs_mean = kstep_stats.get("A_ext_abs_mean", jnp.array(0.0))
    S_int_abs_mean = kstep_stats.get("S_int_abs_mean", jnp.array(0.0))
    A_ext_std_val = kstep_stats.get("A_ext_std", jnp.array(0.0))
    S_int_std_val = kstep_stats.get("S_int_std", jnp.array(0.0))

    metrics = {
        "kstep_A_JT_mean": kstep_stats["A_JT_mean"],
        "kstep_A_JT_std": kstep_stats["A_JT_std"],
        "kstep_A_JT_min": kstep_stats["A_JT_min"],
        "kstep_A_JT_max": kstep_stats["A_JT_max"],
        "kstep_A_ext_mean": kstep_stats["A_ext_kstep_mean"],
        "kstep_S_int_mean": kstep_stats["S_int_kstep_mean"],
        "extrinsic_selection_pct": ext_rate * 100.0,
        "intrinsic_selection_pct": int_rate * 100.0,
        "scale_ratio_abs": scale_ratio_abs,
        "scale_ratio_std": scale_ratio_std,
        "A_ext_abs_mean": A_ext_abs_mean,
        "S_int_abs_mean": S_int_abs_mean,
        "A_ext_std": A_ext_std_val,
        "S_int_std": S_int_std_val,
        "delta_ext_mean": delta_ext_all.mean(),
        "delta_ext_std": delta_ext_all.std(),
        "delta_int_raw_mean": delta_int_all.mean(),
        "delta_int_raw_std": delta_int_all.std(),
        "z_i_mean": z_i.mean(),
        "z_i_std": z_i.std(),
        "r_int_mean": r_int.mean(),
        "r_int_std": r_int.std(),
        "z_prev_mean": z_prev.mean(),
        "z_prev_std": z_prev.std(),
        "z_diff_mean": (z_i - z_prev).mean(),
        "z_diff_std": (z_i - z_prev).std(),
        "budget_constraint_satisfied_pct": jnp.mean((z_i >= 0).astype(jnp.float32)) * 100.0,
        "budget_remaining_min": z_i.min(),
        "budget_remaining_max": z_i.max(),
        "v_int_mean": v_int_all.mean(),
        "v_int_std": v_int_all.std(),
        "v_int_targets_mean": targets_v_int.mean() if targets_v_int is not None else jnp.array(0.0),
    }
    return metrics


def create_fallback_metrics_kreturn(num_agents: int) -> Dict:
    return {
        "kstep_A_JT_mean": jnp.array(0.0),
        "kstep_A_JT_std": jnp.array(0.0),
        "kstep_A_JT_min": jnp.array(0.0),
        "kstep_A_JT_max": jnp.array(0.0),
        "kstep_A_ext_mean": jnp.array(0.0),
        "kstep_S_int_mean": jnp.array(0.0),
        "extrinsic_selection_pct": jnp.array(0.0),
        "intrinsic_selection_pct": jnp.array(0.0),
        "scale_ratio_abs": jnp.array(0.0),
        "scale_ratio_std": jnp.array(0.0),
        "A_ext_abs_mean": jnp.array(0.0),
        "S_int_abs_mean": jnp.array(0.0),
        "A_ext_std": jnp.array(0.0),
        "S_int_std": jnp.array(0.0),
        "delta_ext_mean": jnp.array(0.0),
        "delta_ext_std": jnp.array(0.0),
        "delta_int_raw_mean": jnp.array(0.0),
        "delta_int_raw_std": jnp.array(0.0),
        "z_i_mean": jnp.array(0.0),
        "z_i_std": jnp.array(0.0),
        "r_int_mean": jnp.array(0.0),
        "r_int_std": jnp.array(0.0),
        "z_prev_mean": jnp.array(0.0),
        "z_prev_std": jnp.array(0.0),
        "z_diff_mean": jnp.array(0.0),
        "z_diff_std": jnp.array(0.0),
        "budget_constraint_satisfied_pct": jnp.array(100.0),
        "budget_remaining_min": jnp.array(0.0),
        "budget_remaining_max": jnp.array(0.0),
        "v_int_mean": jnp.array(0.0),
        "v_int_std": jnp.array(0.0),
        "v_int_targets_mean": jnp.array(0.0),
    }


def compute_v_int_gae_targets(
    r_int: jnp.ndarray,
    v_int: jnp.ndarray,
    last_v_int: jnp.ndarray,
    gamma_int: float,
    gae_lambda: float,
    dones: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    v_int_next_shifted = jnp.concatenate([v_int[1:], last_v_int[None, :]], axis=0)
    delta_int_td = r_int + gamma_int * v_int_next_shifted * (1 - dones) - v_int

    def _gae_step(gae, inputs):
        delta, done = inputs
        gae_new = delta + gamma_int * gae_lambda * (1 - done) * gae
        return gae_new, gae_new

    _, advantages_int = jax.lax.scan(
        _gae_step, jnp.zeros_like(last_v_int), (delta_int_td, dones), reverse=True
    )

    targets_int = v_int + advantages_int
    return delta_int_td, targets_int


def create_z_init_metrics_kreturn(
    z_init_budget_new: jnp.ndarray,
    best_ext_reward_new: jnp.ndarray,
    z0_update_info: Dict = None,
) -> Dict:
    metrics = {
        "budget_mean": jnp.mean(z_init_budget_new),
        "best_ext_reward": best_ext_reward_new,
    }

    if z0_update_info is not None:
        metrics["passed_ratio"] = z0_update_info["passed_ratio"]
        metrics["update_delta_mean"] = z0_update_info["update_delta_mean"]
        if "z_decayed_on_failure" in z0_update_info:
            metrics["z_decayed_on_failure"] = z0_update_info["z_decayed_on_failure"]

    return metrics


def print_training_header_kreturn(config: Dict) -> None:
    print("=" * 80)
    print("EFXPLORER K-RETURN: K-step Joint Advantage")
    print("=" * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Training Steps: {config['TOTAL_TIMESTEPS']:.0f}")
    print(f"K-step lambda: {config.get('JOINT_GAE_LAMBDA', config['GAE_LAMBDA'])}")
    print(f"K-step Adv Clip: {config.get('KSTEP_ADV_CLIP_ABS', 20.0)}")
    print("-" * 40)
    print(f"TDD Mode: {'Shared' if config.get('TDD_SHARED_PARAMS', True) else 'Independent'}")
    print(f"z_init Mode: {'Episode-level' if config.get('USE_EPISODE_LEVEL_Z0', True) else 'Adaptive EMA'}")
    if config.get('USE_EPISODE_LEVEL_Z0', True):
        print(f"  - Threshold Decay: {config.get('Z0_BEST_REWARD_DECAY', 0.99)}")
    if config.get('USE_ADAPTIVE_Z_INIT', True):
        print(f"  - EMA alpha_z: {config.get('ALPHA_Z', 0.1)}")
    else:
        print(f"z Initial Budget: {config.get('Z_INIT_BUDGET', 10.0)} (fixed)")
    print(f"z Reset on Done: {config.get('Z_BUDGET_RESET_ON_DONE', True)}")
    print(f"gamma (unified): {config.get('GAMMA', 0.99)}")
    print("=" * 80)
    print()


def save_training_checkpoints(run, config: Dict, out: Dict, save_params_fn) -> None:
    import wandb
    import jax

    if config.get('WANDB_MODE') == "disabled" or not hasattr(run, 'dir'):
        return

    save_dir = os.path.join(run.dir, 'params')
    os.makedirs(save_dir, exist_ok=True)

    train_states = out['runner_state'][0][0]

    actor_train_states = train_states[0]
    if isinstance(actor_train_states, (tuple, list)) and len(actor_train_states) > 1:
        all_actor_params = [s.params for s in actor_train_states]
        stacked_actor_params = jax.tree.map(lambda *xs: jnp.stack(xs), *all_actor_params)
        save_params_fn(stacked_actor_params, os.path.join(save_dir, 'actor_params.safetensors'))
        print(f'  Saved {len(actor_train_states)} independent actor params (stacked)')
    else:
        actor_params = actor_train_states[0].params if isinstance(actor_train_states, (tuple, list)) else actor_train_states.params
        save_params_fn(actor_params, os.path.join(save_dir, 'actor_params.safetensors'))
        print(f'  Saved shared actor params')

    critic_params = train_states[1].params
    save_params_fn(critic_params, os.path.join(save_dir, 'critic_params.safetensors'))
    print(f'  Saved critic params')

    tdd_shared = config.get('TDD_SHARED_PARAMS', True)
    tdd_potential_states = train_states[2]
    tdd_s_encoder_states = train_states[3]

    if tdd_shared:
        potential_params = tdd_potential_states[0].params
        s_encoder_params = tdd_s_encoder_states[0].params
        save_params_fn(potential_params, os.path.join(save_dir, 'tdd_potential_params.safetensors'))
        save_params_fn(s_encoder_params, os.path.join(save_dir, 'tdd_s_encoder_params.safetensors'))
        print(f'  Saved shared TDD params (potential + s_encoder)')
    else:
        all_potential_params = [s.params for s in tdd_potential_states]
        stacked_potential_params = jax.tree.map(lambda *xs: jnp.stack(xs), *all_potential_params)
        save_params_fn(stacked_potential_params, os.path.join(save_dir, 'tdd_potential_params.safetensors'))

        all_s_encoder_params = [s.params for s in tdd_s_encoder_states]
        stacked_s_encoder_params = jax.tree.map(lambda *xs: jnp.stack(xs), *all_s_encoder_params)
        save_params_fn(stacked_s_encoder_params, os.path.join(save_dir, 'tdd_s_encoder_params.safetensors'))
        print(f'  Saved {len(tdd_potential_states)} independent TDD params (stacked)')

    has_v_int = len(train_states) > 4
    if has_v_int:
        v_int_train_state = train_states[4]
        v_int_params = v_int_train_state.params
        save_params_fn(v_int_params, os.path.join(save_dir, 'v_int_params.safetensors'))
        print(f'  Saved V_int params')
    else:
        print(f'  Skipped V_int params (not present in this algorithm)')

    import numpy as np
    metadata = {
        'tdd_shared': tdd_shared,
        'num_agents': len(actor_train_states) if isinstance(actor_train_states, (tuple, list)) else 1,
        'has_v_int': has_v_int,
    }
    metadata_path = os.path.join(save_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f'Checkpoints saved to {save_dir}')

    if hasattr(run, 'log_artifact'):
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(os.path.join(save_dir, 'actor_params.safetensors'))
        artifact.add_file(os.path.join(save_dir, 'critic_params.safetensors'))
        artifact.add_file(os.path.join(save_dir, 'tdd_potential_params.safetensors'))
        artifact.add_file(os.path.join(save_dir, 'tdd_s_encoder_params.safetensors'))
        if has_v_int:
            artifact.add_file(os.path.join(save_dir, 'v_int_params.safetensors'))
        artifact.add_file(metadata_path)
        artifact.add_file(config_path)

        history_csv = os.path.join(run.dir, 'training_history.csv')
        history_json = os.path.join(run.dir, 'training_history.json')
        if os.path.exists(history_csv):
            artifact.add_file(history_csv)
        if os.path.exists(history_json):
            artifact.add_file(history_json)
        run.log_artifact(artifact)
