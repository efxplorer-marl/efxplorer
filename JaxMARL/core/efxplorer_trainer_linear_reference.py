from typing import NamedTuple, Dict
import os
import yaml
import jax.numpy as jnp


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
    vf_coef: float
    max_grad_norm: float
    num_updates: int
    update_epochs: int
    num_minibatches: int
    warmup_rollouts: int
    seed: int

    intrinsic_reward_alpha: float

    tdd_shared: bool
    tdd_latent_dim: int
    tdd_output_dim: int
    tdd_lr: float
    tdd_batch_size: int
    tdd_update_epochs: int
    tdd_discount: float

    use_robust_sd: bool
    sd_threshold_mode: str
    sd_threshold: float
    sd_percentile: float
    sd_scale: float
    use_exponential_sd: bool
    sd_clip_max: float
    intrinsic_reward_scale: float
    sd_adaptive_percentile: float
    sd_threshold_min: float
    sd_threshold_max: float

    use_inter_agent_sd: bool
    include_self_in_inter_sd: bool
    inter_sd_weight: float


def create_trainer_config_linear(config: Dict, num_agents: int) -> TrainerConfigLinear:
    return TrainerConfigLinear(
        num_envs=config["NUM_ENVS"],
        num_steps=config["NUM_STEPS"],
        num_agents=num_agents,
        num_actors=config["NUM_ACTORS"],
        gamma=config["GAMMA"],
        gae_lambda=config["GAE_LAMBDA"],
        clip_eps=config["CLIP_EPS"],
        ent_coef=config["ENT_COEF"],
        anneal_ent=config.get("ANNEAL_ENT", False),
        ent_coef_end=config.get("ENT_COEF_END", config["ENT_COEF"]),
        vf_coef=config["VF_COEF"],
        max_grad_norm=config["MAX_GRAD_NORM"],
        num_updates=config["NUM_UPDATES"],
        update_epochs=config["UPDATE_EPOCHS"],
        num_minibatches=config["NUM_MINIBATCHES"],
        warmup_rollouts=config.get("WARMUP_ROLLOUTS", 0),
        seed=config["SEED"],

        intrinsic_reward_alpha=config.get("INTRINSIC_REWARD_ALPHA", 1.0),

        tdd_shared=config.get("TDD_SHARED_PARAMS", True),
        tdd_latent_dim=config.get("TDD_LATENT_DIM", 64),
        tdd_output_dim=config.get("TDD_OUTPUT_DIM", 32),
        tdd_lr=config.get("TDD_LR", 1e-3),
        tdd_batch_size=config.get("TDD_BATCH_SIZE", 512),
        tdd_update_epochs=config.get("TDD_UPDATE_EPOCHS", 144),
        tdd_discount=config.get("TDD_DISCOUNT", 0.967),

        use_robust_sd=config.get("USE_ROBUST_SD", True),
        sd_threshold_mode=config.get("SD_THRESHOLD_MODE", "fixed"),
        sd_threshold=config.get("SD_THRESHOLD", 0.05),
        sd_percentile=config.get("SD_PERCENTILE", 30.0),
        sd_scale=config.get("SD_SCALE", 5.0),
        use_exponential_sd=config.get("USE_EXPONENTIAL_SD", True),
        sd_clip_max=config.get("SD_CLIP_MAX", 10.0),
        intrinsic_reward_scale=config.get("INTRINSIC_REWARD_SCALE", 1.0),
        sd_adaptive_percentile=config.get("SD_ADAPTIVE_PERCENTILE", 0.3),
        sd_threshold_min=config.get("SD_THRESHOLD_MIN", 0.1),
        sd_threshold_max=config.get("SD_THRESHOLD_MAX", 10.0),

        use_inter_agent_sd=config.get("USE_INTER_AGENT_SD", False),
        include_self_in_inter_sd=config.get("INCLUDE_SELF_IN_INTER_SD", False),
        inter_sd_weight=config.get("INTER_SD_WEIGHT", 0.1),
    )


def create_linear_metrics(
    r_ext: jnp.ndarray,
    r_int: jnp.ndarray,
    r_combined: jnp.ndarray,
    alpha: float,
) -> Dict:
    return {
        "alpha": alpha,
        "r_ext_mean": r_ext.mean(),
        "r_ext_std": r_ext.std(),
        "r_int_mean": r_int.mean(),
        "r_int_std": r_int.std(),
        "r_combined_mean": r_combined.mean(),
        "r_combined_std": r_combined.std(),
        "ext_contribution": r_ext.mean(),
        "int_contribution": (alpha * r_int).mean(),
        "int_contribution_ratio": jnp.abs(alpha * r_int).mean() / (
            jnp.abs(r_ext).mean() + jnp.abs(alpha * r_int).mean() + 1e-8
        ),
    }


def create_robust_sd_metrics_linear(
    use_robust_sd: bool,
    sd_threshold: float,
    filtered_ratio: jnp.ndarray,
    reward_active_ratio: jnp.ndarray,
    sd_threshold_mode: str = "fixed",
    sd_stats_mean: jnp.ndarray = None,
    sd_stats_std: jnp.ndarray = None,
    sd_stats_count: jnp.ndarray = None,
) -> Dict:
    mode_idx = {"fixed": 0, "adaptive": 1, "percentile": 2}.get(sd_threshold_mode, 0)
    metrics = {
        "enabled": use_robust_sd,
        "mode_idx": jnp.array(mode_idx),
        "threshold": sd_threshold,
        "filtered_ratio": filtered_ratio,
        "reward_active_ratio": reward_active_ratio,
    }
    if sd_threshold_mode == "adaptive":
        if sd_stats_mean is not None:
            metrics["stats_mean"] = sd_stats_mean
        if sd_stats_std is not None:
            metrics["stats_std"] = sd_stats_std
        if sd_stats_count is not None:
            metrics["stats_count"] = sd_stats_count
    return metrics


def create_inter_agent_sd_metrics_linear(
    inter_agent_sd_metrics: Dict,
    use_inter_agent_sd: bool,
    include_self_in_inter_sd: bool,
    inter_sd_weight: float,
) -> Dict:
    if inter_agent_sd_metrics is not None:
        intra_raw = inter_agent_sd_metrics["intra_raw"]
        inter_raw = inter_agent_sd_metrics["inter_raw"]
        inter_weighted = inter_agent_sd_metrics["inter_weighted"]
        combined = intra_raw + inter_weighted

        return {
            "enabled": use_inter_agent_sd,
            "include_self": include_self_in_inter_sd,
            "weight": inter_agent_sd_metrics["weight"],
            "intra_raw_mean": jnp.mean(intra_raw),
            "inter_raw_mean": jnp.mean(inter_raw),
            "inter_weighted_mean": jnp.mean(inter_weighted),
            "combined_mean": jnp.mean(combined),
        }
    else:
        return {
            "enabled": use_inter_agent_sd,
            "include_self": include_self_in_inter_sd,
            "weight": inter_sd_weight,
        }


def print_training_header_linear(config: Dict) -> None:
    print("=" * 80)
    print("EFXPLORER LINEAR: Simple r = r_ext + alpha * r_int")
    print("=" * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Training Steps: {config['TOTAL_TIMESTEPS']:.0f}")
    print("-" * 40)
    print("Linear Reward Combination:")
    print(f"  alpha: {config.get('INTRINSIC_REWARD_ALPHA', 1.0)}")
    print(f"  Intrinsic Reward Scale: {config.get('INTRINSIC_REWARD_SCALE', 1.0)}")
    print("-" * 40)
    print(f"TDD Mode: {'Shared' if config.get('TDD_SHARED_PARAMS', True) else 'Independent'}")
    print(f"Robust SD: {config.get('USE_ROBUST_SD', True)}")
    if config.get('USE_ROBUST_SD', True):
        print(f"  - Mode: {config.get('SD_THRESHOLD_MODE', 'percentile')}")
        if config.get('SD_THRESHOLD_MODE', 'percentile') == 'percentile':
            print(f"  - Percentile: {config.get('SD_PERCENTILE', 30.0)}")
    print(f"gamma: {config.get('GAMMA', 0.99)}")
    if config.get('ANNEAL_ENT', False):
        print(f"Entropy Annealing: {config['ENT_COEF']} -> {config.get('ENT_COEF_END', config['ENT_COEF'])}")
    else:
        print(f"Entropy Coefficient: {config['ENT_COEF']} (fixed)")
    print("=" * 80)
    print()


def save_training_checkpoints_linear(run, config: Dict, out: Dict, save_params_fn) -> None:
    import wandb

    if config.get('WANDB_MODE') == "disabled" or not hasattr(run, 'dir'):
        return

    save_dir = os.path.join(run.dir, 'params')
    os.makedirs(save_dir, exist_ok=True)

    train_states = out['runner_state'][0][0]
    actor_params = train_states[0][0].params
    save_params_fn(actor_params, os.path.join(save_dir, 'actor_params.safetensors'))

    critic_params = train_states[1].params
    save_params_fn(critic_params, os.path.join(save_dir, 'critic_params.safetensors'))

    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f'Checkpoints saved to {save_dir}')

    if hasattr(run, 'log_artifact'):
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(os.path.join(save_dir, 'actor_params.safetensors'))
        artifact.add_file(os.path.join(save_dir, 'critic_params.safetensors'))
        artifact.add_file(config_path)

        history_csv = os.path.join(run.dir, 'training_history.csv')
        history_json = os.path.join(run.dir, 'training_history.json')
        if os.path.exists(history_csv):
            artifact.add_file(history_csv)
        if os.path.exists(history_json):
            artifact.add_file(history_json)
        run.log_artifact(artifact)
