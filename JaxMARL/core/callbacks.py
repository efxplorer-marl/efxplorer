import os
import numpy as np
import jax
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, Set, Optional

from .metrics import (
    compute_entropy_per_agent,
    compute_sd_debug_metrics,
    format_progress_string,
)
from .utils import ParamsHolder, save_params, stack_params


class CallbackHandler:

    def __init__(
        self,
        config: Dict[str, Any],
        env,
        actor_network,
        critic_network,
        shared_actor_params: bool,
        num_agents: int,
        video_steps: Set[int],
        tdd_shared: bool = False,
        is_centralized_critic: bool = True,
    ):
        self.config = config
        self.env = env
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.shared_actor_params = shared_actor_params
        self.num_agents = num_agents
        self.video_steps = video_steps
        self.tdd_shared = tdd_shared
        self.is_centralized_critic = is_centralized_critic
        self.env_info = self._extract_env_info(env)

    def _extract_env_info(self, env) -> Dict[str, Any]:
        env_info = {
            "map_x_min": -5.0,
            "map_x_max": 5.0,
            "map_y_min": -3.0,
            "map_y_max": 3.0,
            "num_walls": 0,
            "walls": [],
        }

        if hasattr(env, 'map_size_horizontal'):
            env_info["map_x_min"] = -float(env.map_size_horizontal)
            env_info["map_x_max"] = float(env.map_size_horizontal)
        if hasattr(env, 'map_size_vertical'):
            env_info["map_y_min"] = -float(env.map_size_vertical)
            env_info["map_y_max"] = float(env.map_size_vertical)

        num_walls = getattr(env, 'num_walls', 0)
        env_info["num_walls"] = num_walls

        if num_walls > 0:
            wall_positions = getattr(env, 'wall_positions_jnp', None)
            wall_widths = getattr(env, 'wall_widths_jnp', None)
            wall_heights = getattr(env, 'wall_heights_jnp', None)

            if wall_positions is not None and wall_widths is not None and wall_heights is not None:
                walls = []
                positions_np = np.array(wall_positions)
                widths_np = np.array(wall_widths)
                heights_np = np.array(wall_heights)

                for i in range(num_walls):
                    walls.append({
                        "center": (float(positions_np[i, 0]), float(positions_np[i, 1])),
                        "width": float(widths_np[i]),
                        "height": float(heights_np[i]),
                    })
                env_info["walls"] = walls
            else:
                corridor_width = getattr(env, 'corridor_width', 1.0)
                half_corridor = corridor_width / 2
                map_x_max = env_info["map_x_max"]

                wall_width = map_x_max - half_corridor
                left_wall_center_x = -(map_x_max + half_corridor) / 2
                right_wall_center_x = (map_x_max + half_corridor) / 2

                env_info["walls"] = [
                    {"center": (left_wall_center_x, 0.0), "width": wall_width, "height": 2.0},
                    {"center": (right_wall_center_x, 0.0), "width": wall_width, "height": 2.0},
                ]
                env_info["num_walls"] = 2

        agent_size = getattr(env, 'agent_size', 0.15)
        env_info["agent_size"] = float(agent_size)

        landmark_size = getattr(env, 'landmark_size', 0.05)
        env_info["landmark_size"] = float(landmark_size)

        corridor_width = getattr(env, 'corridor_width', 1.0)
        env_info["corridor_width"] = float(corridor_width)

        return env_info

    def _save_checkpoint_at_video_step(self, metric: Dict, update_steps: int) -> Optional[str]:
        import yaml

        try:
            checkpoint_dir = os.path.join(wandb.run.dir, 'checkpoints', f'update_{update_steps}')
            os.makedirs(checkpoint_dir, exist_ok=True)

            actor_params = metric.get("actor_params")
            if actor_params is not None:
                if isinstance(actor_params, list):
                    stacked_actor = stack_params(actor_params)
                    save_params(stacked_actor, os.path.join(checkpoint_dir, 'actor_params.safetensors'))
                else:
                    save_params(actor_params, os.path.join(checkpoint_dir, 'actor_params.safetensors'))

            critic_params = metric.get("critic_params")
            if critic_params is not None:
                save_params(critic_params, os.path.join(checkpoint_dir, 'critic_params.safetensors'))

            tdd_potential_params = metric.get("tdd_potential_params")
            if tdd_potential_params is not None:
                if isinstance(tdd_potential_params, (list, tuple)):
                    stacked = stack_params(list(tdd_potential_params))
                    save_params(stacked, os.path.join(checkpoint_dir, 'tdd_potential_params.safetensors'))
                else:
                    save_params(tdd_potential_params, os.path.join(checkpoint_dir, 'tdd_potential_params.safetensors'))

            tdd_s_encoder_params = metric.get("tdd_s_encoder_params")
            if tdd_s_encoder_params is not None:
                if isinstance(tdd_s_encoder_params, (list, tuple)):
                    stacked = stack_params(list(tdd_s_encoder_params))
                    save_params(stacked, os.path.join(checkpoint_dir, 'tdd_s_encoder_params.safetensors'))
                else:
                    save_params(tdd_s_encoder_params, os.path.join(checkpoint_dir, 'tdd_s_encoder_params.safetensors'))

            v_int_params = metric.get("v_int_params")
            if v_int_params is not None:
                save_params(v_int_params, os.path.join(checkpoint_dir, 'v_int_params.safetensors'))

            z_init_info = metric.get("z_init", {})
            z_init_budget = z_init_info.get("budget_per_agent")
            if z_init_budget is not None:
                np.save(os.path.join(checkpoint_dir, 'z_init_budget.npy'), np.array(z_init_budget))

            config_path = os.path.join(checkpoint_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

            metadata = {
                'update_step': update_steps,
                'num_agents': self.num_agents,
                'tdd_shared': self.tdd_shared,
            }
            metadata_path = os.path.join(checkpoint_dir, 'metadata.yaml')
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)

            print(f"  [Checkpoint] Saved to {checkpoint_dir}")
            return checkpoint_dir

        except Exception as e:
            print(f"  [Checkpoint] Failed to save: {e}")
            import traceback
            traceback.print_exc()
            return None

    def __call__(self, metric: Dict[str, Any]) -> None:
        update_steps = int(np.array(metric["update_steps"]))
        env_step = update_steps * self.config["NUM_ENVS"] * self.config["NUM_STEPS"]
        should_save_checkpoint = update_steps in self.video_steps

        log_dict = self._compute_light_metrics(metric, env_step)

        if should_save_checkpoint:
            try:
                heavy_metrics = self._compute_heavy_metrics(metric, update_steps)
                log_dict.update(heavy_metrics)
            except Exception as e:
                print(f"Warning: Heavy metrics computation failed: {e}")

            self._save_checkpoint_at_video_step(metric, update_steps)

        wandb.log(log_dict, step=update_steps)

        self._print_progress(metric, update_steps, env_step, should_save_checkpoint)

    def _compute_light_metrics(self, metric: Dict, env_step: int) -> Dict:
        returns = float(np.array(metric["returned_episode_returns"][-1, :].mean()))
        loss_info = metric["loss"]
        reward_info = metric.get("rewards", {})
        kstep_info = metric.get("kstep", {})

        extrinsic_reward = float(np.array(reward_info.get("extrinsic", 0.0)))

        log_dict = {
            "returns": returns,
            "env_step": env_step,
            "reward/extrinsic": extrinsic_reward,
            "reward/intrinsic_scaled": float(np.array(reward_info.get("intrinsic_scaled", 0.0))),
            "reward/combined": float(np.array(reward_info.get("combined", extrinsic_reward))),
            "loss/tdd": float(np.array(metric.get("tdd_loss", 0.0))),
            **{f"loss/{k}": float(np.array(v)) for k, v in loss_info.items()},
        }

        if "extrinsic_min" in reward_info:
            log_dict["reward/extrinsic_min"] = float(np.array(reward_info["extrinsic_min"]))
            log_dict["reward/extrinsic_max"] = float(np.array(reward_info["extrinsic_max"]))
            log_dict["reward/extrinsic_std"] = float(np.array(reward_info["extrinsic_std"]))
            log_dict["reward/extrinsic_range"] = float(np.array(reward_info["extrinsic_range"]))
        if "intrinsic_scaled_min" in reward_info:
            log_dict["reward/intrinsic_scaled_min"] = float(np.array(reward_info["intrinsic_scaled_min"]))
            log_dict["reward/intrinsic_scaled_max"] = float(np.array(reward_info["intrinsic_scaled_max"]))
            log_dict["reward/intrinsic_scaled_std"] = float(np.array(reward_info["intrinsic_scaled_std"]))
            log_dict["reward/intrinsic_scaled_range"] = float(np.array(reward_info["intrinsic_scaled_range"]))
        if "intrinsic_raw" in reward_info:
            log_dict["reward/intrinsic_raw"] = float(np.array(reward_info["intrinsic_raw"]))
            log_dict["reward/intrinsic_raw_min"] = float(np.array(reward_info["intrinsic_raw_min"]))
            log_dict["reward/intrinsic_raw_max"] = float(np.array(reward_info["intrinsic_raw_max"]))
            log_dict["reward/intrinsic_raw_std"] = float(np.array(reward_info["intrinsic_raw_std"]))
            log_dict["reward/intrinsic_raw_range"] = float(np.array(reward_info["intrinsic_raw_range"]))
        if "int_ext_ratio_scaled" in reward_info:
            log_dict["reward/int_ext_ratio_scaled"] = float(np.array(reward_info["int_ext_ratio_scaled"]))
            log_dict["reward/int_ext_ratio_raw"] = float(np.array(reward_info["int_ext_ratio_raw"]))
            log_dict["reward/range_ratio_scaled"] = float(np.array(reward_info["range_ratio_scaled"]))

        if "intrinsic_reward_scale" in metric:
            log_dict["annealing/intrinsic_scale"] = float(np.array(metric["intrinsic_reward_scale"]))
        if "ent_coef" in metric:
            log_dict["annealing/ent_coef"] = float(np.array(metric["ent_coef"]))

        linear_info = metric.get("linear", {})
        if linear_info:
            log_dict["linear/alpha"] = float(np.array(linear_info.get("alpha", 0.0)))
            log_dict["linear/int_contribution_ratio"] = float(np.array(linear_info.get("int_contribution_ratio", 0.0)))

        lagrangian_info = metric.get("lagrangian", {})
        if lagrangian_info:
            log_dict["lagrangian/lambda"] = float(np.array(lagrangian_info.get("lambda", 0.0)))
            log_dict["lagrangian/int_contribution_ratio"] = float(np.array(lagrangian_info.get("int_contribution_ratio", 0.0)))

        for key, value in kstep_info.items():
            if key == "agent_intrinsic_pct":
                agent_pcts = np.array(value)
                for agent_idx in range(len(agent_pcts)):
                    log_dict[f"kstep/intrinsic_pct_agent_{agent_idx}"] = float(agent_pcts[agent_idx])
            else:
                log_dict[f"kstep/{key}"] = float(np.array(value))

        z_init_info = metric.get("z_init", {})
        if z_init_info:
            log_dict["z_init/budget_mean"] = float(np.array(z_init_info.get("budget_mean", 0.0)))
            log_dict["z_init/best_ext_reward"] = float(np.array(z_init_info.get("best_ext_reward", 0.0)))
            if "passed_ratio" in z_init_info:
                log_dict["z_init/passed_ratio"] = float(np.array(z_init_info["passed_ratio"]))
            if "update_delta_mean" in z_init_info:
                log_dict["z_init/update_delta_mean"] = float(np.array(z_init_info["update_delta_mean"]))

        return log_dict

    def _compute_heavy_metrics(self, metric: Dict, update_steps: int) -> Dict:
        log_dict = {}

        num_agents_cb = int(metric["num_agents"])
        num_envs_cb = int(metric["num_envs"])
        intrinsic_scale = float(np.array(metric["intrinsic_reward_scale"]))

        action_logits = np.array(metric["action_logits"])
        entropy_per_agent = compute_entropy_per_agent(action_logits, num_agents_cb, num_envs_cb)
        for i, ent in enumerate(entropy_per_agent):
            log_dict[f"entropy/agent_{i}"] = float(ent)
        log_dict["entropy/std_across_agents"] = float(np.std(entropy_per_agent))
        log_dict["entropy/mean"] = float(np.mean(entropy_per_agent))

        intrinsic_reward_scaled_arr = np.array(metric["intrinsic_reward_scaled"])
        intrinsic_reward_raw_arr = np.array(metric["intrinsic_reward_raw"]) if "intrinsic_reward_raw" in metric else None
        sd_debug = compute_sd_debug_metrics(intrinsic_reward_scaled_arr, intrinsic_scale, intrinsic_reward_raw_arr)
        log_dict.update(sd_debug)

        return log_dict

    def _print_progress(self, metric: Dict, update_steps: int, env_step: int, should_save_checkpoint: bool) -> None:
        should_print = (
            (update_steps % 10 == 0) or
            should_save_checkpoint or
            (update_steps == self.config["NUM_UPDATES"])
        )

        if should_print:
            loss_info = metric["loss"]
            reward_info = metric.get("rewards", {})
            kstep_info = metric.get("kstep", {})

            returns = float(np.array(metric["returned_episode_returns"][-1, :].mean()))
            extrinsic_reward = float(np.array(reward_info.get("extrinsic", 0.0)))
            intrinsic_reward_scaled = float(np.array(reward_info.get("intrinsic_scaled", 0.0)))
            tau_raw = metric.get("tau", kstep_info.get("tau"))
            tau = float(np.array(tau_raw)) if tau_raw is not None else None

            actor_loss = float(np.array(loss_info.get("actor_loss", 0.0)))
            value_loss = float(np.array(loss_info.get("value_loss", 0.0)))
            entropy = float(np.array(loss_info.get("entropy", 0.0)))
            total_loss = float(np.array(loss_info.get("total_loss", 0.0)))

            progress_str = format_progress_string(
                update_steps, self.config["NUM_UPDATES"], env_step,
                returns, extrinsic_reward, intrinsic_reward_scaled,
                tau, total_loss, actor_loss, value_loss, entropy
            )
            print(progress_str)
