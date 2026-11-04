from typing import Set
import numpy as np


def compute_entropy_per_agent(action_logits, num_agents, num_envs):
    action_probs = _softmax(action_logits, axis=-1)
    log_probs = np.log(action_probs + 1e-8)
    entropy = -np.sum(action_probs * log_probs, axis=-1)
    num_steps = entropy.shape[0]
    entropy_reshaped = entropy.reshape(num_steps, num_agents, num_envs)
    entropy_per_agent = np.mean(entropy_reshaped, axis=(0, 2))
    return entropy_per_agent


def _softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_sd_debug_metrics(intrinsic_reward_scaled, intrinsic_reward_scale, intrinsic_reward_raw=None):
    post_robust = intrinsic_reward_scaled / (intrinsic_reward_scale + 1e-8)

    result = {
        "sd/post_robust_min": float(np.min(post_robust)),
        "sd/post_robust_max": float(np.max(post_robust)),
        "sd/post_robust_mean": float(np.mean(post_robust)),
        "sd/post_robust_std": float(np.std(post_robust)),
        "sd/final_scaled_min": float(np.min(intrinsic_reward_scaled)),
        "sd/final_scaled_max": float(np.max(intrinsic_reward_scaled)),
        "sd/final_scaled_mean": float(np.mean(intrinsic_reward_scaled)),
    }

    if intrinsic_reward_raw is not None:
        result.update({
            "sd/true_raw_min": float(np.min(intrinsic_reward_raw)),
            "sd/true_raw_max": float(np.max(intrinsic_reward_raw)),
            "sd/true_raw_mean": float(np.mean(intrinsic_reward_raw)),
            "sd/true_raw_std": float(np.std(intrinsic_reward_raw)),
        })

    return result


def format_progress_string(
    update_step,
    num_updates,
    env_step,
    returns,
    extrinsic_reward,
    intrinsic_reward,
    tau,
    total_loss,
    actor_loss,
    value_loss,
    entropy,
):
    tau_str = f"Tau: {tau:.4f} | " if tau is not None else ""
    return (
        f"Update {update_step}/{num_updates} | "
        f"Env Step: {env_step:8d} | "
        f"Returns: {returns:8.2f} | "
        f"Ext: {extrinsic_reward:7.4f} | "
        f"Int: {intrinsic_reward:7.4f} | "
        f"{tau_str}"
        f"Loss: {total_loss:.4f} | "
        f"Actor: {actor_loss:.4f} | "
        f"Value: {value_loss:.4f} | "
        f"Ent: {entropy:.4f}"
    )


def compute_video_steps(num_updates: int, warmup_rollouts: int, num_intermediate: int = 6) -> Set[int]:
    video_steps = set()
    video_steps.add(num_updates - 1)

    if num_intermediate > 0 and num_updates > 1:
        start_step = 0
        end_step = num_updates - 1
        training_range = end_step - start_step
        if training_range > 0:
            interval = training_range / (num_intermediate + 1)
            for i in range(1, num_intermediate + 1):
                step = int(start_step + interval * i)
                if step > 1:
                    video_steps.add(step)

    return video_steps
