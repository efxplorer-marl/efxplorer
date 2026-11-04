from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchrl.objectives.value.advantages import (
    GAE as TorchRLGAE,
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
)
from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    vec_generalized_advantage_estimate,
    _geom_series_like,
    _get_num_per_traj,
    _split_and_pad_sequence,
)
from torchrl.objectives.value.utils import (
    _custom_conv1d,
)
from torchrl.objectives.utils import (
    default_value_kwargs as torchrl_default_value_kwargs,
    hold_out_net,
)
from torchrl.objectives import ValueEstimators
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, dispatch
from tensordict.utils import NestedKey
from enum import Enum

try:
    from torch._dynamo import is_dynamo_compiling
except ImportError:
    def is_dynamo_compiling():
        return False


class EpigraphValueEstimators(Enum):
    EPIGRAPH_GAE = "EpigraphGAE"
    LAGRANGIAN_GAE = "LagrangianGAE"


def default_value_kwargs(value_type: ValueEstimators | EpigraphValueEstimators | str):
    if isinstance(value_type, EpigraphValueEstimators):
        if value_type == EpigraphValueEstimators.EPIGRAPH_GAE:
            return {"gamma": 0.99, "lmbda": 0.95, "differentiable": True}
        elif value_type == EpigraphValueEstimators.LAGRANGIAN_GAE:
            return {"gamma": 0.99, "lmbda": 0.95, "differentiable": True}
        else:
            raise NotImplementedError(f"Unknown EpigraphValueEstimators value type {value_type}.")
    
    if value_type == "EpigraphGAE" or value_type == EpigraphValueEstimators.EPIGRAPH_GAE.value:
        return {"gamma": 0.99, "lmbda": 0.95, "differentiable": True}
    
    if value_type == "LagrangianGAE" or value_type == EpigraphValueEstimators.LAGRANGIAN_GAE.value:
        return {"gamma": 0.99, "lmbda": 0.95, "differentiable": True}
    
    if isinstance(value_type, ValueEstimators):
        return torchrl_default_value_kwargs(value_type)
    
    try:
        for ve in ValueEstimators:
            if value_type == ve.value or value_type == ve.name:
                return torchrl_default_value_kwargs(ve)
    except:
        pass
    
    raise NotImplementedError(f"Unknown value type {value_type}.")


def _vectorized_gae_from_deltas(
    deltas: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    lmbda: float,
    time_dim: int = 0,
    thr: float = 1e-7,
) -> torch.Tensor:
    assert time_dim == 0, "time_dim must be 0, caller should transpose"
    
    T = deltas.shape[0]
    device = deltas.device
    dtype = deltas.dtype
    original_shape = deltas.shape
    
    deltas_flat = deltas.reshape(T, -1)
    done_flat = done.reshape(T, -1)
    batch_size = deltas_flat.shape[1]
    
    gammalmbda = gamma * lmbda
    
    deltas_bt = deltas_flat.transpose(0, 1).contiguous()
    done_bt = done_flat.transpose(0, 1).contiguous()
    
    try:
        num_per_traj = _get_num_per_traj(done_bt)
        
        deltas_split, mask = _split_and_pad_sequence(deltas_bt, num_per_traj, return_mask=True)
        
        gammalmbdas = _geom_series_like(deltas_split[0], gammalmbda, thr=thr)
        
        advantages_split = _custom_conv1d(deltas_split.unsqueeze(1), gammalmbdas)
        advantages_split = advantages_split.squeeze(1)
        
        advantages_bt = advantages_split[mask].view_as(deltas_bt)
        
    except Exception:
        not_done = (~done_bt.bool()).to(dtype)
        discount_factors = gammalmbda * not_done
        
        advantages_bt = torch.zeros_like(deltas_bt)
        gae_next = torch.zeros(batch_size, device=device, dtype=dtype)
        
        for t in range(T - 1, -1, -1):
            advantages_bt[:, t] = deltas_bt[:, t] + discount_factors[:, t] * gae_next
            gae_next = advantages_bt[:, t]
    
    advantages_flat = advantages_bt.transpose(0, 1).contiguous()
    
    return advantages_flat.reshape(original_shape)


def _compute_kstep_returns(
    delta_ext_split: torch.Tensor,
    delta_int_split: torch.Tensor,
    value_split: torch.Tensor,
    V_int_split: torch.Tensor,
    gamma: float,
    max_T: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_trajs = delta_ext_split.shape[0]
    
    gamma_powers = torch.pow(gamma, torch.arange(max_T, device=device, dtype=dtype))
    
    V_ext_kstep = torch.zeros(num_trajs, max_T, max_T, device=device, dtype=dtype)
    V_int_kstep = torch.zeros(num_trajs, max_T, max_T, device=device, dtype=dtype)
    
    for t in range(max_T):
        remaining = max_T - t
        if remaining <= 0:
            continue
        
        deltas_ext_from_t = delta_ext_split[:, t:]
        weighted_ext = deltas_ext_from_t * gamma_powers[:remaining].unsqueeze(0)
        cumsum_ext = torch.cumsum(weighted_ext, dim=1)
        V_ext_kstep[:, t, :remaining] = value_split[:, t:t+1] + cumsum_ext
        
        deltas_int_from_t = delta_int_split[:, t:]
        weighted_int = deltas_int_from_t * gamma_powers[:remaining].unsqueeze(0)
        cumsum_int = torch.cumsum(weighted_int, dim=1)
        V_int_kstep[:, t, :remaining] = V_int_split[:, t:t+1] + cumsum_int
    
    return V_ext_kstep, V_int_kstep


def _lambda_weighted_value(
    V_kstep: torch.Tensor,
    lmbda: float,
    max_T: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_trajs, max_T_actual, _ = V_kstep.shape
    
    lambda_powers = torch.pow(lmbda, torch.arange(max_T, device=device, dtype=dtype))
    
    weighted_sum = torch.zeros(num_trajs, max_T_actual, device=device, dtype=dtype)
    lambda_sum = torch.zeros(num_trajs, max_T_actual, device=device, dtype=dtype)
    
    for k in range(max_T):
        V_k = V_kstep[:, :, k]
        valid_k = (V_k != 0).float()
        weighted_sum += lambda_powers[k] * V_k * valid_k
        lambda_sum += lambda_powers[k] * valid_k
    
    G_lambda = weighted_sum / (lambda_sum + 1e-8)
    
    return G_lambda


def compute_joint_advantage_kstep(
    value_t: torch.Tensor,
    V_int_t: torch.Tensor,
    delta_ext_t: torch.Tensor,
    delta_int_t: torch.Tensor,
    z_i_t: torch.Tensor,
    done_t: torch.Tensor,
    gamma: float,
    lmbda: float,
    time_dim: int = 0,
    adv_clip_abs: float = 20.0,
    use_soft_min: bool = False,
    soft_min_tau: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    assert time_dim == 0, "time_dim must be 0"
    
    original_shape = value_t.shape
    T = value_t.shape[0]
    batch_dim = value_t[0].numel()
    device, dtype = value_t.device, value_t.dtype
    
    value_bt = value_t.reshape(T, batch_dim).transpose(0, 1).contiguous()
    V_int_bt = V_int_t.reshape(T, batch_dim).transpose(0, 1).contiguous()
    delta_ext_bt = delta_ext_t.reshape(T, batch_dim).transpose(0, 1).contiguous()
    delta_int_bt = delta_int_t.reshape(T, batch_dim).transpose(0, 1).contiguous()
    z_i_bt = z_i_t.reshape(T, batch_dim).transpose(0, 1).contiguous()
    done_bt = done_t.reshape(T, batch_dim).transpose(0, 1).contiguous()
    
    
    num_per_traj = _get_num_per_traj(done_bt)
    delta_ext_split, mask = _split_and_pad_sequence(delta_ext_bt, num_per_traj, return_mask=True)
    delta_int_split, _ = _split_and_pad_sequence(delta_int_bt, num_per_traj, return_mask=True)
    value_split, _ = _split_and_pad_sequence(value_bt, num_per_traj, return_mask=True)
    V_int_split, _ = _split_and_pad_sequence(V_int_bt, num_per_traj, return_mask=True)
    z_i_split, _ = _split_and_pad_sequence(z_i_bt, num_per_traj, return_mask=True)
    
    num_trajs, max_T_split = delta_ext_split.shape
    
    V_ext_kstep, V_int_kstep = _compute_kstep_returns(
        delta_ext_split, delta_int_split,
        value_split, V_int_split,
        gamma, max_T_split, device, dtype
    )
    value_broadcast = value_split.unsqueeze(2)
    V_ext_kstep.sub_(value_broadcast)
    del value_broadcast
    
    if use_soft_min:
        V_int_baseline = V_int_split.unsqueeze(2)
        V_int_kstep.sub_(V_int_baseline)
        del V_int_baseline
        
        valid_kstep_mask = (V_ext_kstep != 0) & (V_int_kstep != 0)
        valid_count = valid_kstep_mask.sum()
        if valid_count > 0:
            ext_selected_count = ((V_ext_kstep < V_int_kstep) & valid_kstep_mask).sum()
            ext_selection_rate = (ext_selected_count.float() / valid_count.float()).item()
            int_selection_rate = 1.0 - ext_selection_rate
            A_ext_kstep_mean = V_ext_kstep[valid_kstep_mask].mean().item()
            aug_V_int_kstep_mean = V_int_kstep[valid_kstep_mask].mean().item()
            del valid_kstep_mask
        else:
            ext_selection_rate = 0.0
            int_selection_rate = 0.0
            A_ext_kstep_mean = 0.0
            aug_V_int_kstep_mean = 0.0
            del valid_kstep_mask
        
        V_ext_kstep.sub_(V_int_kstep).div_(soft_min_tau)
        V_ext_kstep.neg_()
        V_int_kstep.sub_(
            torch.nn.functional.softplus(V_ext_kstep).mul_(soft_min_tau)
        )
        del V_ext_kstep
        V_ext_kstep = V_int_kstep
    else:
        z_i_broadcast = z_i_split.unsqueeze(2)
        V_int_kstep.sub_(z_i_broadcast)
        del z_i_broadcast
        
        valid_kstep_mask = (V_ext_kstep != 0) & (V_int_kstep != 0)
        valid_count = valid_kstep_mask.sum()
        if valid_count > 0:
            ext_selected_count = ((V_ext_kstep < V_int_kstep) & valid_kstep_mask).sum()
            ext_selection_rate = (ext_selected_count.float() / valid_count.float()).item()
            int_selection_rate = 1.0 - ext_selection_rate
            A_ext_kstep_mean = V_ext_kstep[valid_kstep_mask].mean().item()
            aug_V_int_kstep_mean = V_int_kstep[valid_kstep_mask].mean().item()
            del valid_kstep_mask
        else:
            ext_selection_rate = 0.0
            int_selection_rate = 0.0
            A_ext_kstep_mean = 0.0
            aug_V_int_kstep_mean = 0.0
            del valid_kstep_mask
        
        int_baseline = (V_int_split - z_i_split).unsqueeze(2)
        int_selected_mask = (V_int_kstep <= V_ext_kstep)
        torch.minimum(V_ext_kstep, V_int_kstep, out=V_ext_kstep)
        V_ext_kstep.sub_(int_baseline * int_selected_mask.float())
        del int_selected_mask, int_baseline
    
    del V_int_kstep
    
    G_JT = _lambda_weighted_value(V_ext_kstep, lmbda, max_T_split, device, dtype)
    del V_ext_kstep
    A_JT_split = G_JT
    
    A_JT_bt = A_JT_split[mask].view(batch_dim, T).clamp(min=-adv_clip_abs, max=adv_clip_abs)
    A_JT_tf = A_JT_bt.transpose(0, 1).contiguous()
    A_JT = A_JT_tf.reshape(original_shape)
    
    stats = {
        'A_ext_kstep_mean': A_ext_kstep_mean,
        'aug_V_int_kstep_mean': aug_V_int_kstep_mean,
        'ext_selection_rate': ext_selection_rate.item() if isinstance(ext_selection_rate, torch.Tensor) else ext_selection_rate,
        'int_selection_rate': int_selection_rate,
        'A_JT_mean': A_JT.mean().item(),
        'A_JT_std': A_JT.std().item(),
        'A_JT_min': A_JT.min().item(),
        'A_JT_max': A_JT.max().item(),
    }
    
    return A_JT, stats


def roll_z_traj_backward(intrinsic_reward: torch.Tensor, gamma: float) -> torch.Tensor:
    T = intrinsic_reward.shape[0]
    z_next = torch.zeros_like(intrinsic_reward[0])
    z_traj = []
    
    for t in range(T - 1, -1, -1):
        z_t = intrinsic_reward[t] + gamma * z_next
        z_t = torch.clamp(z_t, min=-1e3, max=1e3)
        z_traj.insert(0, z_t.detach())
        z_next = z_t
    
    return torch.stack(z_traj, dim=0).detach()


def compute_delta_int_with_z_prev(
    r_int: torch.Tensor,
    z_prev: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    z_prev_next = torch.cat([z_prev[1:], torch.zeros_like(z_prev[:1])], dim=0)
    expected_r_int = z_prev - gamma * z_prev_next
    return r_int - expected_r_int


class EpigraphGAE(TorchRLGAE):
    def __init__(
        self,
        *,
        gamma: float | torch.Tensor,
        lmbda: float | torch.Tensor,
        value_network: TensorDictModule | None,
        intrinsic_value_network: TensorDictModule | None = None,
        average_gae: bool = False,
        differentiable: bool = False,
        vectorized: bool | None = None,
        skip_existing: bool | None = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        shifted: bool = False,
        device: torch.device | None = None,
        time_dim: int | None = None,
        auto_reset_env: bool = False,
        deactivate_vmap: bool = False,
        intrinsic_gamma: float = 0.99,
        intrinsic_reward_key: NestedKey = None,
        use_soft_min: bool = False,
        soft_min_tau: float = 0.1,
        z_ema_alpha: float = 0.3,
        z0_best_reward_decay: float = 0.99,
        delta_running_momentum: float = 0.99,
        delta_int_scale: float = 1.0,
        delta_normalize: bool = False,
        adv_clip_abs: float = 20.0,
        initial_z_t: float = 0.0,
        ablation_current: bool = False,
    ):
        super().__init__(
            gamma=gamma,
            lmbda=lmbda,
            value_network=value_network,
            average_gae=average_gae,
            differentiable=differentiable,
            vectorized=vectorized,
            skip_existing=skip_existing,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            shifted=shifted,
            device=device,
            time_dim=time_dim,
            auto_reset_env=auto_reset_env,
            deactivate_vmap=deactivate_vmap,
        )
        
        self.intrinsic_gamma = intrinsic_gamma
        self.intrinsic_reward_key = intrinsic_reward_key
        self.intrinsic_value_network = intrinsic_value_network
        self.use_soft_min = use_soft_min
        self.soft_min_tau = soft_min_tau
        self.z_ema_alpha = z_ema_alpha
        self.z0_best_reward_decay = z0_best_reward_decay
        self.adv_clip_abs = adv_clip_abs
        self.initial_z_t = initial_z_t
        self.ablation_current = ablation_current
        
        self._z_prev: Optional[torch.Tensor] = None
        self._prev_iter_mean_ext: Optional[float] = None
        
        self._delta_ext_mean: float = 0.0
        self._delta_int_mean: float = 0.0
        self._v_int_target_mean: float = 0.0
        self._v_int_target_std: float = 0.0
        self._v_int_target_min: float = 0.0
        self._v_int_target_max: float = 0.0
        self._adv_int_gae_mean: float = 0.0
        self._v_int_t_mean: float = 0.0

        self._best_ext_reward: float = float('-inf')
        
        self._kstep_stats: Optional[dict] = None
        self._z_0_ema: Optional[torch.Tensor] = None
        self.warmup_mode: bool = False

    def set_warmup_mode(self, warmup: bool) -> None:
        self.warmup_mode = warmup
    
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        params: list[Tensor] | None = None,
        target_params: list[Tensor] | None = None,
        time_dim: int | None = None,
    ) -> TensorDictBase:
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimension, got "
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        
        intrinsic_reward_key = self.intrinsic_reward_key
        if intrinsic_reward_key is None:
            raise ValueError(
                "EpigraphGAE requires intrinsic_reward_key to be set. "
                "Use standard GAE for extrinsic_only or intrinsic_only modes."
            )
        if intrinsic_reward_key not in tensordict.keys(True, True):
            raise ValueError(
                f"Intrinsic rewards not found at key {intrinsic_reward_key}. "
                "Ensure process_batch() is called before advantage computation."
            )
        
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        if self.lmbda.device != device:
            self.lmbda = self.lmbda.to(device)
        lmbda = self.lmbda
        
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)
        
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network) if (
                params is None and target_params is None
            ) else nullcontext():
                value, next_value = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))
            
            if value is None:
                raise ValueError(
                    f"The tensor with key {self.tensor_keys.value} is missing, and no value network was provided."
                )
            if next_value is None:
                raise ValueError(
                    f"The tensor with key {('next', self.tensor_keys.value)} is missing, and no value network was provided."
                )
        
        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        time_dim = self._get_time_dim(time_dim, tensordict)
        
        if self.auto_reset_env:
            truncated = tensordict.get(("next", "truncated"))
            if truncated.any():
                reward = reward + gamma * value * truncated
        
        intrinsic_rewards = tensordict.get(intrinsic_reward_key)
        
        if intrinsic_rewards.shape != value.shape:
            if intrinsic_rewards.ndim > value.ndim:
                intrinsic_rewards = intrinsic_rewards.squeeze(-1)
        elif intrinsic_rewards.ndim > value.ndim and intrinsic_rewards.shape[-1] == 1:
            intrinsic_rewards = intrinsic_rewards.squeeze(-1)
        
        terminated_float = terminated.float() if not self.auto_reset_env else done.float()
        delta_ext = reward + gamma * next_value * (1.0 - terminated_float) - value

        time_dim = self._get_time_dim(time_dim, tensordict)
        if time_dim != 0:
            delta_ext_t = delta_ext.transpose(0, time_dim).contiguous()
            done_t = done.transpose(0, time_dim).contiguous()
            value_t = value.transpose(0, time_dim).contiguous()
        else:
            delta_ext_t = delta_ext
            done_t = done
            value_t = value

        gamma_val = gamma.item() if isinstance(gamma, Tensor) else gamma
        lmbda_val = lmbda.item() if isinstance(lmbda, Tensor) else lmbda

        adv_ext_gae = _vectorized_gae_from_deltas(
            deltas=delta_ext_t,
            done=done_t,
            gamma=gamma_val,
            lmbda=lmbda_val,
            time_dim=0,
        )
        
        T = tensordict.batch_size[time_dim]
        
        if time_dim != 0:
            intrinsic_rewards_t = intrinsic_rewards.transpose(0, time_dim).contiguous()
            done_t = done.transpose(0, time_dim).contiguous()
        else:
            intrinsic_rewards_t = intrinsic_rewards
            done_t = done
        
        gamma_scalar = gamma.item() if isinstance(gamma, Tensor) else gamma
        lmbda_scalar = lmbda.item() if isinstance(lmbda, Tensor) else lmbda

        T = tensordict.batch_size[time_dim]
        
        if time_dim != 0:
            intrinsic_rewards_t = intrinsic_rewards.transpose(0, time_dim).contiguous()
            done_t = done.transpose(0, time_dim).contiguous()
            terminated_t = terminated.transpose(0, time_dim).contiguous()
        else:
            intrinsic_rewards_t = intrinsic_rewards
            done_t = done
            terminated_t = terminated

        if intrinsic_rewards_t.ndim >= 3:
            num_envs = intrinsic_rewards_t.shape[1]
            num_agents = intrinsic_rewards_t.shape[2]
        elif intrinsic_rewards_t.ndim == 2:
            total_elements = intrinsic_rewards_t.shape[1]
            num_envs = 1
            num_agents = total_elements
        else:
            raise ValueError(f"Unexpected intrinsic_rewards_t.ndim: {intrinsic_rewards_t.ndim}, shape: {intrinsic_rewards_t.shape}")

        if self._z_0_ema is None:
            self._z_0_ema = torch.ones(num_agents, device=intrinsic_rewards_t.device) * self.initial_z_t

        z_0_per_env_agent = self._z_0_ema.unsqueeze(0).expand(num_envs, -1)
        z_0_flat = z_0_per_env_agent.reshape(-1)
        
        original_shape = intrinsic_rewards_t.shape
        flat_shape = (T, -1)
        r_int_flat = intrinsic_rewards_t.reshape(*flat_shape)
        
        z_traj_flat = self._compute_z_trajectory(r_int_flat, gamma_val, z_0_flat)
        z_traj_t = z_traj_flat.reshape(original_shape)

        z_i_t = z_traj_t

        if time_dim != 0:
            z_i_for_td = z_i_t.transpose(0, time_dim).contiguous()
        else:
            z_i_for_td = z_i_t
        if isinstance(intrinsic_reward_key, tuple):
            current_z_key = intrinsic_reward_key[:-1] + ("z_i",)
        else:
            current_z_key = "z_i"
        tensordict.set(current_z_key, z_i_for_td)

        z_i_next_t = torch.cat(
            [z_i_t[1:], z_i_t[-1:].detach()],
            dim=0,
        )
        if time_dim != 0:
            z_i_next = z_i_next_t.transpose(0, time_dim).contiguous()
        else:
            z_i_next = z_i_next_t
        td_next = tensordict.get("next")
        if td_next is None:
            raise ValueError(
                "EpigraphGAE requires 'next' tensordict to compute V^int(o_i_next, z_i_next). "
                "Ensure the batch contains next state information."
            )
        if isinstance(intrinsic_reward_key, tuple):
            next_z_key = intrinsic_reward_key[:-1] + ("z_i",)
        else:
            next_z_key = "z_i"
        td_next.set(next_z_key, z_i_next)

        gamma_scalar_int = gamma_val
        lmbda_scalar_int = lmbda_val
        terminated_float_t = terminated_t.float() if not self.auto_reset_env else done_t.float()

        if self.intrinsic_value_network is None:
            raise ValueError(
                "EpigraphGAE requires intrinsic_value_network to be provided. "
                "Intrinsic value network is required for proper TD bootstrapping in delta^int computation. "
                "Please provide intrinsic_value_network when creating EpigraphGAE."
            )

        with torch.no_grad():
            v_int_output = self.intrinsic_value_network(tensordict)
            v_int_next_output = self.intrinsic_value_network(td_next)

            if isinstance(self.intrinsic_reward_key, tuple):
                intrinsic_value_key = (self.intrinsic_reward_key[0], "state_value_int")
            else:
                intrinsic_value_key = None
                for key in v_int_output.keys(True, True):
                    if "state_value_int" in str(key):
                        intrinsic_value_key = key
                        break
                if intrinsic_value_key is None:
                    raise ValueError(
                        f"Intrinsic value network did not return 'state_value_int' key. "
                        f"Available keys (current): {list(v_int_output.keys(True, True))}. "
                        f"Ensure the intrinsic value network outputs to the expected key."
                    )

            V_int_t = v_int_output.get(intrinsic_value_key)
            V_int_next_t = v_int_next_output.get(intrinsic_value_key)
            if V_int_t is None or V_int_next_t is None:
                raise ValueError(
                    f"Intrinsic value network did not return key '{intrinsic_value_key}' "
                    f"for current and next states. "
                    f"Current keys: {list(v_int_output.keys(True, True))}, "
                    f"Next keys: {list(v_int_next_output.keys(True, True))}."
                )

            if time_dim != 0:
                V_int_t = V_int_t.transpose(0, time_dim).contiguous()
                V_int_next_t = V_int_next_t.transpose(0, time_dim).contiguous()

            if V_int_t.shape != z_traj_t.shape:
                V_int_t = V_int_t.expand_as(z_traj_t)
            if V_int_next_t.shape != z_traj_t.shape:
                V_int_next_t = V_int_next_t.expand_as(z_traj_t)

            V_int_t = torch.clamp(V_int_t, min=0.0)
            V_int_next_t = torch.clamp(V_int_next_t, min=0.0)

        delta_int_t = intrinsic_rewards_t + V_int_next_t * (1.0 - terminated_float_t) - V_int_t

        z_i_t = z_traj_t

        adv_int_gae = _vectorized_gae_from_deltas(
            deltas=delta_int_t,
            done=done_t,
            gamma=gamma_scalar_int,
            lmbda=lmbda_scalar_int,
            time_dim=0,
        )

        v_int_target_t = adv_int_gae + V_int_t
        
        with torch.no_grad():
            self._v_int_target_mean = v_int_target_t.mean().item()
            self._v_int_target_std = v_int_target_t.std().item()
            self._v_int_target_min = v_int_target_t.min().item()
            self._v_int_target_max = v_int_target_t.max().item()
            self._adv_int_gae_mean = adv_int_gae.mean().item()
            self._v_int_t_mean = V_int_t.mean().item()
        
        if time_dim != 0:
            v_int_target = v_int_target_t.transpose(0, time_dim).contiguous()
        else:
            v_int_target = v_int_target_t
        if isinstance(intrinsic_reward_key, tuple):
            v_int_target_key = (intrinsic_reward_key[0], "v_int_target_int")
        else:
            v_int_target_key = "v_int_target_int"
        tensordict.set(v_int_target_key, v_int_target)

        delta_int_scaled = adv_int_gae
        

        self._delta_ext_mean = adv_ext_gae.mean().item()
        self._delta_int_mean = delta_int_scaled.mean().item()
        
        
        if self.warmup_mode:
            adv = adv_ext_gae
            self._kstep_stats = None
        else:
            adv, kstep_stats = compute_joint_advantage_kstep(
                value_t=value_t,
                V_int_t=V_int_t,
                delta_ext_t=delta_ext_t,
                delta_int_t=delta_int_t,
                z_i_t=z_i_t,
                done_t=done_t,
                gamma=gamma_val,
                lmbda=lmbda_val,
                time_dim=0,
                adv_clip_abs=self.adv_clip_abs,
                use_soft_min=self.use_soft_min,
                soft_min_tau=self.soft_min_tau,
            )
            self._kstep_stats = kstep_stats
        
        
        value_target = adv_ext_gae + value_t
        
        if time_dim != 0:
            adv = adv.transpose(0, time_dim)
            value_target = value_target.transpose(0, time_dim)
            z_i = z_i_t.transpose(0, time_dim)
        else:
            z_i = z_i_t
        
        if self.average_gae:
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)
            adv = adv - loc
            adv = adv / scale
        
        tensordict.set(self.tensor_keys.advantage, adv)
        tensordict.set(self.tensor_keys.value_target, value_target)
        
        
        return tensordict
    
    def _compute_z_trajectory(
        self,
        r_int: torch.Tensor,
        gamma: float,
        z_0: torch.Tensor,
    ) -> torch.Tensor:
        T = r_int.shape[0]
        z_traj = []
        z_t = z_0
        
        for t in range(T):
            z_traj.append(z_t)
            z_next = (z_t - r_int[t]) / gamma
            z_next = torch.clamp(z_next, min=-1e3, max=1e3)
            z_t = z_next
        
        z_traj_tensor = torch.stack(z_traj, dim=0)
        return z_traj_tensor
    
    def _update_z0_from_episode_data(self, ext_episode_sums: torch.Tensor, z0_episode_sums: torch.Tensor) -> None:
        if len(ext_episode_sums) == 0:
            return

        device = ext_episode_sums.device
        
        if self._z_0_ema is None:
            best_ext_idx = ext_episode_sums.argmax()
            self._z_0_ema = z0_episode_sums[best_ext_idx].detach()
            self._best_ext_reward = ext_episode_sums[best_ext_idx].item()
            if self.ablation_current:
                self._prev_iter_mean_ext = ext_episode_sums.mean().item()
            return
        
        if self._z_0_ema.device != device:
            self._z_0_ema = self._z_0_ema.to(device)
        
        if self.ablation_current:
            ref_bar_j = (
                self._prev_iter_mean_ext
                if self._prev_iter_mean_ext is not None
                else float("-inf")
            )
            better_mask = ext_episode_sums > ref_bar_j
        else:
            better_mask = ext_episode_sums > self._best_ext_reward
        num_better = better_mask.sum().item()
        
        if num_better > 0:
            intrinsic_totals = z0_episode_sums.sum(dim=1)
            
            masked_intrinsic = torch.where(better_mask, intrinsic_totals, torch.tensor(float('-inf'), device=device))
            
            selected_idx = masked_intrinsic.argmax()
            selected_z0 = z0_episode_sums[selected_idx]
            
            self._z_0_ema = (
                self.z_ema_alpha * selected_z0.detach()
                + (1 - self.z_ema_alpha) * self._z_0_ema
            )
            if not self.ablation_current:
                self._best_ext_reward = ext_episode_sums[selected_idx].item()
        else:
            pass
        
        if self.ablation_current:
            self._prev_iter_mean_ext = ext_episode_sums.mean().item()
        else:
            if self._best_ext_reward >= 0:
                self._best_ext_reward = self._best_ext_reward * self.z0_best_reward_decay
            else:
                self._best_ext_reward = self._best_ext_reward / self.z0_best_reward_decay

    def get_epigraph_stats(self) -> Dict[str, float]:
        stats = {
            "epigraph/delta_ext_mean": self._delta_ext_mean,
            "epigraph/delta_int_mean": self._delta_int_mean,
            "debug/adv_int_gae_mean": self._adv_int_gae_mean,
            "debug/v_int_t_mean": self._v_int_t_mean,
        }

        if hasattr(self, '_kstep_stats') and self._kstep_stats is not None:
            stats.update({
                "epigraph/kstep_A_ext_mean": self._kstep_stats['A_ext_kstep_mean'],
                "epigraph/kstep_aug_V_int_mean": self._kstep_stats['aug_V_int_kstep_mean'],
                "epigraph/kstep_ext_selection_rate": self._kstep_stats['ext_selection_rate'],
                "epigraph/kstep_int_selection_rate": self._kstep_stats['int_selection_rate'],
                "epigraph/A_JT_mean": self._kstep_stats['A_JT_mean'],
                "epigraph/A_JT_std": self._kstep_stats['A_JT_std'],
                "epigraph/A_JT_min": self._kstep_stats['A_JT_min'],
                "epigraph/A_JT_max": self._kstep_stats['A_JT_max'],
            })

        if hasattr(self, '_z_0_ema') and self._z_0_ema is not None:
            stats.update({
                "epigraph/z0_mean": self._z_0_ema.mean().item(),
                "epigraph/z0_std": self._z_0_ema.std().item(),
                "epigraph/z0_min": self._z_0_ema.min().item(),
                "epigraph/z0_max": self._z_0_ema.max().item(),
            })

        if hasattr(self, '_best_ext_reward'):
            stats["epigraph/best_ext_reward"] = self._best_ext_reward
        if self.ablation_current and self._prev_iter_mean_ext is not None:
            stats["epigraph/ablation_bar_J_mean_ext"] = self._prev_iter_mean_ext

        return stats

class LagrangianGAE(TorchRLGAE):
    def __init__(
        self,
        *,
        gamma: float | torch.Tensor,
        lmbda: float | torch.Tensor,
        value_network: TensorDictModule | None,
        intrinsic_value_network: TensorDictModule | None = None,
        average_gae: bool = False,
        differentiable: bool = False,
        vectorized: bool | None = None,
        skip_existing: bool | None = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        shifted: bool = False,
        device: torch.device | None = None,
        time_dim: int | None = None,
        auto_reset_env: bool = False,
        deactivate_vmap: bool = False,
        intrinsic_gamma: float = 0.99,
        intrinsic_reward_key: NestedKey = None,
        lambda_init: float = 1.0,
        lambda_lr: float = 0.01,
        lambda_max: float = 10.0,
        adv_clip_abs: float = 20.0,
        delta_running_momentum: float = 0.99,
        delta_int_scale: float = 1.0,
        delta_normalize: bool = False,
    ):
        super().__init__(
            gamma=gamma,
            lmbda=lmbda,
            value_network=value_network,
            average_gae=average_gae,
            differentiable=differentiable,
            vectorized=vectorized,
            skip_existing=skip_existing,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            shifted=shifted,
            device=device,
            time_dim=time_dim,
            auto_reset_env=auto_reset_env,
            deactivate_vmap=deactivate_vmap,
        )
        
        self.intrinsic_gamma = intrinsic_gamma
        self.intrinsic_reward_key = intrinsic_reward_key
        self.intrinsic_value_network = intrinsic_value_network
        self.adv_clip_abs = adv_clip_abs
        
        self._lambda = lambda_init
        self._lambda_lr = lambda_lr
        self._lambda_max = lambda_max
        
        self._delta_ext_mean: float = 0.0
        self._delta_int_mean: float = 0.0
        self._adv_combined_mean: float = 0.0
        
        self._v_int_target_mean: float = 0.0
        self._v_int_target_std: float = 0.0
        self._v_int_target_min: float = 0.0
        self._v_int_target_max: float = 0.0
        self._adv_int_gae_mean: float = 0.0
        self._v_int_t_mean: float = 0.0
        
        self.warmup_mode: bool = False
    
    def set_lambda(self, value: float) -> None:
        self._lambda = max(0.0, min(value, self._lambda_max))
    
    def set_warmup_mode(self, warmup: bool) -> None:
        self.warmup_mode = warmup
    
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        params: list[Tensor] | None = None,
        target_params: list[Tensor] | None = None,
        time_dim: int | None = None,
    ) -> TensorDictBase:
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimension, got "
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        
        intrinsic_reward_key = self.intrinsic_reward_key
        if intrinsic_reward_key is None:
            raise ValueError(
                "LagrangianGAE requires intrinsic_reward_key to be set."
            )
        if intrinsic_reward_key not in tensordict.keys(True, True):
            raise ValueError(
                f"Intrinsic rewards not found at key {intrinsic_reward_key}."
            )
        
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        if self.lmbda.device != device:
            self.lmbda = self.lmbda.to(device)
        lmbda = self.lmbda
        
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)
        
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network) if (
                params is None and target_params is None
            ) else nullcontext():
                value, next_value = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))
            if value is None or next_value is None:
                raise ValueError("Value network or pre-computed values required.")
        
        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        time_dim = self._get_time_dim(time_dim, tensordict)
        
        if self.auto_reset_env:
            truncated = tensordict.get(("next", "truncated"))
            if truncated.any():
                reward = reward + gamma * value * truncated
        
        intrinsic_rewards = tensordict.get(intrinsic_reward_key)
        if intrinsic_rewards.shape != value.shape:
            if intrinsic_rewards.ndim > value.ndim:
                intrinsic_rewards = intrinsic_rewards.squeeze(-1)
        
        terminated_float = terminated.float() if not self.auto_reset_env else done.float()
        delta_ext = reward + gamma * next_value * (1.0 - terminated_float) - value
        
        time_dim = self._get_time_dim(time_dim, tensordict)
        if time_dim != 0:
            delta_ext_t = delta_ext.transpose(0, time_dim).contiguous()
            done_t = done.transpose(0, time_dim).contiguous()
            value_t = value.transpose(0, time_dim).contiguous()
            intrinsic_rewards_t = intrinsic_rewards.transpose(0, time_dim).contiguous()
            terminated_t = terminated.transpose(0, time_dim).contiguous()
        else:
            delta_ext_t = delta_ext
            done_t = done
            value_t = value
            intrinsic_rewards_t = intrinsic_rewards
            terminated_t = terminated
        
        gamma_val = gamma.item() if isinstance(gamma, Tensor) else gamma
        lmbda_val = lmbda.item() if isinstance(lmbda, Tensor) else lmbda
        
        adv_ext_gae = _vectorized_gae_from_deltas(
            deltas=delta_ext_t,
            done=done_t,
            gamma=gamma_val,
            lmbda=lmbda_val,
            time_dim=0,
        )
        
        T = tensordict.batch_size[time_dim]
        
        z_i_t = roll_z_traj_backward(intrinsic_rewards_t, self.intrinsic_gamma)
        
        if time_dim != 0:
            z_i_for_td = z_i_t.transpose(0, time_dim).contiguous()
        else:
            z_i_for_td = z_i_t
        if isinstance(intrinsic_reward_key, tuple):
            current_z_key = intrinsic_reward_key[:-1] + ("z_i",)
        else:
            current_z_key = "z_i"
        tensordict.set(current_z_key, z_i_for_td)
        
        z_i_next_t = torch.cat([z_i_t[1:], z_i_t[-1:].detach()], dim=0)
        if time_dim != 0:
            z_i_next = z_i_next_t.transpose(0, time_dim).contiguous()
        else:
            z_i_next = z_i_next_t
        td_next = tensordict.get("next")
        if isinstance(intrinsic_reward_key, tuple):
            next_z_key = intrinsic_reward_key[:-1] + ("z_i",)
        else:
            next_z_key = "z_i"
        td_next.set(next_z_key, z_i_next)
        
        terminated_float_t = terminated_t.float() if not self.auto_reset_env else done_t.float()
        
        if self.intrinsic_value_network is None:
            raise ValueError("LagrangianGAE requires intrinsic_value_network.")
        
        with torch.no_grad():
            v_int_output = self.intrinsic_value_network(tensordict)
            v_int_next_output = self.intrinsic_value_network(td_next)
            
            if isinstance(self.intrinsic_reward_key, tuple):
                intrinsic_value_key = (self.intrinsic_reward_key[0], "state_value_int")
            else:
                intrinsic_value_key = None
                for key in v_int_output.keys(True, True):
                    if "state_value_int" in str(key):
                        intrinsic_value_key = key
                        break
                if intrinsic_value_key is None:
                    raise ValueError("Intrinsic value network did not return 'state_value_int' key.")
            
            V_int_t = v_int_output.get(intrinsic_value_key)
            V_int_next_t = v_int_next_output.get(intrinsic_value_key)
            
            if time_dim != 0:
                V_int_t = V_int_t.transpose(0, time_dim).contiguous()
                V_int_next_t = V_int_next_t.transpose(0, time_dim).contiguous()
            
            if V_int_t.shape != z_i_t.shape:
                V_int_t = V_int_t.expand_as(z_i_t)
            if V_int_next_t.shape != z_i_t.shape:
                V_int_next_t = V_int_next_t.expand_as(z_i_t)
            
            V_int_t = torch.clamp(V_int_t, min=0.0)
            V_int_next_t = torch.clamp(V_int_next_t, min=0.0)
        
        delta_int_t = intrinsic_rewards_t + V_int_next_t * (1.0 - terminated_float_t) - V_int_t
        
        adv_int_gae = _vectorized_gae_from_deltas(
            deltas=delta_int_t,
            done=done_t,
            gamma=gamma_val,
            lmbda=lmbda_val,
            time_dim=0,
        )
        
        v_int_target_t = adv_int_gae + V_int_t
        with torch.no_grad():
            self._v_int_target_mean = v_int_target_t.mean().item()
            self._v_int_target_std = v_int_target_t.std().item()
            self._v_int_target_min = v_int_target_t.min().item()
            self._v_int_target_max = v_int_target_t.max().item()
            self._adv_int_gae_mean = adv_int_gae.mean().item()
            self._v_int_t_mean = V_int_t.mean().item()
        
        if time_dim != 0:
            v_int_target = v_int_target_t.transpose(0, time_dim).contiguous()
        else:
            v_int_target = v_int_target_t
        if isinstance(intrinsic_reward_key, tuple):
            v_int_target_key = (intrinsic_reward_key[0], "v_int_target_int")
        else:
            v_int_target_key = "v_int_target_int"
        tensordict.set(v_int_target_key, v_int_target)
        
        adv_int_scaled = adv_int_gae
        
        self._delta_ext_mean = adv_ext_gae.mean().item()
        self._delta_int_mean = adv_int_scaled.mean().item()
        
        if self.warmup_mode:
            adv = adv_ext_gae
        else:
            adv = adv_int_scaled + self._lambda * adv_ext_gae
            
            lambda_grad = adv_ext_gae.mean().item()
            new_lambda = self._lambda - self._lambda_lr * lambda_grad
            self._lambda = max(0.0, min(new_lambda, self._lambda_max))
        
        self._adv_combined_mean = adv.mean().item()
        
        adv = torch.clamp(adv, min=-self.adv_clip_abs, max=self.adv_clip_abs)
        
        value_target = adv_ext_gae + value_t
        
        if time_dim != 0:
            adv = adv.transpose(0, time_dim)
            value_target = value_target.transpose(0, time_dim)
            z_i = z_i_t.transpose(0, time_dim)
        else:
            z_i = z_i_t
        
        if self.average_gae:
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)
            adv = (adv - loc) / scale
        
        tensordict.set(self.tensor_keys.advantage, adv)
        tensordict.set(self.tensor_keys.value_target, value_target)
        
        return tensordict
    
    def get_lagrangian_stats(self) -> Dict[str, float]:
        stats = {
            "lagrangian/lambda": self._lambda,
            "lagrangian/lambda_lr": self._lambda_lr,
            "lagrangian/delta_ext_mean": self._delta_ext_mean,
            "lagrangian/delta_int_mean": self._delta_int_mean,
            "lagrangian/adv_combined_mean": self._adv_combined_mean,
            "debug/adv_int_gae_mean": self._adv_int_gae_mean,
            "debug/v_int_t_mean": self._v_int_t_mean,
        }
        return stats


def make_value_estimator(self, value_type: ValueEstimators | EpigraphValueEstimators | str = None, **hyperparams):
    if value_type is None:
        value_type = self.default_value_estimator
    self.value_type = value_type
    hp = dict(default_value_kwargs(value_type))
    if hasattr(self, "gamma"):
        hp["gamma"] = self.gamma
    hp.update(hyperparams)
    if value_type == ValueEstimators.TD1:
        self._value_estimator = TD1Estimator(
            value_network=self.critic_network, **hp
        )
    elif value_type == ValueEstimators.TD0:
        self._value_estimator = TD0Estimator(
            value_network=self.critic_network, **hp
        )
    elif value_type == ValueEstimators.GAE:
        self._value_estimator = TorchRLGAE(value_network=self.critic_network, **hp)
    elif value_type == EpigraphValueEstimators.EPIGRAPH_GAE or value_type == "EpigraphGAE":
        epigraph_params = {}
        for key in ['intrinsic_gamma', 'intrinsic_reward_key', 'use_soft_min', 
                    'soft_min_tau', 'z_ema_alpha', 'z0_best_reward_decay', 'delta_running_momentum', 
                    'delta_int_scale', 'delta_normalize', 'adv_clip_abs', 'initial_z_t',
                    'ablation_current']:
            if key in hp:
                epigraph_params[key] = hp.pop(key)
        
        self._value_estimator = EpigraphGAE(
            value_network=self.critic_network, 
            **hp,
            **epigraph_params
        )
    elif value_type == EpigraphValueEstimators.LAGRANGIAN_GAE or value_type == "LagrangianGAE":
        lagrangian_params = {}
        for key in ['intrinsic_gamma', 'intrinsic_reward_key', 'lambda_init', 
                    'lambda_lr', 'lambda_max', 'delta_running_momentum', 
                    'delta_int_scale', 'delta_normalize', 'adv_clip_abs',
                    'intrinsic_value_network']:
            if key in hp:
                lagrangian_params[key] = hp.pop(key)
        
        self._value_estimator = LagrangianGAE(
            value_network=self.critic_network, 
            **hp,
            **lagrangian_params
        )
    elif value_type == ValueEstimators.TDLambda:
        self._value_estimator = TDLambdaEstimator(
            value_network=self.critic_network, **hp
        )
    elif value_type == ValueEstimators.VTrace:
        if self.functional:
            actor_with_params = deepcopy(self.actor_network)
            self.actor_network_params.to_module(actor_with_params)
        else:
            actor_with_params = self.actor_network
        self._value_estimator = VTrace(
            value_network=self.critic_network, actor_network=actor_with_params, **hp
        )
    else:
        raise NotImplementedError(f"Unknown value type {value_type}")