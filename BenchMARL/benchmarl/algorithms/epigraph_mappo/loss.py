from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.objectives.utils import _reduce
from tensordict.nn import TensorDictModule
from .torchrl_overrides import default_value_kwargs
from torchrl.objectives.value.advantages import (
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
)
from .torchrl_overrides import EpigraphGAE, LagrangianGAE, EpigraphValueEstimators


class JointClipPPOLoss(ClipPPOLoss):
    def __init__(
        self,
        *args,
        z_key: Optional[Tuple[str, ...]] = None,
        intrinsic_key: Optional[Tuple[str, ...]] = None,
        q_clip_max: float = 100.0,
        vf_clip_max: float = 1000.0,
        use_huber_loss: bool = True,
        huber_delta: float = 1.0,
        use_value_norm: bool = False,
        intrinsic_only: bool = False,
        intrinsic_gamma: Optional[float] = None,
        intrinsic_value_network: Optional[TensorDictModule] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._z_key = z_key
        self._intrinsic_key = intrinsic_key
        self.q_clip_max = q_clip_max
        self.vf_clip_max = vf_clip_max
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.use_value_norm = use_value_norm
        self.intrinsic_only = intrinsic_only
        self.intrinsic_gamma = intrinsic_gamma
        self.intrinsic_value_network = intrinsic_value_network

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        log_prob_key = None
        
        for key, value in tensordict.items(True, True):
            if isinstance(value, torch.Tensor):
                if log_prob_key is None:
                    key_str = str(key)
                    if ('log_prob' in key_str or 'sample_log_prob' in key_str):
                        log_prob_key = key
        
        if log_prob_key is not None:
            log_prob = tensordict.get(log_prob_key)
            if isinstance(log_prob, torch.Tensor):
                tensordict.set(log_prob_key, torch.clamp(log_prob, min=-50.0, max=50.0))
        
        self._apply_critic_stabilization(tensordict)
        
        loss_td = super().forward(tensordict)

        if self.intrinsic_value_network is not None:
            if self._intrinsic_key is not None:
                if isinstance(self._intrinsic_key, tuple):
                    v_int_target_key = (self._intrinsic_key[0], "v_int_target_int")
                else:
                    v_int_target_key = "v_int_target_int"
            else:
                v_int_target_key = None
                for key in tensordict.keys(True, True):
                    if "v_int_target_int" in str(key):
                        v_int_target_key = key
                        break
            v_int_target = tensordict.get(v_int_target_key) if v_int_target_key is not None else None
            if v_int_target is not None:
                if self._intrinsic_key is not None:
                    if isinstance(self._intrinsic_key, tuple):
                        v_int_old_key = (self._intrinsic_key[0], "state_value_int")
                    else:
                        v_int_old_key = "state_value_int"
                else:
                    v_int_old_key = None
                    for key in tensordict.keys(True, True):
                        if "state_value_int" in str(key) and "old" not in str(key):
                            v_int_old_key = key
                            break
                
                v_int_old = tensordict.get(v_int_old_key) if v_int_old_key is not None else None
                
                v_int_output = self.intrinsic_value_network(tensordict)
                v_int_pred_key = v_int_old_key
                v_int_pred = v_int_output.get(v_int_pred_key) if v_int_output is not None else None

                if v_int_pred is not None:
                    if v_int_old is not None:
                        v_int_old = torch.clamp(v_int_old, min=0.0)
                    
                    v_int_loss = self._compute_value_loss_torchrl_style(
                        value_pred=v_int_pred,
                        value_target=v_int_target,
                        value_old=v_int_old,
                    )

                    loss_td.set("loss_critic_int", self.critic_coeff * v_int_loss)

                    loss_td.set("loss_v_int", v_int_loss.detach())
                    
                    loss_td.set("debug/v_int_pred_mean", v_int_pred.mean().detach())
                    loss_td.set("debug/v_int_pred_std", v_int_pred.std().detach())
                    loss_td.set("debug/v_int_pred_min", v_int_pred.min().detach())
                    loss_td.set("debug/v_int_pred_max", v_int_pred.max().detach())
                    
                    loss_td.set("debug/v_int_target_mean", v_int_target.mean().detach())
                    loss_td.set("debug/v_int_target_std", v_int_target.std().detach())
                    loss_td.set("debug/v_int_target_min", v_int_target.min().detach())
                    loss_td.set("debug/v_int_target_max", v_int_target.max().detach())
                    
                    if v_int_old is not None:
                        loss_td.set("debug/v_int_old_mean", v_int_old.mean().detach())
                        loss_td.set("debug/v_int_old_std", v_int_old.std().detach())
                        loss_td.set("debug/v_int_old_min", v_int_old.min().detach())
                        loss_td.set("debug/v_int_old_max", v_int_old.max().detach())
                    
                    v_int_error = (v_int_target - v_int_pred).abs()
                    loss_td.set("debug/v_int_error_mean", v_int_error.mean().detach())
                    loss_td.set("debug/v_int_error_max", v_int_error.max().detach())

        if self._z_key is not None:
            try:
                z_vals = tensordict.get(self._z_key)
                if z_vals is not None:
                    loss_td.set("z_i_mean", z_vals.mean())
                    loss_td.set("z_i_std", z_vals.std(unbiased=False))
            except (KeyError, AttributeError):
                pass

        if self._intrinsic_key is not None:
            try:
                intr_vals = tensordict.get(self._intrinsic_key)
                if intr_vals is not None:
                    loss_td.set("intrinsic_mean", intr_vals.mean())
                    loss_td.set("intrinsic_std", intr_vals.std(unbiased=False))
            except (KeyError, AttributeError):
                pass

        return loss_td


    def _compute_value_loss_torchrl_style(
        self,
        value_pred: torch.Tensor,
        value_target: torch.Tensor,
        value_old: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from torchrl.objectives.utils import distance_loss
        
        loss_value = distance_loss(
            value_target,
            value_pred,
            loss_function=self.loss_critic_type,
        )
        
        if value_old is not None and self.clip_value is not None:
            pre_clipped = value_pred - value_old
            clipped = pre_clipped.clamp(-self.clip_value, self.clip_value)
            value_pred_clipped = value_old + clipped
            
            loss_value_clipped = distance_loss(
                value_target,
                value_pred_clipped,
                loss_function=self.loss_critic_type,
            )
            loss_value = torch.maximum(loss_value, loss_value_clipped)
        
        return loss_value.mean()
    
    def _apply_critic_stabilization(self, tensordict: TensorDictBase) -> None:
        if self.vf_clip_max is None or self.vf_clip_max == float('inf'):
            return
        
        value_key = None
        value_target_key = None
        
        for key in tensordict.keys(True, True):
            key_str = str(key)
            if 'state_value' in key_str or ('value' in key_str and 'target' not in key_str):
                if value_key is None:
                    value_key = key
            if 'value_target' in key_str:
                value_target_key = key
        
        if value_key is not None:
            value = tensordict.get(value_key)
            if isinstance(value, torch.Tensor):
                value_clipped = torch.clamp(value, min=-self.vf_clip_max, max=self.vf_clip_max)
                tensordict.set(value_key, value_clipped)
        
        if value_target_key is not None:
            target = tensordict.get(value_target_key)
            if isinstance(target, torch.Tensor):
                target_clipped = torch.clamp(target, min=-self.vf_clip_max, max=self.vf_clip_max)
                tensordict.set(value_target_key, target_clipped)
                
                if self.use_value_norm and value_key is not None:
                    value = tensordict.get(value_key)
                    if isinstance(value, torch.Tensor):
                        target_mean = target_clipped.mean()
                        target_std = target_clipped.std() + 1e-8
                        target_norm = (target_clipped - target_mean) / target_std
                        value_norm = (value - target_mean) / target_std
                        tensordict.set(value_target_key, target_norm)
                        tensordict.set(value_key, value_norm)
    
    def _compute_reinforce_advantages(self, tensordict: TensorDictBase) -> None:
        intrinsic_rewards = tensordict.get(self._intrinsic_key)
        if intrinsic_rewards is None:
            return
        
        if not isinstance(intrinsic_rewards, torch.Tensor):
            return
        
        done = None
        done_key = None
        for key in tensordict.keys(True, True):
            key_str = str(key)
            if 'done' in key_str and 'terminated' not in key_str:
                done = tensordict.get(key)
                done_key = key
                break
        
        if done is None:
            for key in tensordict.keys(True, True):
                key_str = str(key)
                if 'terminated' in key_str:
                    done = tensordict.get(key)
                    done_key = key
                    break
        
        if done is None:
            done = torch.zeros_like(intrinsic_rewards, dtype=torch.bool)
        
        if intrinsic_rewards.ndim > 3:
            intrinsic_rewards = intrinsic_rewards.squeeze(-1)
        if done.ndim > 3:
            done = done.squeeze(-1)
        elif done.ndim == 2:
            num_agents = intrinsic_rewards.shape[-1] if intrinsic_rewards.ndim == 3 else 1
            done = done.unsqueeze(-1).expand(-1, -1, num_agents)
        
        done = done.bool() if not done.is_floating_point() else (done > 0.5)
        
        advantage_key = None
        for key in tensordict.keys(True, True):
            key_str = str(key)
            if 'advantage' in key_str:
                advantage_key = key
                break
        
        if advantage_key is None:
            return
        
        T = intrinsic_rewards.shape[0]
        returns = torch.zeros_like(intrinsic_rewards)
        returns_next = torch.zeros_like(intrinsic_rewards[0])
        
        for t in range(T - 1, -1, -1):
            done_t = done[t].float()
            returns[t] = intrinsic_rewards[t] + self.intrinsic_gamma * (1.0 - done_t) * returns_next
            returns_next = returns[t]
        
        advantages_expected = tensordict.get(advantage_key)
        if advantages_expected.ndim > returns.ndim:
            returns = returns.unsqueeze(-1)
        
        tensordict.set(advantage_key, returns)
        
        value_target_key = None
        for key in tensordict.keys(True, True):
            key_str = str(key)
            if 'value_target' in key_str:
                value_target_key = key
                break
        
        if value_target_key is not None:
            value_targets_expected = tensordict.get(value_target_key)
            zero_targets = torch.zeros_like(value_targets_expected)
            tensordict.set(value_target_key, zero_targets)
    
    def _huber_loss(self, error: torch.Tensor) -> torch.Tensor:
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=self.huber_delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic.pow(2) + self.huber_delta * linear
    
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
            from torchrl.objectives.value.advantages import GAE as TorchRLGAE
            self._value_estimator = TorchRLGAE(value_network=self.critic_network, **hp)
        elif value_type == EpigraphValueEstimators.EPIGRAPH_GAE or value_type == "EpigraphGAE":
            self._value_estimator = EpigraphGAE(value_network=self.critic_network, **hp)
        elif value_type == EpigraphValueEstimators.LAGRANGIAN_GAE or value_type == "LagrangianGAE":
            self._value_estimator = LagrangianGAE(value_network=self.critic_network, **hp)
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

