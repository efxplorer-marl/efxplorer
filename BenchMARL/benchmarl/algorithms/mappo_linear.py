from dataclasses import dataclass
from typing import Dict, Optional, Type
import torch
import torch.optim as optim
from tensordict import TensorDictBase

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.mappo import Mappo

from benchmarl.algorithms.epigraph_mappo.tdd import (
    SEncoder,
    PotentialNet,
    IntrinsicManager,
)


class MappoLinear(Mappo):
    """MAPPO with linear combination of extrinsic and intrinsic rewards."""
    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: bool,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        minibatch_advantage: bool,
        **kwargs
    ):
        custom_params = {
            'intrinsic_lambda',
            'intrinsic_reward_type',
            'intrinsic_reward_scale',
            'agent_feature_up_to_velocity',
            'use_tdd',
            'tdd_latent_dim',
            'tdd_output_dim',
            'tdd_learning_rate',
            'tdd_discount',
            'tdd_batch_size',
            'tdd_learning_epochs',
            'extrinsic_only',
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in custom_params}
        
        super().__init__(
            share_param_critic=share_param_critic,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            lmbda=lmbda,
            scale_mapping=scale_mapping,
            use_tanh_normal=use_tanh_normal,
            minibatch_advantage=minibatch_advantage,
            **filtered_kwargs
        )
        
        cfg = self.experiment.algorithm_config
        
        self.intrinsic_lambda = getattr(cfg, 'intrinsic_lambda', 0.1)
        self.intrinsic_reward_type = getattr(cfg, 'intrinsic_reward_type', 'tdd')
        self.intrinsic_reward_scale = getattr(cfg, 'intrinsic_reward_scale', 1.0)
        self.agent_feature_up_to_velocity = getattr(cfg, 'agent_feature_up_to_velocity', True)
        self.extrinsic_only = getattr(cfg, 'extrinsic_only', False)
        
        self.use_tdd = getattr(cfg, 'use_tdd', True)
        self.tdd_latent_dim = getattr(cfg, 'tdd_latent_dim', 64)
        self.tdd_output_dim = getattr(cfg, 'tdd_output_dim', 32)
        self.tdd_learning_rate = getattr(cfg, 'tdd_learning_rate', 1e-3)
        self.tdd_discount = getattr(cfg, 'tdd_discount', 0.99)
        self.tdd_batch_size = getattr(cfg, 'tdd_batch_size', 32)
        self.tdd_learning_epochs = getattr(cfg, 'tdd_learning_epochs', 1)
        
        self.use_inter_agent_sd = getattr(cfg, 'use_inter_agent_sd', False)
        self.include_self_in_inter_sd = getattr(cfg, 'include_self_in_inter_sd', True)
        self.inter_sd_weight = getattr(cfg, 'inter_sd_weight', 0.5)
        
        self.tdd_encoders: Dict[str, list] = {}
        self.tdd_potentials: Dict[str, list] = {}
        self.tdd_optimizers: Dict[str, list] = {}
        
        if not self.extrinsic_only and self.intrinsic_reward_type == "tdd":
            self._build_tdd_networks()
        
        self.intrinsic_manager = IntrinsicManager(self)
        
        self._intrinsic_metrics: Dict[str, dict] = {}
    
    def _build_tdd_networks(self) -> None:
        cfg = self.experiment.algorithm_config
        
        for group, agents in self.group_map.items():
            tdd_input_dim = 4 if self.agent_feature_up_to_velocity else 2
            
            encoder_list = []
            potential_list = []
            optim_list = []
            
            for _ in agents:
                encoder = SEncoder(
                    input_dim=tdd_input_dim,
                    latent_dim=self.tdd_latent_dim,
                    output_dim=self.tdd_output_dim,
                ).to(self.device)
                potential = PotentialNet(
                    input_dim=tdd_input_dim,
                    latent_dim=self.tdd_latent_dim,
                ).to(self.device)
                optimizer = optim.Adam(
                    list(encoder.parameters()) + list(potential.parameters()),
                    lr=self.tdd_learning_rate,
                )
                encoder_list.append(encoder)
                potential_list.append(potential)
                optim_list.append(optimizer)
            
            self.tdd_encoders[group] = encoder_list
            self.tdd_potentials[group] = potential_list
            self.tdd_optimizers[group] = optim_list
    
    def _get_tdd_encoder(self, group: str, agent_idx: int) -> SEncoder:
        return self.tdd_encoders[group][agent_idx]

    def _get_tdd_potential(self, group: str, agent_idx: int) -> PotentialNet:
        return self.tdd_potentials[group][agent_idx]

    def _get_tdd_optimizer(self, group: str, agent_idx: int) -> optim.Optimizer:
        return self.tdd_optimizers[group][agent_idx]

    def _normalize_done_shape(self, done: torch.Tensor, num_envs: int, num_agents: int) -> torch.Tensor:
        if done.ndim == 4:
            done = done.squeeze(-1)
        elif done.ndim == 3:
            if done.shape[-1] == 1:
                done = done.squeeze(-1)
            elif done.shape[-1] == num_agents:
                pass
            else:
                raise ValueError(f"Unexpected done shape: {done.shape}")
        elif done.ndim == 2:
            done = done.unsqueeze(-1).expand(-1, -1, num_agents)
        else:
            raise ValueError(f"Unexpected done ndim: {done.ndim}, shape: {done.shape}")
        return done
    
    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        cfg = self.experiment.algorithm_config
        
        group_data = batch.get(group)
        
        if len(group_data.shape) >= 3 and group_data.shape[-1] == len(self.group_map[group]):
            num_envs, T = group_data.shape[0], group_data.shape[1]
            batch_format = "env_first"
        else:
            T, num_envs = group_data.shape[0], group_data.shape[1]
            batch_format = "time_first"
        
        num_agents = len(self.group_map[group])
        
        obs = batch.get((group, "observation"))
        next_obs = batch.get(("next", group, "observation"))
        
        if batch_format == "env_first":
            obs = obs.transpose(0, 1)
            next_obs = next_obs.transpose(0, 1)
        
        actions = batch.get((group, "action"))
        done = batch.get(("next", group, "done"))
        if done is None:
            done = batch.get((group, "done"))
        if done is None:
            done = batch.get(("next", "done"))
        
        if done is not None and batch_format == "env_first":
            done = done.transpose(0, 1)
        
        if done is not None:
            done = self._normalize_done_shape(done, num_envs, num_agents)
        else:
            done = torch.zeros(T, num_envs, num_agents, device=self.device, dtype=torch.bool)
        
        extrinsic_rewards = batch.get((group, "reward"))
        if extrinsic_rewards is None:
            extrinsic_rewards = batch.get(("next", group, "reward"))
        
        if extrinsic_rewards is not None and batch_format == "env_first":
            extrinsic_rewards = extrinsic_rewards.transpose(0, 1)
        
        if extrinsic_rewards is not None:
            if extrinsic_rewards.ndim == 4:
                extrinsic_rewards = extrinsic_rewards.squeeze(-1)
            elif extrinsic_rewards.ndim == 3 and extrinsic_rewards.shape[-1] == 1:
                extrinsic_rewards = extrinsic_rewards.squeeze(-1)
        
        if extrinsic_rewards is not None:
            extrinsic_for_batch = extrinsic_rewards
            if batch_format == "env_first":
                extrinsic_for_batch = extrinsic_rewards.transpose(0, 1)
            batch.set((group, "extrinsic_reward"), extrinsic_for_batch.unsqueeze(-1))
            batch.set(("next", group, "extrinsic_reward"), extrinsic_for_batch.unsqueeze(-1))
        
        if self.extrinsic_only:
            intrinsic_rewards = torch.zeros(T, num_envs, num_agents, device=self.device)
            intrinsic_metrics = {}
        else:
            if self.intrinsic_reward_type != "tdd":
                raise ValueError(
                    f"Unsupported intrinsic_reward_type '{self.intrinsic_reward_type}'. "
                    "This implementation only supports 'tdd' (or extrinsic_only=True)."
                )
            
            feature_dim = 4 if self.agent_feature_up_to_velocity else 2
            agent_features_curr = obs[..., :feature_dim]
            agent_features_next = next_obs[..., :feature_dim]
            intrinsic_rewards, intrinsic_metrics = self.intrinsic_manager.compute_intrinsic_rewards(
                group, agent_features_curr, agent_features_next, actions, done
            )
        
        self._intrinsic_metrics[group] = intrinsic_metrics
        
        if self.intrinsic_reward_scale != 1.0:
            intrinsic_rewards = intrinsic_rewards * self.intrinsic_reward_scale
        
        intrinsic_rewards_sum = intrinsic_rewards.sum(dim=-1, keepdim=True)
        intrinsic_rewards_shared = intrinsic_rewards_sum.expand_as(intrinsic_rewards)
        
        if extrinsic_rewards is not None:
            combined_rewards = extrinsic_rewards + self.intrinsic_lambda * intrinsic_rewards_shared
        else:
            combined_rewards = self.intrinsic_lambda * intrinsic_rewards_shared
        
        if batch_format == "env_first":
            combined_rewards = combined_rewards.transpose(0, 1)
        
        batch.set((group, "reward"), combined_rewards.unsqueeze(-1))
        batch.set(("next", group, "reward"), combined_rewards.unsqueeze(-1))
        
        if ("next", "reward") in batch.keys(True, True):
            if combined_rewards.ndim == 3:
                global_reward = combined_rewards.mean(dim=-1, keepdim=True)
            else:
                global_reward = combined_rewards
            batch.set(("next", "reward"), global_reward)
        
        batch = super().process_batch(group, batch)
        
        return batch
    
    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        intrinsic_metrics = self._intrinsic_metrics.get(group, None)
        if intrinsic_metrics is not None:
            for key, value in intrinsic_metrics.items():
                loss_vals.set(f"intrinsic/{key}", torch.tensor(value, device=self.device))
        
        loss_vals.set("linear/intrinsic_lambda", torch.tensor(self.intrinsic_lambda, device=self.device))
        loss_vals.set("linear/intrinsic_reward_scale", torch.tensor(self.intrinsic_reward_scale, device=self.device))
        
        return loss_vals


@dataclass
class MappoLinearConfig(AlgorithmConfig):
    """Configuration dataclass for the MAPPO Linear algorithm."""
    
    share_param_critic: bool = True
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    critic_coef: float = 0.5
    loss_critic_type: str = "smooth_l1"
    lmbda: float = 0.95
    scale_mapping: str = "softplus"
    use_tanh_normal: bool = False
    minibatch_advantage: bool = False
    
    intrinsic_lambda: float = 0.1
    intrinsic_reward_type: str = "tdd"
    intrinsic_reward_scale: float = 1.0
    agent_feature_up_to_velocity: bool = True
    extrinsic_only: bool = False
    
    use_tdd: bool = True
    tdd_latent_dim: int = 64
    tdd_output_dim: int = 32
    tdd_learning_rate: float = 1e-3
    tdd_discount: float = 0.99
    tdd_batch_size: int = 32
    tdd_learning_epochs: int = 1
    
    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MappoLinear
    
    @staticmethod
    def supports_continuous_actions() -> bool:
        return True
    
    @staticmethod
    def supports_discrete_actions() -> bool:
        return True
    
    @staticmethod
    def on_policy() -> bool:
        return True
    
    @staticmethod
    def has_centralized_critic() -> bool:
        return True
    
    @staticmethod
    def has_independent_critic() -> bool:
        return False

