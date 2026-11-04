from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.ippo import Ippo, IppoConfig

from benchmarl.algorithms.epigraph_mappo.tdd import (
    SEncoder,
    PotentialNet,
    IntrinsicManager,
)

from benchmarl.algorithms.epigraph_mappo.torchrl_overrides import (
    EpigraphGAE,
    EpigraphValueEstimators,
)
from benchmarl.algorithms.epigraph_mappo.loss import JointClipPPOLoss


class EfxplorerIppo(Ippo):
    """Independent PPO with TDD intrinsic rewards and EpigraphGAE."""
    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: float,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        minibatch_advantage: bool,
        **kwargs
    ):
        efxplorer_params = {
            'intrinsic_gamma',
            'intrinsic_reward_type',
            'intrinsic_reward_scale',
            'extrinsic_only',
            'intrinsic_only',
            'use_tdd',
            'tdd_latent_dim',
            'tdd_output_dim',
            'tdd_learning_rate',
            'tdd_discount',
            'tdd_batch_size',
            'tdd_learning_epochs',
            'z_ema_alpha',
            'z0_best_reward_decay',
            'warmup_rollouts',
            'adv_clip_abs',
            'use_soft_min',
            'soft_min_tau',
            'agent_feature_up_to_velocity',
            'critic_int_lr',
            'critic_lr',
            'joint_intrinsic_mode',
            'ablation_current',
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in efxplorer_params}

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

        self.intrinsic_gamma = getattr(cfg, 'intrinsic_gamma', 0.99)
        self.intrinsic_reward_scale = getattr(cfg, 'intrinsic_reward_scale', 0.1)
        self.extrinsic_only = getattr(cfg, 'extrinsic_only', False)
        self.intrinsic_only = getattr(cfg, 'intrinsic_only', False)
        
        self.z_ema_alpha = getattr(cfg, 'z_ema_alpha', 0.1)
        self.z0_best_reward_decay = getattr(cfg, 'z0_best_reward_decay', 0.99)
        self.warmup_rollouts = getattr(cfg, 'warmup_rollouts', 0)
        self.adv_clip_abs = getattr(cfg, 'adv_clip_abs', 5.0)
        
        self.agent_feature_up_to_velocity = getattr(cfg, 'agent_feature_up_to_velocity', False)
        
        self.joint_intrinsic_mode = getattr(cfg, 'joint_intrinsic_mode', 'none')
        self.ablation_current = getattr(cfg, 'ablation_current', False)
        
        self.use_soft_min = getattr(cfg, 'use_soft_min', False)
        self.soft_min_tau = getattr(cfg, 'soft_min_tau', 0.1)
        
        self.update_steps = 0
        
        self.tdd_encoders: Dict[str, List[SEncoder]] = {}
        self.tdd_potentials: Dict[str, List[PotentialNet]] = {}
        self.tdd_optimizers: Dict[str, List[optim.Optimizer]] = {}
        
        self.intrinsic_value_modules: Dict[str, TensorDictModule] = {}
        
        self.loss_modules: Dict[str, JointClipPPOLoss] = {}
        
        self._intrinsic_metrics: Dict[str, dict] = {}
        
        self._build_tdd_networks()
        self._build_intrinsic_value_networks()
        
        self.intrinsic_manager = IntrinsicManager(self)

    def _build_tdd_networks(self) -> None:
        cfg = self.experiment.algorithm_config
        
        if self.extrinsic_only:
            for group in self.group_map.keys():
                self.tdd_encoders[group] = []
                self.tdd_potentials[group] = []
                self.tdd_optimizers[group] = []
            return
        
        for group, agents in self.group_map.items():
            encoder_list = []
            potential_list = []
            optim_list = []
            
            base_feat_dim = 4 if self.agent_feature_up_to_velocity else 2
            num_agents_in_group = len(agents)
            if self.joint_intrinsic_mode == "individual":
                tdd_input_dim = base_feat_dim * num_agents_in_group
            elif self.joint_intrinsic_mode == "team":
                tdd_input_dim = base_feat_dim * num_agents_in_group
            else:
                tdd_input_dim = base_feat_dim
            
            num_networks = 1 if self.joint_intrinsic_mode == "team" else len(agents)
            for _ in range(num_networks):
                encoder = SEncoder(
                    input_dim=tdd_input_dim,
                    latent_dim=cfg.tdd_latent_dim,
                    output_dim=cfg.tdd_output_dim,
                ).to(self.device)
                
                potential = PotentialNet(
                    input_dim=tdd_input_dim,
                    latent_dim=cfg.tdd_latent_dim,
                ).to(self.device)
                
                optimizer = optim.Adam(
                    list(encoder.parameters()) + list(potential.parameters()),
                    lr=cfg.tdd_learning_rate,
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

    def _build_intrinsic_value_networks(self) -> None:
        for group, agents in self.group_map.items():
            obs_dim = self.observation_spec[group, "observation"].shape[-1]
            num_agents_in_group = len(agents)
            
            hidden_sizes = [256, 256]
            if hasattr(self.critic_model_config, "num_cells"):
                hidden_sizes = list(self.critic_model_config.num_cells)
            
            if self.joint_intrinsic_mode == "team":
                state_dim = num_agents_in_group * obs_dim
                
                class TeamIntValueNet(nn.Module):
                    def __init__(self, state_dim, z_dim, hidden_sizes, n_agents):
                        super().__init__()
                        self.n_agents = n_agents
                        layers = []
                        in_dim = state_dim + z_dim
                        for h_dim in hidden_sizes:
                            layers.append(nn.Linear(in_dim, h_dim))
                            layers.append(nn.ReLU())
                            in_dim = h_dim
                        layers.append(nn.Linear(in_dim, 1))
                        self.net = nn.Sequential(*layers)
                    
                    def forward(self, obs: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
                        batch_shape = obs.shape[:-2]
                        full_state = obs.reshape(*batch_shape, -1)
                        
                        if z_i.ndim == obs.ndim - 1:
                            z_team = z_i[..., 0:1]
                        elif z_i.ndim == obs.ndim and z_i.shape[-1] == 1:
                            z_team = z_i[..., 0, :]
                        else:
                            z_team = z_i[..., 0:1]
                        
                        x = torch.cat([full_state, z_team], dim=-1)
                        flat = x.reshape(-1, x.shape[-1])
                        out = self.net(flat)
                        team_val = out.view(*batch_shape, 1)
                        return team_val.unsqueeze(-2).expand(*batch_shape, self.n_agents, 1)
                
                value_net = TeamIntValueNet(
                    state_dim=state_dim, z_dim=1,
                    hidden_sizes=hidden_sizes, n_agents=num_agents_in_group,
                ).to(self.device)
            else:
                class AgentIntValueNet(nn.Module):
                    def __init__(self, obs_dim: int, z_dim: int, hidden_sizes: List[int]):
                        super().__init__()
                        layers = []
                        in_dim = obs_dim + z_dim
                        for h_dim in hidden_sizes:
                            layers.append(nn.Linear(in_dim, h_dim))
                            layers.append(nn.ReLU())
                            in_dim = h_dim
                        layers.append(nn.Linear(in_dim, 1))
                        self.net = nn.Sequential(*layers)
                    
                    def forward(self, obs: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
                        if z_i.ndim == obs.ndim - 1:
                            z_i_exp = z_i.unsqueeze(-1)
                        elif z_i.ndim == obs.ndim and z_i.shape[-1] == 1:
                            z_i_exp = z_i
                        else:
                            raise ValueError(
                                f"Unexpected z_i shape {z_i.shape} for obs shape {obs.shape}"
                            )
                        x = torch.cat([obs, z_i_exp], dim=-1)
                        flat = x.reshape(-1, x.shape[-1])
                        out = self.net(flat)
                        return out.view(*x.shape[:-1], 1)
                
                value_net = AgentIntValueNet(
                    obs_dim=obs_dim, z_dim=1, hidden_sizes=hidden_sizes
                ).to(self.device)
            
            value_module = TensorDictModule(
                value_net,
                in_keys=[(group, "observation"), (group, "z_i")],
                out_keys=[(group, "state_value_int")],
            )
            
            self.intrinsic_value_modules[group] = value_module

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        cfg = self.experiment.algorithm_config
        
        intrinsic_value_net = self.intrinsic_value_modules.get(group)
        
        loss_module = JointClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            z_key=(group, "z_i"),
            intrinsic_key=(group, "intrinsic_reward"),
            intrinsic_only=self.intrinsic_only,
            intrinsic_gamma=self.intrinsic_gamma if self.intrinsic_only else None,
            intrinsic_value_network=intrinsic_value_net,
        )
        
        gamma_to_use = self.intrinsic_gamma if self.intrinsic_only else self.experiment_config.gamma
        loss_module.gamma = gamma_to_use
        
        if self.extrinsic_only or self.intrinsic_only:
            loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=gamma_to_use,
                lmbda=self.lmbda,
            )
        else:
            from benchmarl.algorithms.epigraph_mappo.torchrl_overrides import EpigraphValueEstimators
            
            epigraph_params = {
                "intrinsic_gamma": self.intrinsic_gamma,
                "intrinsic_reward_key": (group, "intrinsic_reward"),
                "intrinsic_value_network": intrinsic_value_net,
                "z_ema_alpha": self.z_ema_alpha,
                "z0_best_reward_decay": self.z0_best_reward_decay,
                "adv_clip_abs": self.adv_clip_abs,
                "ablation_current": self.ablation_current,
                "use_soft_min": self.use_soft_min,
                "soft_min_tau": self.soft_min_tau,
            }
            
            loss_module.make_value_estimator(
                EpigraphValueEstimators.EPIGRAPH_GAE,
                gamma=gamma_to_use,
                lmbda=self.lmbda,
                **epigraph_params,
            )
        
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        
        self.loss_modules[group] = loss_module
        
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        params = super()._get_parameters(group, loss)
        
        if group in self.intrinsic_value_modules:
            intrinsic_net = self.intrinsic_value_modules[group]
            params["loss_critic_int"] = list(intrinsic_net.parameters())
        
        return params

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
        
        if self.extrinsic_only:
            intrinsic_rewards = torch.zeros(T, num_envs, num_agents, device=self.device)
            intrinsic_metrics = {}
        else:
            feature_dim = 4 if self.agent_feature_up_to_velocity else 2
            agent_features_curr = obs[..., :feature_dim]
            agent_features_next = next_obs[..., :feature_dim]
            
            intrinsic_rewards, intrinsic_metrics = self.intrinsic_manager.compute_intrinsic_rewards(
                group, agent_features_curr, agent_features_next, actions, done
            )
        
        self._intrinsic_metrics[group] = intrinsic_metrics
        
        if self.intrinsic_reward_scale != 1.0:
            intrinsic_rewards = intrinsic_rewards * self.intrinsic_reward_scale
        
        intrinsic_for_batch = intrinsic_rewards
        if batch_format == "env_first":
            intrinsic_for_batch = intrinsic_rewards.transpose(0, 1)
        batch.set((group, "intrinsic_reward"), intrinsic_for_batch.unsqueeze(-1))
        
        if self.intrinsic_only:
            batch.set((group, "reward"), intrinsic_for_batch.unsqueeze(-1))
            batch.set(("next", group, "reward"), intrinsic_for_batch.unsqueeze(-1))
            if ("next", "reward") in batch.keys(True, True):
                if intrinsic_for_batch.ndim == 3:
                    global_intrinsic = intrinsic_for_batch.mean(dim=-1, keepdim=True)
                else:
                    global_intrinsic = intrinsic_for_batch
                batch.set(("next", "reward"), global_intrinsic)
        
        if group in self.loss_modules:
            loss_module = self.loss_modules[group]
            if hasattr(loss_module, "_value_estimator") and hasattr(loss_module._value_estimator, "set_warmup_mode"):
                loss_module._value_estimator.set_warmup_mode(self.update_steps < self.warmup_rollouts)
        
        self._store_rollout_for_z0_update(group, batch, intrinsic_rewards, extrinsic_rewards, done)
        
        batch = super().process_batch(group, batch)
        
        if not (self.extrinsic_only or self.intrinsic_only):
            z_i_key = (group, "z_i")
            if z_i_key not in batch.keys(True, True):
                raise ValueError(
                    "Expected z_i to be set by EpigraphGAE but it is missing in batch."
                )
        
        if group == list(self.group_map.keys())[0]:
            self.update_steps += 1
        
        return batch

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

    def _store_rollout_for_z0_update(
        self,
        group: str,
        batch: TensorDictBase,
        intrinsic_rewards: torch.Tensor,
        extrinsic_rewards: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        if not hasattr(self, '_pending_z0_updates'):
            self._pending_z0_updates = {}
        
        self._pending_z0_updates[group] = {
            'intrinsic_rewards': intrinsic_rewards.detach().cpu(),
            'extrinsic_rewards': extrinsic_rewards.detach().cpu() if extrinsic_rewards is not None else None,
            'done': done.detach().cpu(),
            'shapes': {
                'intrinsic': intrinsic_rewards.shape,
                'extrinsic': extrinsic_rewards.shape if extrinsic_rewards is not None else None,
                'done': done.shape
            },
            'device': self.device
        }

    def update_z0_from_pending_rollouts(self) -> None:
        if not hasattr(self, '_pending_z0_updates'):
            return
        
        for group, rollout_data in self._pending_z0_updates.items():
            self._update_z0_from_stored_rollout(group, rollout_data)
        
        self._pending_z0_updates.clear()

    def _update_z0_from_stored_rollout(self, group: str, rollout_data: dict) -> None:
        if group not in self.loss_modules:
            return
        
        loss_module = self.loss_modules[group]
        if not hasattr(loss_module, '_value_estimator'):
            return
        
        value_estimator = loss_module._value_estimator
        if not hasattr(value_estimator, '_update_z0_from_episode_data'):
            return
        
        device = rollout_data['device']
        intrinsic_rewards = rollout_data['intrinsic_rewards'].to(device)
        extrinsic_rewards = rollout_data['extrinsic_rewards']
        if extrinsic_rewards is not None:
            extrinsic_rewards = extrinsic_rewards.to(device)
        else:
            return
        done = rollout_data['done'].to(device)
        
        expected_shapes = rollout_data['shapes']
        if done.shape != expected_shapes['done']:
            if len(done.shape) == 2 and len(expected_shapes['done']) == 3:
                T, num_envs = done.shape
                num_agents = expected_shapes['done'][2]
                done = done.unsqueeze(-1).expand(T, num_envs, num_agents)
            elif len(done.shape) == 3 and done.shape[-1] == 1:
                num_agents = expected_shapes['done'][2]
                done = done.expand(-1, -1, num_agents)
        
        T, num_envs, num_agents = intrinsic_rewards.shape
        gamma = self.intrinsic_gamma
        
        episode_ext_sums = []
        episode_z0_sums = []
        
        for env_idx in range(num_envs):
            episode_start = 0
            
            while episode_start < T:
                episode_end = episode_start
                while episode_end < T and not done[episode_end, env_idx].any():
                    episode_end += 1
                
                if episode_end > episode_start:
                    episode_len = episode_end - episode_start
                    episode_ended_cleanly = episode_end < T and done[episode_end, env_idx].any()
                    
                    if episode_ended_cleanly:
                        episode_ext_sum = extrinsic_rewards[episode_start:episode_end, env_idx, 0].sum()
                        
                        episode_z0 = torch.zeros(num_agents, device=device)
                        discount_factors = torch.pow(gamma, torch.arange(episode_len, device=device))
                        
                        for agent_idx in range(num_agents):
                            agent_intrinsic = intrinsic_rewards[episode_start:episode_end, env_idx, agent_idx]
                            episode_z0[agent_idx] = torch.sum(agent_intrinsic * discount_factors)
                        
                        episode_ext_sums.append(episode_ext_sum)
                        episode_z0_sums.append(episode_z0)
                
                episode_start = episode_end + 1
        
        if len(episode_ext_sums) > 0:
            ext_episode_sums = torch.stack(episode_ext_sums)
            z0_episode_sums = torch.stack(episode_z0_sums)
            value_estimator._update_z0_from_episode_data(ext_episode_sums, z0_episode_sums)

    def process_loss_vals(self, group: str, loss_vals: TensorDictBase) -> TensorDictBase:
        loss_vals.set(
            "loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"]
        )
        del loss_vals["loss_entropy"]
        
        if group in self.loss_modules:
            loss_module = self.loss_modules[group]
            if hasattr(loss_module, '_value_estimator') and hasattr(loss_module._value_estimator, 'get_epigraph_stats'):
                stats = loss_module._value_estimator.get_epigraph_stats()
                for key, value in stats.items():
                    loss_vals.set(key, torch.tensor(value, device=self.device))
        
        intrinsic_metrics = self._intrinsic_metrics.get(group)
        if intrinsic_metrics is not None:
            for key, value in intrinsic_metrics.items():
                loss_vals.set(f"intrinsic/{key}", torch.tensor(value, device=self.device))
        
        return loss_vals


@dataclass
class EfxplorerIppoConfig(AlgorithmConfig):
    """Configuration dataclass for EfxplorerIPPO algorithm."""
    
    share_param_critic: bool = False
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    critic_coef: float = 0.5
    loss_critic_type: str = "l2"
    lmbda: float = 0.95
    scale_mapping: str = "biased_softplus_1.0"
    use_tanh_normal: bool = True
    minibatch_advantage: bool = False
    
    critic_int_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    
    intrinsic_gamma: float = 0.99
    intrinsic_reward_type: str = "tdd"
    intrinsic_reward_scale: float = 0.1
    extrinsic_only: bool = False
    intrinsic_only: bool = False
    
    use_tdd: bool = True
    tdd_latent_dim: int = 64
    tdd_output_dim: int = 32
    tdd_learning_rate: float = 0.001
    tdd_discount: float = 0.7
    tdd_batch_size: int = 128
    tdd_learning_epochs: int = 16
    
    z_ema_alpha: float = 0.1
    z0_best_reward_decay: float = 0.99
    warmup_rollouts: int = 0
    adv_clip_abs: float = 5.0
    use_soft_min: bool = False
    soft_min_tau: float = 0.1
    
    agent_feature_up_to_velocity: bool = False
    
    joint_intrinsic_mode: str = "none"
    
    ablation_current: bool = False

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return EfxplorerIppo

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
    def has_independent_critic() -> bool:
        return True

