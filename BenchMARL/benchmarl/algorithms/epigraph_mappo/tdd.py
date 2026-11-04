from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



def mrn_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(
            f"MRN distance expects matching last dimensions, got "
            f"{x.shape[-1]} and {y.shape[-1]}"
        )
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"MRN distance requires even feature size, got {d}")

    x_prefix, x_suffix = x[..., : d // 2], x[..., d // 2 :]
    y_prefix, y_suffix = y[..., : d // 2], y[..., d // 2 :]

    max_component = torch.clamp(x_prefix - y_prefix, min=0.0).amax(dim=-1)
    l2_component = torch.sqrt(((x_suffix - y_suffix) ** 2).sum(dim=-1) + eps)
    return max_component + l2_component


def _init_orthogonal_(linear: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(linear.weight, gain)
    if linear.bias is not None:
        nn.init.constant_(linear.bias, 0.0)


class PotentialNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        latent_dim: int = 64,
        num_blocks: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks

        self.fc_in = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(latent_dim, latent_dim),
                        nn.Linear(latent_dim, latent_dim),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

        self.fc_final = nn.Linear(latent_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, 1)

        _init_orthogonal_(self.fc_in, math.sqrt(2))
        for block in self.blocks:
            _init_orthogonal_(block[0], math.sqrt(2))
            _init_orthogonal_(block[1], math.sqrt(2))
        _init_orthogonal_(self.fc_final, math.sqrt(2))
        _init_orthogonal_(self.fc_out, 1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_in(obs))
        for block in self.blocks:
            identity = x
            x = F.relu(block[0](x))
            x = F.relu(block[1](x))
            x = x + identity

        x = F.relu(self.fc_final(x))
        value = self.fc_out(x)
        return value


class SEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        latent_dim: int = 64,
        output_dim: int = 64,
        num_blocks: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        self.fc_in = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(latent_dim, latent_dim),
                        nn.Linear(latent_dim, latent_dim),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

        self.fc_final = nn.Linear(latent_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, output_dim)
        _init_orthogonal_(self.fc_in, math.sqrt(2))
        for block in self.blocks:
            _init_orthogonal_(block[0], math.sqrt(2))
            _init_orthogonal_(block[1], math.sqrt(2))
        _init_orthogonal_(self.fc_final, math.sqrt(2))
        _init_orthogonal_(self.fc_out, 1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_in(obs))
        for block in self.blocks:
            identity = x
            x = F.relu(block[0](x))
            x = F.relu(block[1](x))
            x = x + identity
        x = F.relu(self.fc_final(x))
        encoded = self.fc_out(x)
        return encoded


@torch.no_grad()
def compute_successor_distance_grid(
    agent_pos: torch.Tensor,
    agent_vel: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    potential_net: PotentialNet,
    s_encoder: SEncoder,
) -> torch.Tensor:
    device = agent_pos.device
    grid_x = grid_x.to(device)
    grid_y = grid_y.to(device)

    agent_features = torch.cat([agent_vel, agent_pos], dim=-1)

    grid_shape = grid_x.shape
    grid_x_flat = grid_x.reshape(-1, 1)
    grid_y_flat = grid_y.reshape(-1, 1)
    goal_vels = torch.zeros(
        (grid_x_flat.shape[0], 2), dtype=grid_x.dtype, device=device
    )
    goal_features = torch.cat([goal_vels, grid_x_flat, grid_y_flat], dim=-1)

    phi_s = s_encoder(agent_features.unsqueeze(0))
    phi_g = s_encoder(goal_features)

    dists = mrn_distance(phi_s.unsqueeze(1), phi_g.unsqueeze(0)).squeeze(0)
    return dists.reshape(grid_shape)


def tdd_contrastive_loss(
    encoder: SEncoder,
    potential_net: PotentialNet,
    obs: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    phi_s = encoder(obs)
    phi_g = encoder(goal)
    c_g = potential_net(goal).squeeze(-1)
    logits = c_g.unsqueeze(0) - mrn_distance(phi_s.unsqueeze(1), phi_g.unsqueeze(0))
    log_softmax = logits.log_softmax(dim=-1)
    identity = torch.eye(logits.size(-1), device=logits.device)
    loss1 = -(identity * log_softmax).sum(dim=-1).mean()
    loss2 = -(identity * log_softmax.transpose(0, 1)).sum(dim=-1).mean()
    return 0.5 * (loss1 + loss2)


def discounted_sampling(
    ranges: torch.Tensor, discount: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    assert 0 <= discount <= 1

    seeds = torch.rand(ranges.shape, generator=generator, device=ranges.device)

    if discount == 0:
        samples = torch.zeros_like(seeds, dtype=torch.int32)
    elif discount == 1:
        samples = torch.floor(seeds * ranges.float()).int()
    else:
        samples = torch.log(1 - (1 - discount**ranges.float()) * seeds) / math.log(discount)
        samples = torch.minimum(torch.floor(samples).int(), ranges - 1)

    return samples


@torch.no_grad()
def compute_successor_distance(
    agent_features_curr: torch.Tensor,
    agent_features_next: torch.Tensor,
    encoder: SEncoder,
    dones: torch.Tensor | None = None,
) -> torch.Tensor:
    num_steps, num_envs, feat_dim = agent_features_curr.shape
    device = agent_features_curr.device
    dtype = agent_features_curr.dtype

    phi_curr_all = encoder(agent_features_curr.reshape(-1, feat_dim)).reshape(
        num_steps, num_envs, -1
    )
    phi_next_all = encoder(agent_features_next.reshape(-1, feat_dim)).reshape(
        num_steps, num_envs, -1
    )
    
    output_dim = phi_curr_all.shape[-1]

    if dones is None:
        k_indices = torch.arange(num_steps, device=device).unsqueeze(1)
        t_indices = torch.arange(num_steps, device=device).unsqueeze(0)
        triangular_mask = k_indices <= t_indices
        
        triangular_mask = triangular_mask.unsqueeze(-1).expand(-1, -1, num_envs)
        
        phi_curr_expanded = phi_curr_all.unsqueeze(1)
        phi_next_expanded = phi_next_all.unsqueeze(0)
        
        all_dists = mrn_distance(phi_curr_expanded, phi_next_expanded)
        
        masked_dists = all_dists.masked_fill(~triangular_mask, float('inf'))
        
        dists = masked_dists.min(dim=0)[0]
        
        return dists

    if dones.ndim == 3:
        dones = dones.squeeze(-1)
    if dones.shape != (num_steps, num_envs):
        raise ValueError(f"dones must have shape (T, env), got {dones.shape}")

    dones_shift = torch.zeros_like(dones)
    if num_steps > 1:
        dones_shift[1:] = dones[:-1]
    episode_id = dones_shift.cumsum(dim=0)

    k_indices = torch.arange(num_steps, device=device).unsqueeze(1).unsqueeze(2)
    t_indices = torch.arange(num_steps, device=device).unsqueeze(0).unsqueeze(2)
    triangular_mask = k_indices <= t_indices
    triangular_mask = triangular_mask.expand(-1, -1, num_envs)
    
    ep_id_k = episode_id.unsqueeze(1)
    ep_id_t = episode_id.unsqueeze(0)
    episode_mask = ep_id_k == ep_id_t
    
    valid_mask = triangular_mask & episode_mask

    phi_curr_expanded = phi_curr_all.unsqueeze(1)
    phi_next_expanded = phi_next_all.unsqueeze(0)
    
    all_dists = mrn_distance(phi_curr_expanded, phi_next_expanded)
    
    masked_dists = all_dists.masked_fill(~valid_mask, float('inf'))
    
    dists = masked_dists.min(dim=0)[0]
    
    dists = torch.where(torch.isinf(dists), torch.zeros_like(dists), dists)

    return dists


class IntrinsicManager:
    def __init__(self, algorithm) -> None:
        self._algo = algorithm

    def compute_intrinsic_rewards(
        self,
        group: str,
        features_curr: torch.Tensor,
        features_next: torch.Tensor,
        actions: torch.Tensor,
        done: torch.Tensor,
        update_networks: bool = True,
        logits: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self._algo.experiment.algorithm_config
        num_agents = len(self._algo.group_map[group])
        T, num_envs = features_curr.shape[0], features_curr.shape[1]

        if update_networks:
            tdd_metrics = self._update_tdd_networks(group, features_curr, T, num_envs)
        else:
            tdd_metrics = {"tdd_loss": 0.0}

        intrinsic_rewards = torch.zeros(
            T, num_envs, num_agents, device=features_curr.device
        )
        
        joint_mode = getattr(self._algo, 'joint_intrinsic_mode', 'none')
        
        if joint_mode == "team":
            team_feats_curr = features_curr.reshape(T, num_envs, -1)
            team_feats_next = features_next.reshape(T, num_envs, -1)
            
            encoder = self._algo._get_tdd_encoder(group, 0)
            
            dones_team = done[:, :, 0] if done.ndim >= 3 else done
            
            team_sd = compute_successor_distance(
                team_feats_curr, team_feats_next, encoder, dones=dones_team
            )
            team_sd = torch.clamp(team_sd, min=0.0)
            
            for agent_idx in range(num_agents):
                intrinsic_rewards[:, :, agent_idx] = team_sd
        else:
            for agent_idx in range(num_agents):
                encoder = self._algo._get_tdd_encoder(group, agent_idx)

                if joint_mode == "individual":
                    own_curr = features_curr[:, :, agent_idx, :]
                    own_next = features_next[:, :, agent_idx, :]
                    other_indices = [j for j in range(num_agents) if j != agent_idx]
                    if other_indices:
                        others_curr = features_curr[:, :, other_indices, :]
                        others_next = features_next[:, :, other_indices, :]
                        others_curr_flat = others_curr.reshape(*others_curr.shape[:2], -1)
                        others_next_flat = others_next.reshape(*others_next.shape[:2], -1)
                        feats_curr_agent = torch.cat([own_curr, others_curr_flat], dim=-1)
                        feats_next_agent = torch.cat([own_next, others_next_flat], dim=-1)
                    else:
                        feats_curr_agent = own_curr
                        feats_next_agent = own_next
                else:
                    feats_curr_agent = features_curr[:, :, agent_idx, :]
                    feats_next_agent = features_next[:, :, agent_idx, :]

                if done.ndim == 4:
                    dones_agent = done[:, :, agent_idx, :].squeeze(-1)
                elif done.ndim == 3:
                    dones_agent = done[:, :, agent_idx]
                else:
                    dones_agent = done

                sd = compute_successor_distance(
                    feats_curr_agent, feats_next_agent, encoder, dones=dones_agent
                )
                sd = torch.clamp(sd, min=0.0)
                intrinsic_rewards[:, :, agent_idx] = sd

        return intrinsic_rewards, tdd_metrics

    def _update_tdd_networks(
        self, group: str, features: torch.Tensor, T: int, num_envs: int
    ) -> Dict[str, float]:
        cfg = self._algo.experiment.algorithm_config
        if not cfg.use_tdd:
            return {"tdd_loss": 0.0}

        num_agents = len(self._algo.group_map[group])
        batch_size = cfg.tdd_batch_size
        learning_epochs = getattr(cfg, 'tdd_learning_epochs', 1)
        tdd_losses = []

        joint_mode = getattr(self._algo, 'joint_intrinsic_mode', 'none')
        
        for _ in range(learning_epochs):
            if joint_mode == "team":
                encoder = self._algo._get_tdd_encoder(group, 0)
                potential = self._algo._get_tdd_potential(group, 0)
                tdd_opt = self._algo._get_tdd_optimizer(group, 0)
                
                team_features = features.reshape(T, num_envs, -1)
                
                traj_idx = torch.randint(0, num_envs, (batch_size,), device=features.device)
                step_idx = torch.randint(0, T, (batch_size,), device=features.device)
                
                obs = team_features[step_idx, traj_idx, :]
                
                max_intervals = (T - step_idx).float()
                intervals = discounted_sampling(max_intervals, cfg.tdd_discount)
                goal_idx = torch.clamp(step_idx + intervals.long(), max=T - 1)
                goal = team_features[goal_idx, traj_idx, :]
                
                loss = tdd_contrastive_loss(encoder, potential, obs, goal)
                
                tdd_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(potential.parameters()), 10.0
                )
                tdd_opt.step()
                tdd_losses.append(loss.item())
            else:
                for agent_idx in range(num_agents):
                    encoder = self._algo._get_tdd_encoder(group, agent_idx)
                    potential = self._algo._get_tdd_potential(group, agent_idx)
                    tdd_opt = self._algo._get_tdd_optimizer(group, agent_idx)

                    traj_idx = torch.randint(0, num_envs, (batch_size,), device=features.device)
                    step_idx = torch.randint(0, T, (batch_size,), device=features.device)

                    if joint_mode == "individual":
                        own_obs = features[step_idx, traj_idx, agent_idx, :]
                        other_indices = [j for j in range(num_agents) if j != agent_idx]
                        if other_indices:
                            others_obs = features[step_idx, traj_idx][:, other_indices, :]
                            others_obs_flat = others_obs.reshape(others_obs.shape[0], -1)
                            obs = torch.cat([own_obs, others_obs_flat], dim=-1)
                        else:
                            obs = own_obs
                    else:
                        obs = features[step_idx, traj_idx, agent_idx, :]

                    max_intervals = (T - step_idx).float()
                    intervals = discounted_sampling(max_intervals, cfg.tdd_discount)
                    goal_idx = torch.clamp(step_idx + intervals.long(), max=T - 1)

                    if joint_mode == "individual":
                        own_goal = features[goal_idx, traj_idx, agent_idx, :]
                        if other_indices:
                            others_goal = features[goal_idx, traj_idx][:, other_indices, :]
                            others_goal_flat = others_goal.reshape(others_goal.shape[0], -1)
                            goal = torch.cat([own_goal, others_goal_flat], dim=-1)
                        else:
                            goal = own_goal
                    else:
                        goal = features[goal_idx, traj_idx, agent_idx, :]

                    loss = tdd_contrastive_loss(encoder, potential, obs, goal)

                    tdd_opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(potential.parameters()), 10.0
                    )
                    tdd_opt.step()
                    tdd_losses.append(loss.item())

        return {"tdd_loss": float(sum(tdd_losses) / max(len(tdd_losses), 1))}

