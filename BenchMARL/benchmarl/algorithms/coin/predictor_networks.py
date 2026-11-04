from __future__ import annotations

"""COIN predictor networks for intrinsic reward computation.

These networks predict observations/states for curiosity and influence-based intrinsic rewards.
Architectures now mirror the TDD SEncoder/PotentialNet style (residual blocks + orthogonal init)
for a fair comparison with the EFXPLORER TDD module.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_orthogonal_(linear: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(linear.weight, gain)
    if linear.bias is not None:
        nn.init.constant_(linear.bias, 0.0)


class PredictNetworkOS(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        hidden_dim: int = 64,
        obs_shape: int | None = None,
        state_shape: int | None = None,
        n_agents: int | None = None,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.latent_dim = hidden_dim
        self.num_blocks = num_blocks
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.n_agents = n_agents
        
        self.fc_in = nn.Linear(num_inputs, self.latent_dim)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.latent_dim, self.latent_dim),
                        nn.Linear(self.latent_dim, self.latent_dim),
                    ]
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.fc_final = nn.Linear(self.latent_dim, self.latent_dim)
        
        self.last_fc_obs = (
            nn.Linear(self.latent_dim, obs_shape * n_agents)
            if obs_shape is not None and n_agents is not None
            else None
        )
        self.last_fc_state = (
            nn.Linear(self.latent_dim, state_shape) if state_shape is not None else None
        )
        
        _init_orthogonal_(self.fc_in, math.sqrt(2))
        for block in self.blocks:
            _init_orthogonal_(block[0], math.sqrt(2))
            _init_orthogonal_(block[1], math.sqrt(2))
        _init_orthogonal_(self.fc_final, math.sqrt(2))
        if self.last_fc_obs is not None:
            _init_orthogonal_(self.last_fc_obs, 1.0)
        if self.last_fc_state is not None:
            _init_orthogonal_(self.last_fc_state, 1.0)
    
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        x = F.relu(self.fc_in(input))
        for block in self.blocks:
            identity = x
            x = F.relu(block[0](x))
            x = F.relu(block[1](x))
            x = x + identity
        x = F.relu(self.fc_final(x))
        
        next_obs = self.last_fc_obs(x) if self.last_fc_obs is not None else None
        next_state = self.last_fc_state(x) if self.last_fc_state is not None else None
        return next_obs, next_state
    
    def update(
        self,
        own_variable: torch.Tensor,
        other_obs: torch.Tensor,
        other_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() > 0:
            predict_obs, predict_state = self.forward(own_variable)
            
            obs_loss = F.mse_loss(predict_obs, other_obs, reduction="none")
            obs_loss = obs_loss.sum(dim=-1, keepdim=True)
            
            state_loss = F.mse_loss(predict_state, other_state, reduction="none")
            state_loss = state_loss.sum(dim=-1, keepdim=True)
            
            total_loss = (obs_loss + state_loss) * mask
            loss = total_loss.sum() / mask.sum()
            return loss
        return torch.tensor(0.0, device=own_variable.device)


class PredictNetwork(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        hidden_dim: int = 64,
        output_shape: int | None = None,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.latent_dim = hidden_dim
        self.num_blocks = num_blocks
        self.output_shape = output_shape
        
        self.fc_in = nn.Linear(num_inputs, self.latent_dim)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.latent_dim, self.latent_dim),
                        nn.Linear(self.latent_dim, self.latent_dim),
                    ]
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.fc_final = nn.Linear(self.latent_dim, self.latent_dim)
        self.last_fc = nn.Linear(self.latent_dim, output_shape) if output_shape is not None else None
        
        _init_orthogonal_(self.fc_in, math.sqrt(2))
        for block in self.blocks:
            _init_orthogonal_(block[0], math.sqrt(2))
            _init_orthogonal_(block[1], math.sqrt(2))
        _init_orthogonal_(self.fc_final, math.sqrt(2))
        if self.last_fc is not None:
            _init_orthogonal_(self.last_fc, 1.0)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_in(input))
        for block in self.blocks:
            identity = x
            x = F.relu(block[0](x))
            x = F.relu(block[1](x))
            x = x + identity
        x = F.relu(self.fc_final(x))
        output = self.last_fc(x) if self.last_fc is not None else x
        return output
    
    def get_log_pi(self, own_variable: torch.Tensor, other_variable: torch.Tensor) -> torch.Tensor:
        predict_variable = self.forward(own_variable)
        log_prob = -1 * F.mse_loss(
            predict_variable,
            other_variable,
            reduction="none"
        )
        log_prob = torch.sum(log_prob, dim=-1, keepdim=False)
        return log_prob
    
    def update(
        self,
        own_variable: torch.Tensor,
        other_variable: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            loss = F.mse_loss(
                predict_variable,
                other_variable,
                reduction="none"
            )
            loss = loss.sum(dim=-1, keepdim=False)
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()
            return loss
        return torch.tensor(0.0, device=own_variable.device)

