#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union
import numpy as np

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.objectives import DiscreteSACLoss, LossModule, SACLoss

from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.masac import Masac, MasacConfig


class Hasac(Masac):
    """Heterogeneous-Agent Soft Actor Critic (from `https://arxiv.org/abs/2304.09870`)."""
    def __init__(
        self,
        share_param_critic: bool,
        num_qvalue_nets: int,
        loss_function: str,
        delay_qvalue: bool,
        target_entropy: Union[float, str],
        discrete_target_entropy_weight: float,
        alpha_init: float,
        min_alpha: Optional[float],
        max_alpha: Optional[float],
        fixed_alpha: bool,
        scale_mapping: str,
        use_tanh_normal: bool,
        coupled_discrete_values: bool,
        fixed_order: bool = False,
        per_agent_alpha: bool = False,
        **kwargs,
    ):
        super().__init__(
            share_param_critic=share_param_critic,
            num_qvalue_nets=num_qvalue_nets,
            loss_function=loss_function,
            delay_qvalue=delay_qvalue,
            target_entropy=target_entropy,
            discrete_target_entropy_weight=discrete_target_entropy_weight,
            alpha_init=alpha_init,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            fixed_alpha=fixed_alpha,
            scale_mapping=scale_mapping,
            use_tanh_normal=use_tanh_normal,
            coupled_discrete_values=coupled_discrete_values,
            **kwargs,
        )
        self.fixed_order = fixed_order
        self.per_agent_alpha = per_agent_alpha
        
        self._hasac_losses: Dict[str, "HasacLoss"] = {}

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        base_loss, has_target_net = super()._get_loss(group, policy_for_loss, continuous)

        n_agents = len(self.group_map[group])
        hasac_loss = HasacLoss(
            base_sac_loss=base_loss,
            group=group,
            n_agents=n_agents,
            fixed_order=self.fixed_order,
            continuous=continuous,
            per_agent_alpha=self.per_agent_alpha,
            alpha_init=self.alpha_init,
            min_alpha=self.min_alpha,
            max_alpha=self.max_alpha,
            fixed_alpha=self.fixed_alpha,
            target_entropy=self.target_entropy,
            algorithm=self,
        )
        
        self._hasac_losses[group] = hasac_loss

        return hasac_loss, has_target_net

    def _get_parameters(self, group: str, loss: "HasacLoss") -> Dict[str, Iterable]:
        base_loss = loss.base_loss
        items = {
            "loss_actor": list(base_loss.actor_network_params.flatten_keys().values()),
            "loss_qvalue": list(base_loss.qvalue_network_params.flatten_keys().values()),
        }
        if not self.fixed_alpha:
            if self.per_agent_alpha and loss.log_alphas is not None:
                items.update({"loss_alpha": list(loss.log_alphas)})
            else:
                items.update({"loss_alpha": [base_loss.log_alpha]})
        return items
    
    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        if "loss_actor" in loss_vals.keys():
            del loss_vals["loss_actor"]
        if "loss_alpha" in loss_vals.keys():
            del loss_vals["loss_alpha"]
        return loss_vals


class HasacLoss(LossModule):
    def __init__(
        self,
        base_sac_loss: Union[SACLoss, DiscreteSACLoss],
        group: str,
        n_agents: int,
        fixed_order: bool = False,
        continuous: bool = True,
        per_agent_alpha: bool = False,
        alpha_init: float = 1.0,
        min_alpha: Optional[float] = None,
        max_alpha: Optional[float] = None,
        fixed_alpha: bool = False,
        target_entropy: Union[float, str] = "auto",
        algorithm: Optional["Hasac"] = None,
    ):
        super().__init__()
        self.base_loss = base_sac_loss
        self.group = group
        self.n_agents = n_agents
        self.fixed_order = fixed_order
        self.continuous = continuous
        self.per_agent_alpha = per_agent_alpha
        self.fixed_alpha = fixed_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self._target_entropy = target_entropy
        self._algorithm = algorithm

        if per_agent_alpha and not fixed_alpha:
            log_alpha_init = torch.log(torch.tensor(alpha_init, dtype=torch.float32))
            self.log_alphas = nn.ParameterList([
                nn.Parameter(log_alpha_init.clone()) for _ in range(n_agents)
            ])
        else:
            self.log_alphas = None

        if hasattr(base_sac_loss, 'target_qvalue_network_params'):
            self.target_qvalue_network_params = base_sac_loss.target_qvalue_network_params
        if hasattr(base_sac_loss, 'target_actor_network_params'):
            self.target_actor_network_params = base_sac_loss.target_actor_network_params
        if hasattr(base_sac_loss, 'target_value_network_params') and base_sac_loss.target_value_network_params is not None:
            self.target_value_network_params = base_sac_loss.target_value_network_params

    @property
    def in_keys(self):
        return self.base_loss.in_keys

    @property
    def out_keys(self):
        return self.base_loss.out_keys

    def _get_alpha(self, agent_idx: int) -> torch.Tensor:
        if self.per_agent_alpha and self.log_alphas is not None:
            alpha = self.log_alphas[agent_idx].exp()
            if self.min_alpha is not None:
                alpha = alpha.clamp(min=self.min_alpha)
            if self.max_alpha is not None:
                alpha = alpha.clamp(max=self.max_alpha)
            return alpha
        else:
            return self.base_loss._alpha

    def _get_agent_order(self) -> List[int]:
        if self.fixed_order:
            return list(range(self.n_agents))
        else:
            return list(np.random.permutation(self.n_agents))
    
    def _get_optimizers(self) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.Optimizer]]:
        if self._algorithm is None:
            return None, None
        
        try:
            experiment = self._algorithm.experiment
            if not hasattr(experiment, 'optimizers'):
                return None, None
            
            group_optimizers = experiment.optimizers.get(self.group, {})
            actor_optimizer = group_optimizers.get("loss_actor")
            alpha_optimizer = group_optimizers.get("loss_alpha")
            return actor_optimizer, alpha_optimizer
        except Exception:
            return None, None

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        from tensordict import TensorDict
        
        loss_qvalue, qvalue_metadata = self.base_loss.qvalue_loss(tensordict)

        actor_optimizer, alpha_optimizer = self._get_optimizers()
        
        if actor_optimizer is not None:
            metrics = self._sequential_actor_train(
                tensordict, actor_optimizer, alpha_optimizer
            )
            loss_actor = torch.tensor(0.0, device=loss_qvalue.device)
            loss_alpha = torch.tensor(0.0, device=loss_qvalue.device)
            mean_log_prob = metrics.get("mean_log_prob", torch.tensor(0.0))
        else:
            loss_actor, mean_log_prob = self._accumulated_actor_loss(tensordict)
            if self.fixed_alpha:
                loss_alpha = torch.tensor(0.0, device=loss_actor.device)
            elif self.per_agent_alpha:
                loss_alpha = self._per_agent_alpha_loss(mean_log_prob)
            else:
                loss_alpha = self.base_loss.alpha_loss(mean_log_prob)

        if "td_error" in qvalue_metadata:
            tensordict.set(
                self.base_loss.tensor_keys.priority,
                qvalue_metadata["td_error"]
            )

        if loss_qvalue.dim() > 0:
            loss_qvalue = loss_qvalue.mean()
        if loss_alpha.dim() > 0:
            loss_alpha = loss_alpha.mean()

        out = {
            "loss_actor": loss_actor,
            "loss_qvalue": loss_qvalue,
            "loss_alpha": loss_alpha,
            "alpha": self._get_alpha(0),
            "entropy": -mean_log_prob.detach() if hasattr(mean_log_prob, 'detach') else torch.tensor(0.0),
        }
        
        out_td = TensorDict(out, [])
        return out_td

    def _sequential_actor_train(
        self,
        tensordict: TensorDictBase,
        actor_optimizer: torch.optim.Optimizer,
        alpha_optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, torch.Tensor]:
        agent_order = self._get_agent_order()
        
        if self.continuous:
            return self._sequential_train_continuous(
                tensordict, actor_optimizer, alpha_optimizer, agent_order
            )
        else:
            return self._sequential_train_discrete(
                tensordict, actor_optimizer, alpha_optimizer, agent_order
            )

    def _sequential_train_continuous(
        self,
        tensordict: TensorDictBase,
        actor_optimizer: torch.optim.Optimizer,
        alpha_optimizer: Optional[torch.optim.Optimizer],
        agent_order: List[int],
    ) -> Dict[str, torch.Tensor]:
        actor_network = self.base_loss.actor_network
        qvalue_network = self.base_loss.qvalue_network
        
        action_key = (self.group, "action")
        log_prob_key = (self.group, "log_prob")
        q_value_key = (self.group, "state_action_value")
        
        with torch.no_grad():
            actor_td = actor_network(tensordict.clone())
            actions = actor_td.get(action_key).clone()
            logp_actions = actor_td.get(log_prob_key).clone()
        
        all_log_probs = []
        
        for agent_idx in agent_order:
            actor_td = actor_network(tensordict.clone())
            all_actions = actor_td.get(action_key)
            all_logp = actor_td.get(log_prob_key)
            
            actions = actions.detach().clone()
            actions[..., agent_idx, :] = all_actions[..., agent_idx, :]
            logp_agent = all_logp[..., agent_idx]
            
            q_td = tensordict.clone()
            q_td.set(action_key, actions)
            qvalue_out = qvalue_network(q_td)
            q_values = qvalue_out.get(q_value_key)
            
            if q_values.shape[-1] > 1:
                q_value = q_values.min(dim=-1)[0]
            else:
                q_value = q_values.squeeze(-1)
            
            q_value_agent = q_value[..., agent_idx]
            
            alpha = self._get_alpha(agent_idx)
            actor_loss = -(q_value_agent - alpha * logp_agent).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            if alpha_optimizer is not None and not self.fixed_alpha:
                if self.per_agent_alpha and self.log_alphas is not None:
                    log_alpha = self.log_alphas[agent_idx]
                    target_entropy = self.base_loss.target_entropy
                    alpha_loss = -(log_alpha * (logp_agent.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    if agent_idx == agent_order[-1]:
                        log_alpha = self.base_loss.log_alpha
                        target_entropy = self.base_loss.target_entropy
                        mean_logp = torch.stack(all_log_probs + [logp_agent.detach()], dim=-1).mean()
                        alpha_loss = -(log_alpha * (mean_logp + target_entropy)).mean()
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
            
            all_log_probs.append(logp_agent.detach())
            
            with torch.no_grad():
                actor_td = actor_network(tensordict.clone())
                actions = actions.detach().clone()
                actions[..., agent_idx, :] = actor_td.get(action_key)[..., agent_idx, :]
        
        stacked_log_probs = torch.stack(all_log_probs, dim=-1)
        mean_log_prob = stacked_log_probs.mean()
        
        return {"mean_log_prob": mean_log_prob}

    def _sequential_train_discrete(
        self,
        tensordict: TensorDictBase,
        actor_optimizer: torch.optim.Optimizer,
        alpha_optimizer: Optional[torch.optim.Optimizer],
        agent_order: List[int],
    ) -> Dict[str, torch.Tensor]:
        actor_network = self.base_loss.actor_network
        qvalue_network = self.base_loss.qvalue_network
        
        action_value_key = (self.group, "action_value")
        
        td_q = tensordict.select(*qvalue_network.in_keys, strict=False)
        with torch.no_grad():
            qvalue_params_detached = self.base_loss.qvalue_network_params.detach()
            td_q = self.base_loss._vmap_qnetworkN0(td_q, qvalue_params_detached)
            min_q = td_q.get(action_value_key).min(0)[0]
        
        all_log_probs = []
        
        for agent_idx in agent_order:
            with self.base_loss.actor_network_params.to_module(actor_network):
                dist = actor_network.get_dist(tensordict.clone(False))
            
            prob = dist.probs
            log_prob = dist.logits
            
            agent_prob = prob[..., agent_idx, :]
            agent_log_prob = log_prob[..., agent_idx, :]
            agent_q = min_q[..., agent_idx, :]
            
            alpha = self._get_alpha(agent_idx)
            actor_loss = (agent_prob * (alpha * agent_log_prob - agent_q)).sum(-1).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            with torch.no_grad():
                agent_expected_log_prob = (agent_prob * agent_log_prob).sum(-1)
            
            if alpha_optimizer is not None and not self.fixed_alpha:
                if self.per_agent_alpha and self.log_alphas is not None:
                    log_alpha = self.log_alphas[agent_idx]
                    target_entropy = self.base_loss.target_entropy
                    alpha_loss = -(log_alpha * (agent_expected_log_prob + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    if agent_idx == agent_order[-1]:
                        log_alpha = self.base_loss.log_alpha
                        target_entropy = self.base_loss.target_entropy
                        mean_logp = torch.stack(all_log_probs + [agent_expected_log_prob], dim=-1).mean()
                        alpha_loss = -(log_alpha * (mean_logp + target_entropy)).mean()
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
            
            all_log_probs.append(agent_expected_log_prob)
        
        stacked_log_probs = torch.stack(all_log_probs, dim=-1)
        mean_log_prob = stacked_log_probs.mean()
        
        return {"mean_log_prob": mean_log_prob}

    def _accumulated_actor_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        agent_order = self._get_agent_order()
        
        if self.continuous:
            return self._accumulated_loss_continuous(tensordict, agent_order)
        else:
            return self._accumulated_loss_discrete(tensordict, agent_order)

    def _accumulated_loss_continuous(
        self, tensordict: TensorDictBase, agent_order: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actor_network = self.base_loss.actor_network
        qvalue_network = self.base_loss.qvalue_network

        action_key = (self.group, "action")
        log_prob_key = (self.group, "log_prob")
        q_value_key = (self.group, "state_action_value")

        with torch.no_grad():
            actor_td = actor_network(tensordict.clone())
            initial_actions = actor_td.get(action_key).clone()

        current_actions = initial_actions.clone()
        device = current_actions.device
        total_actor_loss = torch.tensor(0.0, device=device)
        all_log_probs = []

        for agent_idx in agent_order:
            actor_td = actor_network(tensordict.clone())
            all_actions_with_grad = actor_td.get(action_key)
            all_log_probs_tensor = actor_td.get(log_prob_key)

            agent_action = all_actions_with_grad[..., agent_idx, :]
            agent_log_prob = all_log_probs_tensor[..., agent_idx]

            updated_actions = current_actions.detach().clone()
            updated_actions[..., agent_idx, :] = agent_action

            q_td = tensordict.clone()
            q_td.set(action_key, updated_actions)
            qvalue_out = qvalue_network(q_td)
            q_values = qvalue_out.get(q_value_key)

            if q_values.shape[-1] > 1:
                q_value = q_values.min(dim=-1)[0]
            else:
                q_value = q_values.squeeze(-1)

            q_value_agent = q_value[..., agent_idx]

            alpha = self._get_alpha(agent_idx)
            agent_actor_loss = (alpha * agent_log_prob - q_value_agent).mean()

            total_actor_loss = total_actor_loss + agent_actor_loss
            all_log_probs.append(agent_log_prob)

            current_actions = updated_actions.detach()

        stacked_log_probs = torch.stack(all_log_probs, dim=-1)
        mean_log_prob = stacked_log_probs.mean()

        return total_actor_loss, mean_log_prob

    def _accumulated_loss_discrete(
        self, tensordict: TensorDictBase, agent_order: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actor_network = self.base_loss.actor_network
        qvalue_network = self.base_loss.qvalue_network
        
        action_value_key = (self.group, "action_value")
        
        td_q = tensordict.select(*qvalue_network.in_keys, strict=False)
        with torch.no_grad():
            qvalue_params_detached = self.base_loss.qvalue_network_params.detach()
            td_q = self.base_loss._vmap_qnetworkN0(td_q, qvalue_params_detached)
            min_q = td_q.get(action_value_key).min(0)[0]
        
        device = min_q.device
        total_actor_loss = torch.tensor(0.0, device=device)
        all_log_probs = []
        
        for agent_idx in agent_order:
            with self.base_loss.actor_network_params.to_module(actor_network):
                dist = actor_network.get_dist(tensordict.clone(False))
            
            prob = dist.probs
            log_prob = dist.logits
            
            agent_prob = prob[..., agent_idx, :]
            agent_log_prob = log_prob[..., agent_idx, :]
            agent_q = min_q[..., agent_idx, :]
            
            alpha = self._get_alpha(agent_idx)
            agent_loss = (agent_prob * (alpha * agent_log_prob - agent_q)).sum(-1)
            agent_actor_loss = agent_loss.mean()
            
            total_actor_loss = total_actor_loss + agent_actor_loss
            
            with torch.no_grad():
                agent_expected_log_prob = (agent_prob * agent_log_prob).sum(-1)
            all_log_probs.append(agent_expected_log_prob)
        
        stacked_log_probs = torch.stack(all_log_probs, dim=-1)
        mean_log_prob = stacked_log_probs.mean()
        
        return total_actor_loss, mean_log_prob

    def _per_agent_alpha_loss(self, mean_log_prob: torch.Tensor) -> torch.Tensor:
        if self.log_alphas is None:
            return torch.tensor(0.0)

        target_entropy = self.base_loss.target_entropy

        total_alpha_loss = torch.tensor(0.0, device=mean_log_prob.device)
        for agent_idx in range(self.n_agents):
            log_alpha = self.log_alphas[agent_idx]
            agent_alpha_loss = -log_alpha * (mean_log_prob + target_entropy).detach()
            total_alpha_loss = total_alpha_loss + agent_alpha_loss.mean()

        return total_alpha_loss

    def make_value_estimator(self, *args, **kwargs):
        return self.base_loss.make_value_estimator(*args, **kwargs)

    @property
    def actor_network_params(self):
        return self.base_loss.actor_network_params

    @property
    def qvalue_network_params(self):
        return self.base_loss.qvalue_network_params

    @property
    def target_qvalue_network_params(self):
        return self.base_loss.target_qvalue_network_params

    @property
    def actor_network(self):
        return self.base_loss.actor_network

    @property
    def qvalue_network(self):
        return self.base_loss.qvalue_network

    @property
    def log_alpha(self):
        if self.per_agent_alpha and self.log_alphas is not None:
            return self.log_alphas[0]
        return self.base_loss.log_alpha


@dataclass
class HasacConfig(MasacConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Hasac`."""
    fixed_order: bool = MISSING
    per_agent_alpha: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Hasac

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
