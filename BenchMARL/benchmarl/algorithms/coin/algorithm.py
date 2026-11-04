from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Type
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import Composite, Unbounded
from torchrl.modules import EGreedyModule, QMixer, QValueModule
from torchrl.objectives import LossModule, QMixerLoss, ValueEstimators
from torchrl.objectives.utils import distance_loss

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.qmix import Qmix
from benchmarl.models.common import ModelConfig

from .predictor_networks import PredictNetworkOS, PredictNetwork


class CoinMixerLoss(QMixerLoss):
    def __init__(self, *args, predictor_networks_os=None, predictor_networks_with_action=None,
                 predictor_networks_without_action=None, group=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_networks_os = predictor_networks_os
        self.predictor_networks_with_action = predictor_networks_with_action
        self.predictor_networks_without_action = predictor_networks_without_action
        self.group = group
    
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_copy = tensordict.clone(False)
        
        influence_intrinsic_reward = tensordict.get(
            ("coin", "influence_intrinsic_reward"),
            default=None
        )
        curiosity_intrinsic_reward = tensordict.get(
            ("coin", "curiosity_intrinsic_reward"),
            default=None
        )
        
        with self.local_value_network_params.to_module(self.local_value_network):
            self.local_value_network(td_copy)
        
        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(self.tensor_keys.action_value)
        
        if self.action_space == "categorical":
            if action.shape != pred_val.shape:
                action = action.unsqueeze(-1)
            pred_val_index = torch.gather(pred_val, -1, index=action)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1, keepdim=True)
        
        if influence_intrinsic_reward is not None:
            pred_val_index = pred_val_index - influence_intrinsic_reward
        
        td_copy.set(self.tensor_keys.local_value, pred_val_index)
        
        with self.mixer_network_params.to_module(self.mixer_network):
            self.mixer_network(td_copy)
        pred_val_index = td_copy.get(self.tensor_keys.global_value).squeeze(-1)
        
        target_value = self.value_estimator.value_estimate(
            td_copy,
            target_params=self._cached_target_params,
        ).squeeze(-1)
        
        if curiosity_intrinsic_reward is not None:
            target_value = target_value + curiosity_intrinsic_reward.squeeze(-1)
        
        with torch.no_grad():
            priority_tensor = (pred_val_index - target_value).pow(2)
            priority_tensor = priority_tensor.unsqueeze(-1)
        if tensordict.device is not None:
            priority_tensor = priority_tensor.to(tensordict.device)
        
        tensordict.set(
            self.tensor_keys.priority,
            priority_tensor,
            inplace=True,
        )
        
        qmix_loss = distance_loss(pred_val_index, target_value, self.loss_function)
        
        return TensorDict({"loss": qmix_loss.mean()}, batch_size=[])


class Coin(Qmix):
    """COIN: Two Heads are Better Than One for Multi-Agent RL."""
    def __init__(
        self,
        mixing_embed_dim: int,
        delay_value: bool,
        loss_function: str,
        **kwargs
    ):
        coin_params = {
            'beta1',
            'beta2',
            'beta',
            'predictor_lr',
            'predict_net_dim',
            'influence_exp_decay',
            'influence_exp_decay_rate',
            'influence_exp_decay_cycle',
            'influence_exp_decay_stop',
            'curiosity_exp_decay',
            'curiosity_exp_decay_rate',
            'curiosity_exp_decay_cycle',
            'curiosity_exp_decay_stop',
            'target_update_interval',
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in coin_params}
        
        super().__init__(
            mixing_embed_dim=mixing_embed_dim,
            delay_value=delay_value,
            loss_function=loss_function,
            **filtered_kwargs
        )
        
        cfg = self.experiment.algorithm_config
        
        self.beta1 = getattr(cfg, 'beta1', 0.0005)
        self.beta2 = getattr(cfg, 'beta2', 0.0005)
        self.beta = getattr(cfg, 'beta', 0.5)
        self.predictor_lr = getattr(cfg, 'predictor_lr', 1e-3)
        self.predict_net_dim = getattr(cfg, 'predict_net_dim', 64)
        
        self.influence_exp_decay = getattr(cfg, 'influence_exp_decay', False)
        self.influence_exp_decay_rate = getattr(cfg, 'influence_exp_decay_rate', 0.9)
        self.influence_exp_decay_cycle = getattr(cfg, 'influence_exp_decay_cycle', 100000)
        self.influence_exp_decay_stop = getattr(cfg, 'influence_exp_decay_stop', 0.00001)
        self.curiosity_exp_decay = getattr(cfg, 'curiosity_exp_decay', False)
        self.curiosity_exp_decay_rate = getattr(cfg, 'curiosity_exp_decay_rate', 0.9)
        self.curiosity_exp_decay_cycle = getattr(cfg, 'curiosity_exp_decay_cycle', 50000)
        self.curiosity_exp_decay_stop = getattr(cfg, 'curiosity_exp_decay_stop', 0.0)
        
        self.decay_stats_inf_t = 0
        self.decay_stats_cur_t = 0
        
        self._training_steps: Dict[str, int] = {}
        self._last_target_update: Dict[str, int] = {}
        self.target_update_interval = getattr(cfg, 'target_update_interval', 200)
        
        self._build_predictor_networks()
        
        self._coin_metrics: Dict[str, dict] = {}
        
        self._predictor_batch_data: Dict[str, dict] = {}
    
    def _build_predictor_networks(self) -> None:
        self.predict_networks_os: Dict[str, PredictNetworkOS] = {}
        self.target_predict_networks_os: Dict[str, PredictNetworkOS] = {}
        self.predict_networks_with_action: Dict[str, PredictNetwork] = {}
        self.target_predict_networks_with_action: Dict[str, PredictNetwork] = {}
        self.predict_networks_without_action: Dict[str, PredictNetwork] = {}
        self.target_predict_networks_without_action: Dict[str, PredictNetwork] = {}
        self.predictor_optimizers: Dict[str, optim.Adam] = {}
        
        for group, agents in self.group_map.items():
            n_agents = len(agents)
            obs_shape = self.observation_spec[group, "observation"].shape[-1]
            
            if self.state_spec is not None:
                state_key = list(self.state_spec.keys(True, True))[0]
                state_shape = self.state_spec[state_key].shape[-1]
            else:
                state_shape = obs_shape * n_agents
            
            rnn_hidden_dim = 64
            if hasattr(self.model_config, 'hidden_size'):
                rnn_hidden_dim = self.model_config.hidden_size
            elif hasattr(self.model_config, 'num_cells') and len(self.model_config.num_cells) > 0:
                rnn_hidden_dim = self.model_config.num_cells[0]
            
            action_spec_item = self.action_spec[group, "action"]
            if hasattr(action_spec_item, 'space') and hasattr(action_spec_item.space, 'n'):
                n_actions = int(action_spec_item.space.n)
            else:
                raise ValueError("COIN only supports discrete actions")
            
            num_inputs_os = (rnn_hidden_dim + n_actions) * n_agents
            predict_os = PredictNetworkOS(
                num_inputs=num_inputs_os,
                hidden_dim=self.predict_net_dim,
                obs_shape=obs_shape,
                state_shape=state_shape,
                n_agents=n_agents,
            ).to(self.device)
            target_predict_os = copy.deepcopy(predict_os)
            
            num_inputs_with = rnn_hidden_dim + n_actions + n_actions
            predict_with = PredictNetwork(
                num_inputs=num_inputs_with,
                hidden_dim=self.predict_net_dim,
                output_shape=obs_shape,
            ).to(self.device)
            target_predict_with = copy.deepcopy(predict_with)
            
            num_inputs_without = rnn_hidden_dim + n_actions
            predict_without = PredictNetwork(
                num_inputs=num_inputs_without,
                hidden_dim=self.predict_net_dim,
                output_shape=obs_shape,
            ).to(self.device)
            target_predict_without = copy.deepcopy(predict_without)
            
            predictor_params = (
                list(predict_os.parameters()) +
                list(predict_with.parameters()) +
                list(predict_without.parameters())
            )
            predictor_optimizer = optim.Adam(predictor_params, lr=self.predictor_lr)
            
            self.predict_networks_os[group] = predict_os
            self.target_predict_networks_os[group] = target_predict_os
            self.predict_networks_with_action[group] = predict_with
            self.target_predict_networks_with_action[group] = target_predict_with
            self.predict_networks_without_action[group] = predict_without
            self.target_predict_networks_without_action[group] = target_predict_without
            self.predictor_optimizers[group] = predictor_optimizer
    
    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if continuous:
            raise NotImplementedError("COIN is not compatible with continuous actions.")
        
        loss_module = CoinMixerLoss(
            policy_for_loss,
            self.get_mixer(group),
            delay_value=self.delay_value,
            loss_function=self.loss_function,
            action_space=self.action_spec[group, "action"],
            predictor_networks_os=self.predict_networks_os.get(group),
            predictor_networks_with_action=self.predict_networks_with_action.get(group),
            predictor_networks_without_action=self.predict_networks_without_action.get(group),
            group=group,
        )
        loss_module.set_keys(
            reward="reward",
            action=(group, "action"),
            done="done",
            terminated="terminated",
            action_value=(group, "action_value"),
            local_value=(group, "chosen_action_value"),
            global_value="chosen_action_value",
            priority="td_error",
        )
        loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=self.experiment_config.gamma
        )
        
        return loss_module, True
    
    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        params = super()._get_parameters(group, loss)
        
        predictor_params = (
            list(self.predict_networks_os[group].parameters()) +
            list(self.predict_networks_with_action[group].parameters()) +
            list(self.predict_networks_without_action[group].parameters())
        )
        params["predictor"] = predictor_params
        
        return params
    
    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        batch = super().process_batch(group, batch)
        
        group_data = batch.get(group)
        n_agents = len(self.group_map[group])
        
        if len(group_data.shape) >= 3 and group_data.shape[-1] == n_agents:
            num_envs, T = group_data.shape[0], group_data.shape[1]
            batch_format = "env_first"
        else:
            T, num_envs = group_data.shape[0], group_data.shape[1]
            batch_format = "time_first"
        
        obs = batch.get((group, "observation"))
        obs_next = batch.get(("next", group, "observation"))
        actions = batch.get((group, "action"))
        done = batch.get(("next", group, "done"))
        if done is None:
            done = batch.get((group, "done"))
        if done is None:
            done = batch.get(("next", "done"))
        terminated = batch.get(("next", group, "terminated"), default=done)
        
        if batch_format == "env_first":
            obs = obs.transpose(0, 1)
            if obs_next is not None:
                obs_next = obs_next.transpose(0, 1)
            if actions is not None:
                actions = actions.transpose(0, 1)
            if done is not None:
                done = done.transpose(0, 1)
            if terminated is not None:
                terminated = terminated.transpose(0, 1)
        
        if obs.ndim == 3:
            obs = obs.unsqueeze(-2).expand(-1, -1, n_agents, -1)
        elif obs.ndim == 4 and obs.shape[2] != n_agents:
            if obs.shape[2] == 1:
                obs = obs.expand(-1, -1, n_agents, -1)
        
        if obs_next is not None:
            if obs_next.ndim == 3:
                obs_next = obs_next.unsqueeze(-2).expand(-1, -1, n_agents, -1)
            elif obs_next.ndim == 4 and obs_next.shape[2] != n_agents:
                if obs_next.shape[2] == 1:
                    obs_next = obs_next.expand(-1, -1, n_agents, -1)
        
        if actions is not None:
            if actions.ndim == 4:
                actions = actions.squeeze(-1)
            elif actions.ndim == 3 and actions.shape[-1] == 1:
                actions = actions.squeeze(-1)
            elif actions.ndim == 2:
                actions = actions.unsqueeze(-1).expand(-1, -1, n_agents)
        
        if done is not None:
            if done.ndim == 2:
                done = done.unsqueeze(-1).expand(-1, -1, n_agents)
            elif done.ndim == 3 and done.shape[-1] == 1:
                done = done.expand(-1, -1, n_agents)
        
        if terminated is not None:
            if terminated.ndim == 2:
                terminated = terminated.unsqueeze(-1).expand(-1, -1, n_agents)
            elif terminated.ndim == 3 and terminated.shape[-1] == 1:
                terminated = terminated.expand(-1, -1, n_agents)
        
        if self.state_spec is not None:
            state_key = list(self.state_spec.keys(True, True))[0]
            state = batch.get(state_key)
            state_next = batch.get(("next", state_key))
            
            if batch_format == "env_first" and state is not None:
                state = state.transpose(0, 1)
            if batch_format == "env_first" and state_next is not None:
                state_next = state_next.transpose(0, 1)
        else:
            if obs.ndim == 4:
                state = obs.flatten(-2, -1)
                state_next = obs_next.flatten(-2, -1) if obs_next is not None else None
            else:
                state = obs
                state_next = obs_next
        
        batch_dim = num_envs
        time_dim = T
        
        rnn_hidden_dim = 64
        if hasattr(self.model_config, 'hidden_size'):
            rnn_hidden_dim = self.model_config.hidden_size
        elif hasattr(self.model_config, 'num_cells') and len(self.model_config.num_cells) > 0:
            rnn_hidden_dim = self.model_config.num_cells[0]
        
        action_spec_item = self.action_spec[group, "action"]
        if hasattr(action_spec_item, 'space') and hasattr(action_spec_item.space, 'n'):
            n_actions = int(action_spec_item.space.n)
        else:
            raise ValueError("COIN only supports discrete actions")
        
        obs_shape = obs.shape[-1]
        state_shape = state.shape[-1] if state is not None else obs_shape * n_agents
        
        hidden_states, hidden_state_fallback = self._extract_hidden_states(
            group=group,
            batch=batch,
            time_dim=time_dim,
            num_envs=num_envs,
            n_agents=n_agents,
            rnn_hidden_dim=rnn_hidden_dim,
        )
        
        if actions is not None and actions.shape[0] > 1:
            actions_curr = actions[:-1]
        else:
            actions_curr = actions
        
        if obs_next is not None and obs_next.shape[0] > 1:
            obs_next_actual = obs_next[:-1]
        elif obs_next is not None:
            obs_next_actual = obs_next
        else:
            if obs.shape[0] > 1:
                obs_next_actual = obs[1:]
            else:
                obs_next_actual = obs
        
        if state_next is not None and state_next.shape[0] > 1:
            state_next_actual = state_next[:-1]
        elif state_next is not None:
            state_next_actual = state_next
        else:
            if state.shape[0] > 1:
                state_next_actual = state[1:]
            else:
                state_next_actual = state
        
        if hidden_states is not None and hidden_states.shape[0] > 1:
            hidden_states_curr = hidden_states[:-1]
        else:
            hidden_states_curr = hidden_states
        
        if actions_curr is not None:
            time_dim_curr = actions_curr.shape[0]
        elif obs_next_actual is not None:
            time_dim_curr = obs_next_actual.shape[0]
        elif hidden_states_curr is not None:
            time_dim_curr = hidden_states_curr.shape[0]
        else:
            time_dim_curr = max(1, time_dim - 1)
        
        if terminated is not None and terminated.shape[0] > 1:
            terminated_curr = terminated[:-1].float()
        elif terminated is not None:
            terminated_curr = terminated.float()
        else:
            terminated_curr = None
        
        if terminated_curr is not None:
            if terminated_curr.ndim == 3:
                mask_next = (1.0 - terminated_curr.mean(dim=-1, keepdim=True))
            else:
                mask_next = (1.0 - terminated_curr)
                if mask_next.ndim == 2:
                    mask_next = mask_next.unsqueeze(-1)
            
            if mask_next.shape[0] != time_dim_curr:
                if mask_next.shape[0] == 1 and time_dim_curr > 1:
                    mask_next = mask_next.expand(time_dim_curr, -1, -1)
                elif mask_next.shape[0] > time_dim_curr:
                    mask_next = mask_next[:time_dim_curr]
        else:
            mask_next = torch.ones(time_dim_curr, num_envs, 1, device=self.device)
        
        if actions_curr is None:
            actions_curr = torch.zeros(time_dim_curr, num_envs, n_agents, dtype=torch.long, device=self.device)
        elif actions_curr.ndim == 2:
            actions_curr = actions_curr.unsqueeze(0).expand(time_dim_curr, -1, -1)
        
        actions_one_hot = torch.zeros(
            time_dim_curr, num_envs, n_agents, n_actions,
            device=self.device
        )
        actions_curr_long = actions_curr.long()
        if actions_curr_long.ndim == 3:
            actions_one_hot.scatter_(
                dim=-1,
                index=actions_curr_long.unsqueeze(-1),
                value=1.0
            )
        else:
            actions_one_hot.scatter_(
                dim=-1,
                index=actions_curr_long.unsqueeze(0).unsqueeze(-1),
                value=1.0
            )
        
        with torch.no_grad():
            if hidden_states_curr is None:
                hidden_states_curr = torch.zeros(
                    time_dim_curr, num_envs, n_agents, rnn_hidden_dim,
                    device=self.device
                )
            
            if hidden_states_curr.shape[0] != actions_one_hot.shape[0]:
                min_time = min(hidden_states_curr.shape[0], actions_one_hot.shape[0])
                hidden_states_curr = hidden_states_curr[:min_time]
                actions_one_hot = actions_one_hot[:min_time]
                time_dim_curr = min_time
            
            os_predict_input = torch.cat(
                [hidden_states_curr, actions_one_hot], dim=-1
            ).flatten(-2, -1)
            
            actual_time_from_input = os_predict_input.shape[0]
            
            os_predict_input = os_predict_input.transpose(0, 1)
            
            target_predict_os = self.target_predict_networks_os[group]
            
            if os_predict_input.ndim == 2:
                os_predict_input = os_predict_input.unsqueeze(1)
            elif os_predict_input.ndim == 1:
                os_predict_input = os_predict_input.unsqueeze(0).unsqueeze(0)
            
            predict_obs_next, predict_state_next = target_predict_os(os_predict_input)
            
            if predict_obs_next is None:
                predict_obs_next = torch.zeros(
                    os_predict_input.shape[0], os_predict_input.shape[1], obs_shape * n_agents,
                    device=self.device
                )
            if predict_state_next is None:
                predict_state_next = torch.zeros(
                    os_predict_input.shape[0], os_predict_input.shape[1], state_shape,
                    device=self.device
                )
            
            actual_time_from_output = predict_obs_next.shape[1] if predict_obs_next.ndim >= 2 else 1
            
            if obs_next_actual.ndim == 3:
                obs_next_actual = obs_next_actual.unsqueeze(1).expand(-1, num_envs, -1, -1)
            elif obs_next_actual.ndim == 4:
                if obs_next_actual.shape[0] > obs_next_actual.shape[1]:
                    pass
                else:
                    obs_next_actual = obs_next_actual.transpose(0, 1)
            
            if obs_next_actual.shape[0] == time_dim_curr and obs_next_actual.shape[1] == num_envs:
                obs_next_actual_transposed = obs_next_actual.transpose(0, 1)
            else:
                obs_next_actual_transposed = obs_next_actual
            
            obs_time_dim = obs_next_actual_transposed.shape[1] if obs_next_actual_transposed.ndim >= 2 else 1
            min_time_dim = min(actual_time_from_output, obs_time_dim, actual_time_from_input)
            
            if predict_obs_next.shape[1] > min_time_dim:
                predict_obs_next = predict_obs_next[:, :min_time_dim]
            if predict_state_next.shape[1] > min_time_dim:
                predict_state_next = predict_state_next[:, :min_time_dim]
            if obs_next_actual_transposed.shape[1] > min_time_dim:
                obs_next_actual_transposed = obs_next_actual_transposed[:, :min_time_dim]
            
            obs_next_flatten = obs_next_actual_transposed.flatten(-2, -1)
            
            if predict_obs_next.ndim < 3:
                if predict_obs_next.ndim == 2:
                    predict_obs_next = predict_obs_next.unsqueeze(-1)
                elif predict_obs_next.ndim == 1:
                    predict_obs_next = predict_obs_next.unsqueeze(0).unsqueeze(-1)
            
            if obs_next_flatten.ndim < 3:
                if obs_next_flatten.ndim == 2:
                    obs_next_flatten = obs_next_flatten.unsqueeze(-1)
                elif obs_next_flatten.ndim == 1:
                    obs_next_flatten = obs_next_flatten.unsqueeze(0).unsqueeze(-1)
            
            if predict_obs_next.shape[0] != obs_next_flatten.shape[0]:
                min_batch = min(predict_obs_next.shape[0], obs_next_flatten.shape[0])
                predict_obs_next = predict_obs_next[:min_batch]
                obs_next_flatten = obs_next_flatten[:min_batch]
            
            if predict_obs_next.shape[1] != obs_next_flatten.shape[1]:
                min_time = min(predict_obs_next.shape[1], obs_next_flatten.shape[1])
                predict_obs_next = predict_obs_next[:, :min_time]
                obs_next_flatten = obs_next_flatten[:, :min_time]
            
            if predict_obs_next.shape[-1] != obs_next_flatten.shape[-1]:
                min_feat = min(predict_obs_next.shape[-1], obs_next_flatten.shape[-1])
                predict_obs_next = predict_obs_next[..., :min_feat]
                obs_next_flatten = obs_next_flatten[..., :min_feat]
            
            mask_next_transposed = mask_next.transpose(0, 1)
            
            if mask_next_transposed.shape[0] != predict_obs_next.shape[0]:
                min_mask_batch = min(mask_next_transposed.shape[0], predict_obs_next.shape[0])
                mask_next_transposed = mask_next_transposed[:min_mask_batch]
            
            if mask_next_transposed.shape[1] != predict_obs_next.shape[1]:
                if mask_next_transposed.shape[1] == 1:
                    mask_next_transposed = mask_next_transposed.expand(-1, predict_obs_next.shape[1], -1)
                elif mask_next_transposed.shape[1] > predict_obs_next.shape[1]:
                    mask_next_transposed = mask_next_transposed[:, :predict_obs_next.shape[1]]
                else:
                    mask_next_transposed = mask_next_transposed.expand(-1, predict_obs_next.shape[1], -1)
            
            if predict_obs_next.shape != obs_next_flatten.shape:
                target_shape = predict_obs_next.shape
                if obs_next_flatten.numel() == predict_obs_next.numel():
                    obs_next_flatten = obs_next_flatten.reshape(target_shape)
                else:
                    raise RuntimeError(
                        f"Shape mismatch in COIN curiosity reward computation:\n"
                        f"  predict_obs_next shape: {predict_obs_next.shape}, numel: {predict_obs_next.numel()}\n"
                        f"  obs_next_flatten shape: {obs_next_flatten.shape}, numel: {obs_next_flatten.numel()}\n"
                        f"  os_predict_input shape: {os_predict_input.shape}\n"
                        f"  obs_next_actual shape: {obs_next_actual.shape if 'obs_next_actual' in locals() else 'N/A'}\n"
                        f"  obs_next_actual_transposed shape: {obs_next_actual_transposed.shape if 'obs_next_actual_transposed' in locals() else 'N/A'}\n"
                        f"  time_dim_curr: {time_dim_curr}, num_envs: {num_envs}, n_agents: {n_agents}\n"
                        f"  obs_shape: {obs_shape}, state_shape: {state_shape}\n"
                        f"  actual_time_from_output: {actual_time_from_output}, actual_time_from_input: {actual_time_from_input}\n"
                        f"  min_time_dim: {min_time_dim}"
                    )
            
            if mask_next_transposed.shape[:2] != predict_obs_next.shape[:2]:
                mask_next_transposed = mask_next_transposed[:predict_obs_next.shape[0], :predict_obs_next.shape[1]]
            
            if predict_obs_next.shape[:2] != obs_next_flatten.shape[:2]:
                raise RuntimeError(
                    f"Final shape check failed:\n"
                    f"  predict_obs_next shape: {predict_obs_next.shape}\n"
                    f"  obs_next_flatten shape: {obs_next_flatten.shape}\n"
                    f"  Mismatch at dimensions: {predict_obs_next.shape[:2]} vs {obs_next_flatten.shape[:2]}"
                )
            
            mask_expanded = mask_next_transposed.expand(-1, -1, predict_obs_next.shape[-1])
            obs_error = ((predict_obs_next - obs_next_flatten) ** 2) * mask_expanded
            
            if state_next_actual.ndim == 2:
                state_next_actual = state_next_actual.unsqueeze(1).expand(-1, num_envs, -1)
            
            state_next_actual_transposed = state_next_actual.transpose(0, 1)
            
            if predict_state_next.shape[1] != state_next_actual_transposed.shape[1]:
                min_time_state = min(predict_state_next.shape[1], state_next_actual_transposed.shape[1])
                predict_state_next = predict_state_next[:, :min_time_state]
                state_next_actual_transposed = state_next_actual_transposed[:, :min_time_state]
            
            if predict_state_next.shape[0] != state_next_actual_transposed.shape[0]:
                min_batch_state = min(predict_state_next.shape[0], state_next_actual_transposed.shape[0])
                predict_state_next = predict_state_next[:min_batch_state]
                state_next_actual_transposed = state_next_actual_transposed[:min_batch_state]
            
            if predict_state_next.shape[-1] != state_next_actual_transposed.shape[-1]:
                min_feat_state = min(predict_state_next.shape[-1], state_next_actual_transposed.shape[-1])
                predict_state_next = predict_state_next[..., :min_feat_state]
                state_next_actual_transposed = state_next_actual_transposed[..., :min_feat_state]
            
            mask_state_expanded = mask_next_transposed.expand(-1, -1, predict_state_next.shape[-1])
            state_error = ((predict_state_next - state_next_actual_transposed) ** 2) * mask_state_expanded
            
            obs_error_sum = obs_error.sum(dim=-1, keepdim=True)
            state_error_sum = state_error.sum(dim=-1, keepdim=True)
            
            curiosity_intrinsic_reward = (obs_error_sum + state_error_sum) * self.beta2
            
            curiosity_intrinsic_reward = curiosity_intrinsic_reward.transpose(0, 1)
        
        with torch.no_grad():
            obs_diverge = []
            target_predict_with = self.target_predict_networks_with_action[group]
            target_predict_without = self.target_predict_networks_without_action[group]
            
            for i in range(n_agents):
                other_indices = [j for j in range(n_agents) if j != i]
                
                actions_one_hot_i_pre = actions_one_hot[:, :, other_indices, :]
                actions_one_hot_i = actions_one_hot[:, :, i:i+1, :].expand(-1, -1, n_agents-1, -1)
                obs_next_i = obs_next_actual[:, :, other_indices, :]
                hidden_out_without_i = hidden_states_curr[:, :, other_indices, :]
                
                influence_inputs_i_without_action = torch.cat(
                    [hidden_out_without_i, actions_one_hot_i_pre], dim=-1
                )
                
                influence_inputs_i_with_action = torch.cat(
                    [hidden_out_without_i, actions_one_hot_i_pre, actions_one_hot_i], dim=-1
                )
                
                influence_inputs_i_without_action_flat = influence_inputs_i_without_action.flatten(0, 1)
                influence_inputs_i_with_action_flat = influence_inputs_i_with_action.flatten(0, 1)
                obs_next_i_flat = obs_next_i.flatten(0, 1)
                
                log_p_o_i = target_predict_without.get_log_pi(
                    influence_inputs_i_without_action_flat, obs_next_i_flat
                )
                
                log_q_o_i = target_predict_with.get_log_pi(
                    influence_inputs_i_with_action_flat, obs_next_i_flat
                )
                
                log_p_o_i = log_p_o_i.reshape(time_dim_curr, num_envs, n_agents - 1)
                log_q_o_i = log_q_o_i.reshape(time_dim_curr, num_envs, n_agents - 1)
                
                obs_diverge_i = (
                    self.beta * log_q_o_i.sum(dim=-1, keepdim=True) -
                    log_p_o_i.sum(dim=-1, keepdim=True)
                ) * mask_next * self.beta1
                
                obs_diverge.append(obs_diverge_i)
            
            influence_intrinsic_reward = torch.stack(obs_diverge, dim=2)
        
        pad_curiosity = torch.zeros(1, num_envs, 1, device=self.device)
        pad_influence = torch.zeros(1, num_envs, n_agents, 1, device=self.device)
        
        curiosity_intrinsic_reward = torch.cat([curiosity_intrinsic_reward, pad_curiosity], dim=0)
        influence_intrinsic_reward = torch.cat([influence_intrinsic_reward, pad_influence], dim=0)
        
        if batch_format == "env_first":
            curiosity_intrinsic_reward = curiosity_intrinsic_reward.transpose(0, 1)
            influence_intrinsic_reward = influence_intrinsic_reward.transpose(0, 1)
        
        batch.set(("coin", "curiosity_intrinsic_reward"), curiosity_intrinsic_reward)
        batch.set(("coin", "influence_intrinsic_reward"), influence_intrinsic_reward)
        
        self._coin_metrics[group] = {
            "curiosity_reward_mean": curiosity_intrinsic_reward.mean().item(),
            "influence_reward_mean": influence_intrinsic_reward.mean().item(),
            "hidden_state_fallback": float(hidden_state_fallback),
        }
        
        if hidden_states_curr is not None and actions_one_hot is not None:
            self._predictor_batch_data[group] = {
                "hidden_states_curr": hidden_states_curr.detach(),
                "actions_one_hot": actions_one_hot.detach(),
                "obs_next_actual": obs_next_actual.detach() if obs_next_actual is not None else None,
                "state_next_actual": state_next_actual.detach() if state_next_actual is not None else None,
                "mask_next": mask_next.detach(),
                "n_agents": n_agents,
                "obs_shape": obs_shape,
                "state_shape": state_shape,
                "rnn_hidden_dim": rnn_hidden_dim,
                "n_actions": n_actions,
                "time_dim_curr": time_dim_curr,
                "num_envs": num_envs,
            }
        
        return batch
    
    def _normalize_hidden_states(
        self,
        hidden_states: torch.Tensor,
        time_dim: int,
        num_envs: int,
        n_agents: int,
        rnn_hidden_dim: int,
    ) -> torch.Tensor | None:
        if hidden_states is None:
            return None
        
        hs = hidden_states
        if hs.ndim == 5:
            hs = hs[..., -1, :]
        elif hs.ndim == 4 and hs.shape[-1] != rnn_hidden_dim and hs.shape[-2] == 1:
            hs = hs.squeeze(-2).unsqueeze(-2).expand(-1, -1, n_agents, -1)
        
        if hs.shape[0] == num_envs and hs.shape[1] == time_dim:
            hs = hs.transpose(0, 1)
        
        if hs.ndim == 3:
            hs = hs.unsqueeze(-2).expand(-1, -1, n_agents, -1)
        elif hs.ndim == 2:
            hs = hs.unsqueeze(0).unsqueeze(-2).expand(time_dim, -1, n_agents, -1)
        
        if hs.shape[0] != time_dim or hs.shape[1] != num_envs:
            return None
        
        if hs.shape[-1] != rnn_hidden_dim:
            if hs.shape[-1] > rnn_hidden_dim:
                hs = hs[..., :rnn_hidden_dim]
            else:
                pad_size = rnn_hidden_dim - hs.shape[-1]
                hs = F.pad(hs, (0, pad_size))
        return hs
    
    def _extract_hidden_states(
        self,
        group: str,
        batch: TensorDictBase,
        time_dim: int,
        num_envs: int,
        n_agents: int,
        rnn_hidden_dim: int,
    ) -> tuple[torch.Tensor, bool]:
        candidate_keys = []
        for key in batch.keys(True, True):
            if isinstance(key, tuple) and len(key) >= 2 and key[0] == group:
                if "hidden" in str(key[1]).lower():
                    candidate_keys.append(key)
        
        priorities = ["_hidden_gru", "_hidden_lstm", "hidden", "rnn"]
        for prefix in priorities:
            for key in candidate_keys:
                if prefix in str(key[1]).lower():
                    hs = self._normalize_hidden_states(
                        batch.get(key),
                        time_dim=time_dim,
                        num_envs=num_envs,
                        n_agents=n_agents,
                        rnn_hidden_dim=rnn_hidden_dim,
                    )
                    if hs is not None:
                        return hs, False
        
        hs = torch.zeros(
            time_dim, num_envs, n_agents, rnn_hidden_dim, device=self.device
        )
        return hs, True
    
    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        loss_vals.set("coin/beta1", torch.tensor(self.beta1, device=self.device))
        loss_vals.set("coin/beta2", torch.tensor(self.beta2, device=self.device))
        loss_vals.set("coin/beta", torch.tensor(self.beta, device=self.device))
        
        coin_metrics = self._coin_metrics.get(group, {})
        for key, value in coin_metrics.items():
            loss_vals.set(f"coin/{key}", torch.tensor(value, device=self.device))
        
        predictor_loss, predictor_os_loss, p_o_loss, q_o_loss = self._compute_predictor_loss(group)
        
        if predictor_loss is not None:
            loss_vals.set("coin/predictor_loss", torch.tensor(predictor_loss, device=self.device))
        if predictor_os_loss is not None:
            loss_vals.set("coin/predictor_os_loss", torch.tensor(predictor_os_loss, device=self.device))
        if p_o_loss is not None:
            loss_vals.set("coin/p_o_loss", torch.tensor(p_o_loss, device=self.device))
        if q_o_loss is not None:
            loss_vals.set("coin/q_o_loss", torch.tensor(q_o_loss, device=self.device))
        
        self._update_beta_decay(group)
        
        self._update_target_networks(group)
        
        return loss_vals
    
    def _compute_predictor_loss(self, group: str) -> Tuple[float, float, float, float]:
        if group not in self._predictor_batch_data:
            return None, None, None, None
        
        batch_data = self._predictor_batch_data[group]
        hidden_states_curr = batch_data["hidden_states_curr"]
        actions_one_hot = batch_data["actions_one_hot"]
        obs_next_actual = batch_data["obs_next_actual"]
        state_next_actual = batch_data["state_next_actual"]
        mask_next = batch_data["mask_next"]
        n_agents = batch_data["n_agents"]
        obs_shape = batch_data["obs_shape"]
        state_shape = batch_data["state_shape"]
        rnn_hidden_dim = batch_data["rnn_hidden_dim"]
        n_actions = batch_data["n_actions"]
        time_dim_curr = batch_data["time_dim_curr"]
        num_envs = batch_data["num_envs"]
        
        eval_predict_os = self.predict_networks_os[group]
        eval_predict_with = self.predict_networks_with_action[group]
        eval_predict_without = self.predict_networks_without_action[group]
        predictor_optimizer = self.predictor_optimizers[group]
        
        os_predict_input = torch.cat(
            [hidden_states_curr, actions_one_hot], dim=-1
        ).flatten(-2, -1)
        
        os_predict_input = os_predict_input.transpose(0, 1)
        
        eval_predict_obs_next, eval_predict_state_next = eval_predict_os(os_predict_input)
        
        if obs_next_actual is not None:
            obs_next_transposed = obs_next_actual.transpose(0, 1)
            obs_next_flatten = obs_next_transposed.flatten(-2, -1)
        else:
            obs_next_flatten = None
        
        if state_next_actual is not None:
            state_next_transposed = state_next_actual.transpose(0, 1)
        else:
            state_next_transposed = None
        
        mask_next_transposed = mask_next.transpose(0, 1)
        
        if obs_next_flatten is not None and state_next_transposed is not None:
            min_time = min(mask_next_transposed.shape[1], os_predict_input.shape[1])
            mask_os = mask_next_transposed[:, :min_time]
            
            predictor_os_loss_tensor = eval_predict_os.update(
                os_predict_input, obs_next_flatten, state_next_transposed, mask_os
            )
            predictor_os_loss = predictor_os_loss_tensor.item() if isinstance(predictor_os_loss_tensor, torch.Tensor) else predictor_os_loss_tensor
        else:
            predictor_os_loss = 0.0
            predictor_os_loss_tensor = torch.tensor(0.0, device=self.device)
        
        p_o_loss_total = torch.tensor(0.0, device=self.device)
        q_o_loss_total = torch.tensor(0.0, device=self.device)
        
        for i in range(n_agents):
            other_indices = [j for j in range(n_agents) if j != i]
            
            actions_one_hot_i_pre = actions_one_hot[:, :, other_indices, :]
            actions_one_hot_i = actions_one_hot[:, :, i:i+1, :].expand(-1, -1, n_agents-1, -1)
            obs_next_i = obs_next_actual[:, :, other_indices, :]
            hidden_out_without_i = hidden_states_curr[:, :, other_indices, :]
            
            influence_inputs_i_without_action = torch.cat(
                [hidden_out_without_i, actions_one_hot_i_pre], dim=-1
            )
            
            influence_inputs_i_with_action = torch.cat(
                [hidden_out_without_i, actions_one_hot_i_pre, actions_one_hot_i], dim=-1
            )
            
            influence_inputs_i_without_action_flat = influence_inputs_i_without_action.flatten(0, 1)
            influence_inputs_i_with_action_flat = influence_inputs_i_with_action.flatten(0, 1)
            obs_next_i_flat = obs_next_i.flatten(0, 1)
            
            mask_next_flat = mask_next.flatten(0, 1)
            
            p_o_i_loss = eval_predict_without.update(
                influence_inputs_i_without_action_flat, obs_next_i_flat, mask_next_flat
            )
            q_o_i_loss = eval_predict_with.update(
                influence_inputs_i_with_action_flat, obs_next_i_flat, mask_next_flat
            )
            
            if p_o_i_loss is not None:
                p_o_loss_total += p_o_i_loss
            if q_o_i_loss is not None:
                q_o_loss_total += q_o_i_loss
        
        p_o_loss = p_o_loss_total.item() if isinstance(p_o_loss_total, torch.Tensor) else p_o_loss_total
        q_o_loss = q_o_loss_total.item() if isinstance(q_o_loss_total, torch.Tensor) else q_o_loss_total
        
        predictor_loss = predictor_os_loss + p_o_loss + q_o_loss
        
        if isinstance(predictor_os_loss_tensor, torch.Tensor) and predictor_os_loss_tensor.requires_grad:
            predictor_optimizer.zero_grad()
            predictor_os_loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(
                list(eval_predict_os.parameters()) +
                list(eval_predict_with.parameters()) +
                list(eval_predict_without.parameters()),
                10.0
            )
            predictor_optimizer.step()
        
        return predictor_loss, predictor_os_loss, p_o_loss, q_o_loss
    
    def _update_beta_decay(self, group: str) -> None:
        if group not in self._training_steps:
            self._training_steps[group] = 0
        self._training_steps[group] += 1
        t_env = self._training_steps[group]
        
        if self.influence_exp_decay:
            if t_env - self.decay_stats_inf_t >= self.influence_exp_decay_cycle:
                if self.influence_exp_decay_rate <= 1.0:
                    if self.beta1 > self.influence_exp_decay_stop:
                        self.beta1 = self.beta1 * self.influence_exp_decay_rate
                    else:
                        self.beta1 = self.influence_exp_decay_stop
                else:
                    if self.beta1 < self.influence_exp_decay_stop:
                        self.beta1 = self.beta1 * self.influence_exp_decay_rate
                    else:
                        self.beta1 = self.influence_exp_decay_stop
                self.decay_stats_inf_t = t_env
        
        if self.curiosity_exp_decay:
            if t_env - self.decay_stats_cur_t >= self.curiosity_exp_decay_cycle:
                if self.curiosity_exp_decay_rate <= 1.0:
                    if self.beta2 > self.curiosity_exp_decay_stop:
                        self.beta2 = self.beta2 * self.curiosity_exp_decay_rate
                    else:
                        self.beta2 = self.curiosity_exp_decay_stop
                else:
                    if self.beta2 < self.curiosity_exp_decay_stop:
                        self.beta2 = self.beta2 * self.curiosity_exp_decay_rate
                    else:
                        self.beta2 = self.curiosity_exp_decay_stop
                self.decay_stats_cur_t = t_env
    
    def _update_target_networks(self, group: str) -> None:
        if group not in self._last_target_update:
            self._last_target_update[group] = 0
        
        if group not in self._training_steps:
            self._training_steps[group] = 0
        current_step = self._training_steps[group]
        
        if (current_step - self._last_target_update[group]) >= self.target_update_interval:
            self.target_predict_networks_os[group].load_state_dict(
                self.predict_networks_os[group].state_dict()
            )
            self.target_predict_networks_with_action[group].load_state_dict(
                self.predict_networks_with_action[group].state_dict()
            )
            self.target_predict_networks_without_action[group].load_state_dict(
                self.predict_networks_without_action[group].state_dict()
            )
            self._last_target_update[group] = current_step


@dataclass
class CoinConfig(AlgorithmConfig):
    """Configuration dataclass for the COIN algorithm."""
    
    mixing_embed_dim: int = 32
    delay_value: bool = True
    loss_function: str = "l2"
    
    beta1: float = 0.0005
    beta2: float = 0.0005
    beta: float = 0.5
    
    predictor_lr: float = 1e-3
    predict_net_dim: int = 64
    
    influence_exp_decay: bool = False
    influence_exp_decay_rate: float = 0.9
    influence_exp_decay_cycle: int = 100000
    influence_exp_decay_stop: float = 0.00001
    
    curiosity_exp_decay: bool = False
    curiosity_exp_decay_rate: float = 0.9
    curiosity_exp_decay_cycle: int = 50000
    curiosity_exp_decay_stop: float = 0.0
    
    target_update_interval: int = 200
    
    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Coin
    
    @staticmethod
    def supports_continuous_actions() -> bool:
        return False
    
    @staticmethod
    def supports_discrete_actions() -> bool:
        return True
    
    @staticmethod
    def on_policy() -> bool:
        return False

