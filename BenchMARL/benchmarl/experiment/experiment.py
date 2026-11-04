#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import copy
import importlib

import os
import pickle
import shutil
import time
import warnings
from collections import deque, OrderedDict
from dataclasses import dataclass, MISSING
from pathlib import Path

from typing import Any, Dict, List, Optional, Union

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector

from torchrl.envs import ParallelEnv, SerialEnv, TransformedEnv
from torchrl.envs.transforms import Compose
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.record.loggers import generate_exp_name
from tqdm import tqdm

from benchmarl.algorithms import IppoConfig, MappoConfig

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task, TaskClass
from benchmarl.experiment.callback import Callback, CallbackNotifier
from benchmarl.experiment.logger import Logger
from benchmarl.models import GnnConfig, SequenceModelConfig
from benchmarl.models.common import ModelConfig
from benchmarl.utils import (
    _add_rnn_transforms,
    _read_yaml_config,
    local_seed,
    seed_everything,
)

_has_hydra = importlib.util.find_spec("hydra") is not None
if _has_hydra:
    from hydra.core.hydra_config import HydraConfig


@dataclass
class ExperimentConfig:
    """
    Configuration class for experiments.
    This class acts as a schema for loading and validating yaml configurations.

    Parameters in this class aim to be agnostic of the algorithm, task or model used.
    To know their meaning, please check out the descriptions in ``benchmarl/conf/experiment/base_experiment.yaml``
    """

    sampling_device: str = MISSING
    train_device: str = MISSING
    buffer_device: str = MISSING

    share_policy_params: bool = MISSING
    prefer_continuous_actions: bool = MISSING
    collect_with_grad: bool = MISSING
    parallel_collection: bool = MISSING

    gamma: float = MISSING
    lr: float = MISSING
    adam_eps: float = MISSING
    adam_extra_kwargs: Dict[str, Any] = MISSING
    clip_grad_norm: bool = MISSING
    clip_grad_val: Optional[float] = MISSING

    soft_target_update: bool = MISSING
    polyak_tau: float = MISSING
    hard_target_update_frequency: int = MISSING

    exploration_eps_init: float = MISSING
    exploration_eps_end: float = MISSING
    exploration_anneal_frames: Optional[int] = MISSING

    max_n_iters: Optional[int] = MISSING
    max_n_frames: Optional[int] = MISSING

    on_policy_collected_frames_per_batch: int = MISSING
    on_policy_n_envs_per_worker: int = MISSING
    on_policy_n_minibatch_iters: int = MISSING
    on_policy_minibatch_size: int = MISSING

    off_policy_collected_frames_per_batch: int = MISSING
    off_policy_n_envs_per_worker: int = MISSING
    off_policy_n_optimizer_steps: int = MISSING
    off_policy_train_batch_size: int = MISSING
    off_policy_memory_size: int = MISSING
    off_policy_init_random_frames: int = MISSING
    off_policy_use_prioritized_replay_buffer: bool = MISSING
    off_policy_prb_alpha: float = MISSING
    off_policy_prb_beta: float = MISSING

    evaluation: bool = MISSING
    render: bool = MISSING
    evaluation_interval: int = MISSING
    evaluation_episodes: int = MISSING
    evaluation_deterministic_actions: bool = MISSING
    evaluation_static: bool = MISSING

    loggers: List[str] = MISSING
    project_name: str = MISSING
    wandb_extra_kwargs: Dict[str, Any] = MISSING
    create_json: bool = MISSING

    save_folder: Optional[str] = MISSING
    restore_file: Optional[str] = MISSING
    restore_map_location: Optional[Any] = MISSING
    checkpoint_interval: int = MISSING
    checkpoint_at_end: bool = MISSING
    keep_checkpoints_num: Optional[int] = MISSING
    exclude_buffer_from_checkpoint: bool = MISSING
    
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None

    def train_batch_size(self, on_policy: bool) -> int:
        """
        The batch size of tensors used for training

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        return (
            self.collected_frames_per_batch(on_policy)
            if on_policy
            else self.off_policy_train_batch_size
        )

    def train_minibatch_size(self, on_policy: bool) -> int:
        """
        The minibatch size of tensors used for training.
        On-policy algorithms are trained by splitting the train_batch_size (equal to the collected frames) into minibatches.
        Off-policy algorithms do not go through this process and thus have the ``train_minibatch_size==train_batch_size``

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        return (
            self.on_policy_minibatch_size
            if on_policy
            else self.train_batch_size(on_policy)
        )

    def n_optimizer_steps(self, on_policy: bool) -> int:
        """
        Number of times to loop over the training step per collection iteration.

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        return (
            self.on_policy_n_minibatch_iters
            if on_policy
            else self.off_policy_n_optimizer_steps
        )

    def replay_buffer_memory_size(self, on_policy: bool) -> int:
        """
        Size of the replay buffer memory in terms of frames

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        return (
            self.collected_frames_per_batch(on_policy)
            if on_policy
            else self.off_policy_memory_size
        )

    def collected_frames_per_batch(self, on_policy: bool) -> int:
        """
        Number of collected frames per collection iteration.

         Args:
             on_policy (bool): is the algorithms on_policy

        """
        return (
            self.on_policy_collected_frames_per_batch
            if on_policy
            else self.off_policy_collected_frames_per_batch
        )

    def n_envs_per_worker(self, on_policy: bool) -> int:
        """
        Number of environments used for collection

        - In vectorized environments, this will be the vectorized batch_size.
        - In other environments, this will be emulated by running them sequentially.

        Args:
            on_policy (bool): is the algorithms on_policy


        """
        return (
            self.on_policy_n_envs_per_worker
            if on_policy
            else self.off_policy_n_envs_per_worker
        )

    def get_max_n_frames(self, on_policy: bool) -> int:
        """
        Get the maximum number of frames collected before the experiment ends.

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        if self.max_n_frames is not None and self.max_n_iters is not None:
            return min(
                self.max_n_frames,
                self.max_n_iters * self.collected_frames_per_batch(on_policy),
            )
        elif self.max_n_frames is not None:
            return self.max_n_frames
        elif self.max_n_iters is not None:
            return self.max_n_iters * self.collected_frames_per_batch(on_policy)

    def get_max_n_iters(self, on_policy: bool) -> int:
        """
        Get the maximum number of experiment iterations before the experiment ends.

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        return -(
            -self.get_max_n_frames(on_policy)
            // self.collected_frames_per_batch(on_policy)
        )

    def get_exploration_anneal_frames(self, on_policy: bool):
        """
        Get the number of frames for exploration annealing.
        If self.exploration_anneal_frames is None this will be a third of the total frames to collect.

        Args:
            on_policy (bool): is the algorithms on_policy
        """
        return (
            (self.get_max_n_frames(on_policy) // 3)
            if self.exploration_anneal_frames is None
            else self.exploration_anneal_frames
        )

    @staticmethod
    def get_from_yaml(path: Optional[str] = None):
        """
        Load the experiment configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                ``benchmarl/conf/experiment/base_experiment.yaml``

        Returns:
            the loaded :class:`~benchmarl.experiment.ExperimentConfig`
        """
        if path is None:
            yaml_path = (
                Path(__file__).parent.parent
                / "conf"
                / "experiment"
                / "base_experiment.yaml"
            )
            return ExperimentConfig(**_read_yaml_config(str(yaml_path.resolve())))
        else:
            return ExperimentConfig(**_read_yaml_config(path))

    def validate(self, on_policy: bool):
        """
        Validates config.

        Args:
            on_policy (bool): is the algorithms on_policy

        """
        if (
            self.evaluation
            and self.evaluation_interval % self.collected_frames_per_batch(on_policy)
            != 0
        ):
            raise ValueError(
                f"evaluation_interval ({self.evaluation_interval}) "
                f"is not a multiple of the collected_frames_per_batch ({self.collected_frames_per_batch(on_policy)})"
            )
        if (
            self.checkpoint_interval != 0
            and self.checkpoint_interval % self.collected_frames_per_batch(on_policy)
            != 0
        ):
            raise ValueError(
                f"checkpoint_interval ({self.checkpoint_interval}) "
                f"is not a multiple of the collected_frames_per_batch ({self.collected_frames_per_batch(on_policy)})"
            )
        if self.keep_checkpoints_num is not None and self.keep_checkpoints_num <= 0:
            raise ValueError("keep_checkpoints_num must be greater than zero or null")
        if self.max_n_frames is None and self.max_n_iters is None:
            raise ValueError("max_n_frames and max_n_iters are both not set")
        if self.max_n_frames is not None and self.max_n_iters is not None:
            warnings.warn(
                f"max_n_frames and max_n_iters have both been set. The experiment will terminate after "
                f"{self.get_max_n_iters(on_policy)} iterations ({self.get_max_n_frames(on_policy)} frames)."
            )


class Experiment(CallbackNotifier):
    """
    Main experiment class in BenchMARL.

    Args:
        task (TaskClass): the task
        algorithm_config (AlgorithmConfig): the algorithm configuration
        model_config (ModelConfig): the policy model configuration
        seed (int): the seed for the experiment
        config (ExperimentConfig): The experiment config. Note that some of the parameters
            of this config may go un-consumed based on the provided algorithm or model config.
            For example, all parameters off-policy algorithm would not be used when running
            an experiment with an on-policy algorithm.
        critic_model_config (ModelConfig, optional): the policy model configuration.
            If None, it defaults to model_config
        callbacks (list of Callback, optional): callbacks for this experiment
    """

    def __init__(
        self,
        task: Union[Task, TaskClass],
        algorithm_config: AlgorithmConfig,
        model_config: ModelConfig,
        seed: int,
        config: ExperimentConfig,
        critic_model_config: Optional[ModelConfig] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        super().__init__(
            experiment=self, callbacks=callbacks if callbacks is not None else []
        )

        self.config = config

        if isinstance(task, Task):
            warnings.warn(
                "Call `.get_task()` or `.get_from_yaml()` on your task Enum before passing it to the experiment. "
                "If you do not do this, benchmarl will load the default task config from yaml."
            )
            task = task.get_task()
        self.task = task
        self.model_config = model_config
        self.critic_model_config = (
            critic_model_config
            if critic_model_config is not None
            else copy.deepcopy(model_config)
        )
        self.critic_model_config.is_critic = True

        self.algorithm_config = algorithm_config
        self.seed = seed

        self._setup()

        self.total_time = 0
        self.total_frames = 0
        self.n_iters_performed = 0
        self.mean_return = 0

        if self.config.restore_file is not None:
            self._load_experiment()

    @property
    def on_policy(self) -> bool:
        """Whether the algorithm has to be run on policy."""
        return self.algorithm_config.on_policy()

    def _setup(self):
        self.config.validate(self.on_policy)
        seed_everything(self.seed)
        self._perform_checks()
        self._set_action_type()
        self._setup_name()
        self._setup_task()
        self._setup_algorithm()
        self._setup_collector()
        self._setup_logger()
        self._on_setup()

    def _perform_checks(self):
        for config in (self.model_config, self.critic_model_config):
            if isinstance(config, SequenceModelConfig):
                for layer_config in config.model_configs[1:]:
                    if isinstance(layer_config, GnnConfig) and (
                        layer_config.position_key is not None
                        or layer_config.velocity_key is not None
                    ):
                        raise ValueError(
                            "GNNs reading position or velocity keys are currently only usable in first"
                            " layer of sequence models"
                        )

        if self.algorithm_config in (MappoConfig, IppoConfig):
            critic_model_config = self.critic_model_config
            if isinstance(critic_model_config, SequenceModelConfig):
                critic_model_config = self.critic_model_config.model_configs[0]
            if (
                isinstance(critic_model_config, GnnConfig)
                and critic_model_config.topology == "from_pos"
            ):
                raise ValueError(
                    "GNNs in PPO critics with topology 'from_pos' are currently not available, "
                    "see https://github.com/pytorch/rl/issues/2537"
                )

    def _set_action_type(self):
        if (
            self.task.supports_continuous_actions()
            and self.algorithm_config.supports_continuous_actions()
            and self.config.prefer_continuous_actions
        ):
            self.continuous_actions = True
        elif (
            self.task.supports_discrete_actions()
            and self.algorithm_config.supports_discrete_actions()
        ):
            self.continuous_actions = False
        elif (
            self.task.supports_continuous_actions()
            and self.algorithm_config.supports_continuous_actions()
        ):
            self.continuous_actions = True
        else:
            raise ValueError(
                f"Algorithm {self.algorithm_config} is not compatible"
                f" with the action space of task {self.task} "
            )

    def _setup_task(self):
        test_env = self.task.get_env_fun(
            num_envs=self.config.evaluation_episodes,
            continuous_actions=self.continuous_actions,
            seed=self.seed,
            device=self.config.sampling_device,
        )()
        env_func = self.task.get_env_fun(
            num_envs=self.config.n_envs_per_worker(self.on_policy),
            continuous_actions=self.continuous_actions,
            seed=self.seed,
            device=self.config.sampling_device,
        )

        transforms_env = self.task.get_env_transforms(test_env)
        transforms_training = transforms_env + [
            self.task.get_reward_sum_transform(test_env)
        ]
        transforms_env = Compose(*transforms_env)
        transforms_training = Compose(*transforms_training)

        # Initialize test env
        self.test_env = TransformedEnv(test_env, transforms_env.clone()).to(
            self.config.sampling_device
        )

        self.observation_spec = self.task.observation_spec(self.test_env)
        self.info_spec = self.task.info_spec(self.test_env)
        self.state_spec = self.task.state_spec(self.test_env)
        self.action_mask_spec = self.task.action_mask_spec(self.test_env)
        self.action_spec = self.task.action_spec(self.test_env)
        self.group_map = self.task.group_map(self.test_env)
        self.train_group_map = copy.deepcopy(self.group_map)
        self.max_steps = self.task.max_steps(self.test_env)

        # Add rnn transforms here so they do not show in the benchmarl specs
        if self.model_config.is_rnn:
            self.test_env = _add_rnn_transforms(
                lambda: self.test_env, self.group_map, self.model_config
            )()
            env_func = _add_rnn_transforms(env_func, self.group_map, self.model_config)

        # Initialize train env
        if self.test_env.batch_size == ():
            # If the environment is not vectorized, we simulate vectorization using parallel or serial environments
            env_class = (
                SerialEnv if not self.config.parallel_collection else ParallelEnv
            )
            self.env_func = lambda: TransformedEnv(
                env_class(self.config.n_envs_per_worker(self.on_policy), env_func),
                transforms_training.clone(),
            )
        else:
            # Otherwise it is already vectorized
            self.env_func = lambda: TransformedEnv(
                env_func(), transforms_training.clone()
            )

    def _setup_algorithm(self):
        self.algorithm = self.algorithm_config.get_algorithm(experiment=self)

        self.test_env = self.algorithm.process_env_fun(lambda: self.test_env)()
        self.env_func = self.algorithm.process_env_fun(self.env_func)

        self.replay_buffers = {
            group: self.algorithm.get_replay_buffer(
                group=group,
                transforms=self.task.get_replay_buffer_transforms(self.test_env, group),
            )
            for group in self.group_map.keys()
        }
        self.losses = {
            group: self.algorithm.get_loss_and_updater(group)[0]
            for group in self.group_map.keys()
        }
        self.target_updaters = {
            group: self.algorithm.get_loss_and_updater(group)[1]
            for group in self.group_map.keys()
        }
        actor_lr = self.config.actor_lr if self.config.actor_lr is not None else self.config.lr
        critic_lr = self.config.critic_lr if self.config.critic_lr is not None else self.config.lr
        
        lr_map = {
            "loss_objective": actor_lr,
            "loss_actor": actor_lr,
            "loss_critic": critic_lr,
            "loss_qvalue": critic_lr,
            "loss_critic_int": critic_lr,
        }
        
        for key, value in self.algorithm_config.__dict__.items():
            if key.endswith("_lr") and value is not None:
                loss_name = f"loss_{key[:-3]}"
                lr_map[loss_name] = value
        
        self.optimizers = {
            group: {
                loss_name: torch.optim.Adam(
                    params,
                    lr=lr_map.get(loss_name, self.config.lr),
                    eps=self.config.adam_eps,
                    **self.config.adam_extra_kwargs,
                )
                for loss_name, params in self.algorithm.get_parameters(group).items()
            }
            for group in self.group_map.keys()
        }

    def _setup_collector(self):
        self.policy = self.algorithm.get_policy_for_collection()

        self.group_policies = {}
        for group in self.group_map.keys():
            group_policy = self.policy.select_subsequence(out_keys=[(group, "action")])
            assert len(group_policy) == 1
            self.group_policies.update({group: group_policy[0]})

        if not self.config.collect_with_grad:
            self.collector = SyncDataCollector(
                self.env_func,
                self.policy,
                device=self.config.sampling_device,
                storing_device=self.config.sampling_device,
                frames_per_batch=self.config.collected_frames_per_batch(self.on_policy),
                total_frames=self.config.get_max_n_frames(self.on_policy),
                init_random_frames=(
                    self.config.off_policy_init_random_frames
                    if not self.on_policy
                    else 0
                ),
            
            )
        else:
            if self.config.off_policy_init_random_frames and not self.on_policy:
                raise TypeError(
                    "Collection via rollouts does not support initial random frames as of now."
                )
            self.rollout_env = self.env_func().to(self.config.sampling_device)

    def _setup_name(self):
        self.algorithm_name = self.algorithm_config.associated_class().__name__.lower()
        self.model_name = self.model_config.associated_class().__name__.lower()
        self.critic_model_name = (
            self.critic_model_config.associated_class().__name__.lower()
        )
        self.environment_name = self.task.env_name().lower()
        self.task_name = self.task.name.lower()
        self._checkpointed_files = deque([])

        if self.config.save_folder is not None:
            # If the user specified a folder for the experiment we use that
            save_folder = Path(self.config.save_folder)
        else:
            # Otherwise, if the user is restoring from a folder, we will save in the folder they are restoring from
            if self.config.restore_file is not None:
                save_folder = Path(
                    self.config.restore_file
                ).parent.parent.parent.resolve()
            # Otherwise, the user is not restoring and did not specify a save_folder so we save in the hydra directory
            # of the experiment or in the directory where the experiment was run (if hydra is not used)
            else:
                if _has_hydra and HydraConfig.initialized():
                    save_folder = Path(HydraConfig.get().runtime.output_dir)
                else:
                    save_folder = Path(os.getcwd()) / "wandb"

        if self.config.restore_file is None:
            name_parts = [self.algorithm_name]
            
            if hasattr(self.algorithm_config, 'intrinsic_lambda'):
                intrinsic_lambda = getattr(self.algorithm_config, 'intrinsic_lambda', None)
                if intrinsic_lambda is not None:
                    lambda_str = f"lambda{intrinsic_lambda:g}".replace('.', '_')
                    name_parts.append(lambda_str)
            
            if hasattr(self.algorithm_config, 'lambda_init'):
                lambda_init = getattr(self.algorithm_config, 'lambda_init', None)
                if lambda_init is not None:
                    lambda_str = f"lambda{lambda_init:g}".replace('.', '_')
                    name_parts.append(lambda_str)
            
            name_parts.extend([self.task_name, self.model_name])
            
            base_name = "_".join(name_parts)
            self.name = generate_exp_name(base_name, "")
            self.folder_name = save_folder / self.name

        else:
            # If restoring, we use the name of the previous experiment
            self.name = Path(self.config.restore_file).parent.parent.resolve().name
            self.folder_name = save_folder / self.name

        self.folder_name.mkdir(parents=True, exist_ok=True)
        with open(self.folder_name / "config.pkl", "wb") as f:
            pickle.dump(self.task, f)
            pickle.dump(self.task.config if self.task.config is not None else {}, f)
            pickle.dump(self.algorithm_config, f)
            pickle.dump(self.model_config, f)
            pickle.dump(self.seed, f)
            pickle.dump(self.config, f)
            pickle.dump(self.critic_model_config, f)
            pickle.dump(self.callbacks, f)

    def _setup_logger(self):
        hparams_kwargs = {
            "critic_model_name": self.critic_model_name,
            "experiment_config": self.config.__dict__,
            "algorithm_config": self.algorithm_config.__dict__,
            "model_config": self.model_config.__dict__,
            "critic_model_config": self.critic_model_config.__dict__,
            "task_config": self.task.config,
            "continuous_actions": self.continuous_actions,
            "on_policy": self.on_policy,
            "algorithm_name": self.algorithm_name,
            "model_name": self.model_name,
            "task_name": self.task_name,
            "environment_name": self.environment_name,
            "seed": self.seed,
        }
        self.logger = Logger(
            experiment_name=self.name,
            folder_name=str(self.folder_name),
            experiment_config=self.config,
            algorithm_name=self.algorithm_name,
            model_name=self.model_name,
            environment_name=self.environment_name,
            task_name=self.task_name,
            group_map=self.group_map,
            seed=self.seed,
            project_name=self.config.project_name,
            wandb_extra_kwargs={
                **self.config.wandb_extra_kwargs,
                "config": hparams_kwargs,
            },
        )
        self.logger.log_hparams(**hparams_kwargs)

    def run(self):
        """Run the experiment until completion."""
        try:
            seed_everything(self.seed)
            torch.cuda.empty_cache()
            self._collection_loop()
        except KeyboardInterrupt as interrupt:
            print("\n\nExperiment was closed gracefully\n\n")
            self.close()
            raise interrupt
        except Exception as err:
            print("\n\nExperiment failed and is closing gracefully\n\n")
            self.close()
            raise err

    def evaluate(self):
        """Run just the evaluation loop once."""
        seed_everything(self.seed)
        self._evaluation_loop()
        self.logger.commit()
        print(
            f"Evaluation results logged to loggers={self.config.loggers}"
            f"{' and to a json file in the experiment folder.' if self.config.create_json else ''}"
        )

    def _collection_loop(self):
        pbar = tqdm(
            initial=self.n_iters_performed,
            total=self.config.get_max_n_iters(self.on_policy),
        )

        if not self.config.collect_with_grad:
            iterator = iter(self.collector)
        else:
            reset_batch = self.rollout_env.reset()

        # Training/collection iterations
        for _ in range(
            self.n_iters_performed, self.config.get_max_n_iters(self.on_policy)
        ):
            iteration_start = time.time()
            if not self.config.collect_with_grad:
                batch = next(iterator)
            else:
                with set_exploration_type(ExplorationType.RANDOM):
                    batch = self.rollout_env.rollout(
                        max_steps=-(
                            -self.config.collected_frames_per_batch(self.on_policy)
                            // self.rollout_env.batch_size.numel()
                        ),
                        policy=self.policy,
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                        auto_reset=False,
                        tensordict=reset_batch,
                    )
                    reset_batch = step_mdp(
                        batch[..., -1],
                        reward_keys=self.rollout_env.reward_keys,
                        action_keys=self.rollout_env.action_keys,
                        done_keys=self.rollout_env.done_keys,
                    )

            # Logging collection
            collection_time = time.time() - iteration_start
            current_frames = batch.numel()
            self.total_frames += current_frames
            self.mean_return = self.logger.log_collection(
                batch,
                total_frames=self.total_frames,
                task=self.task,
                step=self.n_iters_performed,
            )
            pbar.set_description(f"mean return = {self.mean_return}", refresh=False)

            # Callback
            self._on_batch_collected(batch)
            batch = batch.detach()

            # Loop over groups
            training_start = time.time()
            for group in self.train_group_map.keys():
                group_batch = batch.exclude(*self._get_excluded_keys(group)).to(
                    self.config.train_device
                )
                group_batch = self.algorithm.process_batch(group, group_batch)
                if not self.algorithm.has_rnn:
                    group_batch = group_batch.reshape(-1)

                group_buffer = self.replay_buffers[group]
                group_buffer.extend(group_batch.to(group_buffer.storage.device))

                training_tds = []
                for _ in range(self.config.n_optimizer_steps(self.on_policy)):
                    for _ in range(
                        -(
                            -self.config.train_batch_size(self.on_policy)
                            // self.config.train_minibatch_size(self.on_policy)
                        )
                    ):
                        training_tds.append(self._optimizer_loop(group))
                training_td = torch.stack(training_tds)
                self.logger.log_training(
                    group, training_td, step=self.n_iters_performed
                )

                # Callback
                self._on_train_end(training_td, group)

                # Exploration update
                if isinstance(self.group_policies[group], TensorDictSequential):
                    explore_layer = self.group_policies[group][-1]
                else:
                    explore_layer = self.group_policies[group]
                if hasattr(explore_layer, "step"):  # Step exploration annealing
                    explore_layer.step(current_frames)

            if hasattr(self.algorithm, 'update_z0_from_pending_rollouts'):
                self.algorithm.update_z0_from_pending_rollouts()

            # Update policy in collector
            if not self.config.collect_with_grad:
                self.collector.update_policy_weights_()

            # Training timer
            training_time = time.time() - training_start

            # Evaluation
            if (
                self.config.evaluation
                and (
                    self.total_frames % self.config.evaluation_interval == 0
                    or self.n_iters_performed == 0
                )
                and (len(self.config.loggers) or self.config.create_json)
            ):
                self._evaluation_loop()

            # End of step
            iteration_time = time.time() - iteration_start
            self.total_time += iteration_time
            self.logger.log(
                {
                    "timers/collection_time": collection_time,
                    "timers/training_time": training_time,
                    "timers/iteration_time": iteration_time,
                    "timers/total_time": self.total_time,
                    "counters/current_frames": current_frames,
                    "counters/total_frames": self.total_frames,
                    "counters/iter": self.n_iters_performed,
                },
                step=self.n_iters_performed,
            )
            self.n_iters_performed += 1
            self.logger.commit()
            if (
                self.config.checkpoint_interval > 0
                and self.total_frames % self.config.checkpoint_interval == 0
            ):
                self._save_experiment()
            pbar.update()

        if self.config.checkpoint_at_end:
            self._save_experiment()
        self.close()

    def close(self):
        """Close the experiment."""
        if not self.config.collect_with_grad:
            self.collector.shutdown()
        else:
            self.rollout_env.close()
        self.test_env.close()
        self.logger.finish()

        for buffer in self.replay_buffers.values():
            if hasattr(buffer.storage, "scratch_dir"):
                shutil.rmtree(buffer.storage.scratch_dir, ignore_errors=False)

    def _get_excluded_keys(self, group: str):
        excluded_keys = []
        for other_group in self.group_map.keys():
            if other_group != group:
                excluded_keys += [other_group, ("next", other_group)]
        excluded_keys += ["info", (group, "info"), ("next", group, "info")]
        return excluded_keys

    def _optimizer_loop(self, group: str) -> TensorDictBase:
        subdata = self.replay_buffers[group].sample().to(self.config.train_device)
        loss_vals = self.losses[group](subdata)
        training_td = loss_vals.detach()
        loss_vals = self.algorithm.process_loss_vals(group, loss_vals)

        for key, value in loss_vals.items():
            if key not in training_td.keys():
                training_td.set(key, value.detach())

        for loss_name, loss_value in loss_vals.items():
            if loss_name in self.optimizers[group].keys():
                optimizer = self.optimizers[group][loss_name]

                loss_value.backward()

                grad_norm = self._grad_clip(optimizer)

                training_td.set(
                    f"grad_norm_{loss_name}",
                    torch.tensor(grad_norm, device=self.config.train_device),
                )

                optimizer.step()
                optimizer.zero_grad()
        self.replay_buffers[group].update_tensordict_priority(subdata)
        if self.target_updaters[group] is not None:
            self.target_updaters[group].step()

        callback_loss = self._on_train_step(subdata, group)
        if callback_loss is not None:
            training_td.update(callback_loss)

        return training_td

    def _grad_clip(self, optimizer: torch.optim.Optimizer) -> float:
        params = []
        for param_group in optimizer.param_groups:
            params += param_group["params"]

        if self.config.clip_grad_norm and self.config.clip_grad_val is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                params, self.config.clip_grad_val
            )
        else:
            norm_type = 2.0
            norms = [
                torch.linalg.vector_norm(p.grad, norm_type)
                for p in params
                if p.grad is not None
            ]
            total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
            if self.config.clip_grad_val is not None:
                torch.nn.utils.clip_grad_value_(params, self.config.clip_grad_val)

        return float(total_norm)

    @local_seed()
    @torch.no_grad()
    def _evaluation_loop(self):
        if self.config.evaluation_static:
            seed_everything(self.seed)
            try:
                self.test_env.set_seed(self.seed)
            except NotImplementedError:
                warnings.warn(
                    "`experiment.evaluation_static` set to true but the environment does not allow to set seeds."
                    "Static evaluation is not guaranteed."
                )
        evaluation_start = time.time()
        with set_exploration_type(
            ExplorationType.DETERMINISTIC
            if self.config.evaluation_deterministic_actions
            else ExplorationType.RANDOM
        ):
            if self.task.has_render(self.test_env) and self.config.render:
                video_frames = []
                reward_history = []
                intrinsic_reward_history = []
                has_intrinsic = (hasattr(self.algorithm, 'intrinsic_manager') and self.algorithm.intrinsic_manager is not None)
                max_steps = self.max_steps
                
                intrinsic_buffer = {
                    'features_curr': [],
                    'features_next': [],
                    'actions': [],
                    'dones': [],
                    'group': None
                }

                def callback(env, td):
                    is_vectorized = self.test_env.batch_size != ()
                    env_idx = 0
                    
                    reward = None
                    for group in self.group_map.keys():
                        group_reward = td.get(("next", group, "reward"), None)
                        if group_reward is not None:
                            if is_vectorized:
                                if group_reward.ndim >= 2:
                                    reward = group_reward[env_idx, :].mean().item()
                                elif group_reward.ndim == 1:
                                    reward = group_reward[env_idx].item()
                                else:
                                    reward = group_reward.item()
                            else:
                                if group_reward.ndim > 0:
                                    reward = group_reward.mean().item()
                                else:
                                    reward = group_reward.item()
                            break
                    if reward is None:
                        global_reward = td.get(("next", "reward"), None)
                        if global_reward is not None:
                            if is_vectorized:
                                if global_reward.ndim >= 1:
                                    reward = global_reward[env_idx].item()
                                else:
                                    reward = global_reward.item()
                            else:
                                reward = global_reward.item() if global_reward.numel() == 1 else global_reward.mean().item()
                    
                    if reward is not None:
                        reward_history.append(reward)
                    
                    intrinsic_reward = None
                    
                    if intrinsic_reward is None and has_intrinsic:
                        cfg = getattr(self.algorithm, 'experiment', None)
                        extrinsic_only = False
                        if cfg is not None:
                            cfg = getattr(cfg, 'algorithm_config', None)
                            if cfg is not None:
                                extrinsic_only = getattr(cfg, 'extrinsic_only', False)
                        
                        if not extrinsic_only:
                            for group in self.group_map.keys():
                                group_intrinsic = td.get((group, "intrinsic_reward"), None)
                                if group_intrinsic is None:
                                    group_intrinsic = td.get(("next", group, "intrinsic_reward"), None)
                                if group_intrinsic is not None:
                                    if is_vectorized:
                                        if group_intrinsic.ndim >= 2:
                                            intrinsic_reward = group_intrinsic[env_idx, :].mean().item()
                                        elif group_intrinsic.ndim == 1:
                                            intrinsic_reward = group_intrinsic[env_idx].item()
                                        else:
                                            intrinsic_reward = group_intrinsic.item()
                                    else:
                                        if group_intrinsic.ndim > 0:
                                            intrinsic_reward = group_intrinsic.mean().item()
                                        else:
                                            intrinsic_reward = group_intrinsic.item()
                                    break
                            
                            if intrinsic_reward is None:
                                try:
                                    for group in self.group_map.keys():
                                        obs = td.get((group, "observation"), None)
                                        next_obs = td.get(("next", group, "observation"), None)
                                        actions = td.get((group, "action"), None)
                                        done_tensor = td.get(("next", group, "done"), None) or td.get(("next", "done"), None)
                                        
                                        if obs is not None and next_obs is not None and actions is not None and done_tensor is not None:
                                            if is_vectorized:
                                                if obs.ndim >= 1:
                                                    obs = obs[env_idx, ...] if obs.ndim > 1 else obs[env_idx]
                                                if next_obs.ndim >= 1:
                                                    next_obs = next_obs[env_idx, ...] if next_obs.ndim > 1 else next_obs[env_idx]
                                                if actions.ndim >= 1:
                                                    actions = actions[env_idx, ...] if actions.ndim > 1 else actions[env_idx]
                                                if done_tensor.ndim >= 1:
                                                    done_tensor = done_tensor[env_idx] if done_tensor.ndim == 1 else done_tensor[env_idx, ...]
                                            
                                            if obs.ndim < 2:
                                                continue
                                            feature_dim = 4 if getattr(self.algorithm, "agent_feature_up_to_velocity", True) else 2
                                            feat_curr = obs[..., :feature_dim]
                                            feat_next = next_obs[..., :feature_dim]
                                            
                                            if feat_curr.ndim == 1:
                                                feat_curr = feat_curr.unsqueeze(0).unsqueeze(0)
                                                feat_next = feat_next.unsqueeze(0).unsqueeze(0)
                                            elif feat_curr.ndim == 2:
                                                if feat_curr.shape[0] == len(self.group_map[group]):
                                                    feat_curr = feat_curr.unsqueeze(0)
                                                    feat_next = feat_next.unsqueeze(0)
                                                else:
                                                    feat_curr = feat_curr.unsqueeze(0).unsqueeze(0)
                                                    feat_next = feat_next.unsqueeze(0).unsqueeze(0)
                                            
                                            if intrinsic_buffer['group'] is None:
                                                intrinsic_buffer['group'] = group
                                            
                                            if intrinsic_buffer['group'] == group:
                                                intrinsic_buffer['features_curr'].append(feat_curr)
                                                intrinsic_buffer['features_next'].append(feat_next)
                                                intrinsic_buffer['actions'].append(actions)
                                                intrinsic_buffer['dones'].append(done_tensor)
                                                
                                                max_buffer_size = 50
                                                if len(intrinsic_buffer['features_curr']) > max_buffer_size:
                                                    intrinsic_buffer['features_curr'] = intrinsic_buffer['features_curr'][-max_buffer_size:]
                                                    intrinsic_buffer['features_next'] = intrinsic_buffer['features_next'][-max_buffer_size:]
                                                    intrinsic_buffer['actions'] = intrinsic_buffer['actions'][-max_buffer_size:]
                                                    intrinsic_buffer['dones'] = intrinsic_buffer['dones'][-max_buffer_size:]
                                                
                                                if len(intrinsic_buffer['features_curr']) >= 2:
                                                    feat_curr_batch = torch.stack(intrinsic_buffer['features_curr'], dim=0)
                                                    feat_next_batch = torch.stack(intrinsic_buffer['features_next'], dim=0)
                                                    actions_batch = torch.stack(intrinsic_buffer['actions'], dim=0)
                                                    dones_batch = torch.stack(intrinsic_buffer['dones'], dim=0)
                                                    
                                                    T = feat_curr_batch.shape[0]
                                                    num_agents = len(self.group_map[group])
                                                    
                                                    if feat_curr_batch.ndim == 3:
                                                        feat_curr_batch = feat_curr_batch.unsqueeze(1)  # (T, 1, agent, feat)
                                                        feat_next_batch = feat_next_batch.unsqueeze(1)
                                                    elif feat_curr_batch.ndim == 4:
                                                        if feat_curr_batch.shape[2] == 1:
                                                            feat_curr_batch = feat_curr_batch.expand(-1, -1, num_agents, -1)
                                                            feat_next_batch = feat_next_batch.expand(-1, -1, num_agents, -1)
                                                    else:
                                                        pass
                                                    
                                                    if feat_curr_batch.shape[1] == 0 or feat_curr_batch.shape[1] != dones_batch.shape[1] if dones_batch.ndim >= 2 else True:
                                                        if dones_batch.ndim >= 2:
                                                            num_envs = dones_batch.shape[1]
                                                        else:
                                                            num_envs = 1
                                                        if feat_curr_batch.shape[1] != num_envs:
                                                            if feat_curr_batch.shape[1] == 1:
                                                                feat_curr_batch = feat_curr_batch.expand(-1, num_envs, -1, -1)
                                                                feat_next_batch = feat_next_batch.expand(-1, num_envs, -1, -1)
                                                    
                                                    if dones_batch.ndim == 1:
                                                        dones_batch = dones_batch.unsqueeze(1).unsqueeze(-1).expand(-1, -1, num_agents)
                                                    elif dones_batch.ndim == 2:
                                                        dones_batch = dones_batch.unsqueeze(-1).expand(-1, -1, num_agents)
                                                    elif dones_batch.ndim == 3 and dones_batch.shape[-1] == 1:
                                                        dones_batch = dones_batch.expand(-1, -1, num_agents)
                                                    elif dones_batch.ndim == 3 and dones_batch.shape[-1] != num_agents:
                                                        if dones_batch.shape[-1] == 1:
                                                            dones_batch = dones_batch.expand(-1, -1, num_agents)
                                                    
                                                    if actions_batch.ndim == 2:
                                                        actions_batch = actions_batch.unsqueeze(1)
                                                    elif actions_batch.ndim == 3 and actions_batch.shape[1] == 1:
                                                        pass
                                                    elif actions_batch.ndim == 3 and actions_batch.shape[1] != feat_curr_batch.shape[1]:
                                                        if actions_batch.shape[1] == 1:
                                                            num_envs = feat_curr_batch.shape[1]
                                                            actions_batch = actions_batch.expand(-1, num_envs, -1)
                                                    
                                                    with torch.no_grad():
                                                        intrinsic_rewards, _ = self.algorithm.intrinsic_manager.compute_intrinsic_rewards(
                                                            group, feat_curr_batch, feat_next_batch, actions_batch, dones_batch, update_networks=False
                                                        )
                                                        if intrinsic_rewards.ndim == 3:
                                                            intrinsic_reward = intrinsic_rewards[-1, 0, :].mean().item()
                                                        elif intrinsic_rewards.ndim == 2:
                                                            intrinsic_reward = intrinsic_rewards[-1].mean().item()
                                                        else:
                                                            intrinsic_reward = intrinsic_rewards[-1].item()
                                                if intrinsic_reward is not None:
                                                    break
                                except Exception as e:
                                    if len(intrinsic_reward_history) < 3:
                                        import traceback
                                        print(f"Intrinsic reward computation failed (step {len(reward_history)}): {e}")
                                        if len(intrinsic_reward_history) == 0:
                                            print(f"Traceback: {traceback.format_exc()}")
                                    pass
                    
                    if intrinsic_reward is not None:
                        intrinsic_reward_history.append(intrinsic_reward)
                    
                    try:
                        frame = env.render(mode="rgb_array")
                    except TypeError:
                        frame = env.render()
                    
                    if is_vectorized:
                        if isinstance(frame, (list, tuple)):
                            frame = frame[env_idx] if env_idx < len(frame) else frame[0]
                        elif hasattr(frame, 'ndim') and frame.ndim == 4:
                            if isinstance(frame, torch.Tensor):
                                frame = frame[env_idx].cpu().numpy() if frame.is_cuda else frame[env_idx].numpy()
                            else:
                                frame = frame[env_idx]
                    
                    if len(reward_history) > 0 or len(intrinsic_reward_history) > 0:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        import numpy as np
                        
                        n_plots = 2 if has_intrinsic else 1
                        fig_height = frame.shape[0] / frame.shape[1] * 4
                        if n_plots == 2:
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, fig_height * 2), dpi=frame.shape[0] / 4)
                            
                            steps_ext = list(range(len(reward_history))) if len(reward_history) > 0 else []
                            if len(steps_ext) > 0:
                                ax1.plot(steps_ext, reward_history, 'b-', linewidth=2, label='Extrinsic')
                            ax1.set_xlim(0, max_steps)
                            ax1.set_xlabel('Step', fontsize=10)
                            ax1.set_ylabel('Reward', fontsize=10)
                            ax1.set_title('Extrinsic Reward History', fontsize=12)
                            ax1.grid(True, alpha=0.3)
                            ax1.tick_params(labelsize=8)
                            if len(reward_history) > 0:
                                ax1.legend()
                            
                            steps_int = list(range(len(intrinsic_reward_history))) if len(intrinsic_reward_history) > 0 else []
                            if len(steps_int) > 0:
                                ax2.plot(steps_int, intrinsic_reward_history, 'b-', linewidth=2, label='Intrinsic')
                            ax2.set_xlim(0, max_steps)
                            ax2.set_xlabel('Step', fontsize=10)
                            ax2.set_ylabel('Reward', fontsize=10)
                            ax2.set_title('Intrinsic Reward History', fontsize=12)
                            ax2.grid(True, alpha=0.3)
                            ax2.tick_params(labelsize=8)
                            if len(intrinsic_reward_history) > 0:
                                ax2.legend()
                        else:
                            fig, ax1 = plt.subplots(figsize=(4, fig_height), dpi=frame.shape[0] / 4)
                            steps = list(range(len(reward_history))) if len(reward_history) > 0 else []
                            if len(steps) > 0:
                                ax1.plot(steps, reward_history, 'r-', linewidth=2, label='Extrinsic')
                            ax1.set_xlim(0, max_steps)
                            ax1.set_xlabel('Step', fontsize=10)
                            ax1.set_ylabel('Reward', fontsize=10)
                            ax1.set_title('Extrinsic Reward History', fontsize=12)
                            ax1.grid(True, alpha=0.3)
                            ax1.tick_params(labelsize=8)
                            if len(reward_history) > 0:
                                ax1.legend()
                        
                        plt.tight_layout()
                        
                        fig.canvas.draw()
                        buf = fig.canvas.buffer_rgba()
                        plot_array = np.asarray(buf)
                        width, height = fig.canvas.get_width_height()
                        plot_array = plot_array.reshape((height, width, 4))
                        plot_array = plot_array[:, :, :3]
                        plt.close(fig)
                        
                        frame_height = frame.shape[0]
                        plot_height = plot_array.shape[0]
                        
                        if plot_height != frame_height:
                            from scipy.ndimage import zoom
                            zoom_factor = frame_height / plot_height
                            plot_array = zoom(plot_array, (zoom_factor, 1.0, 1.0), order=1)
                        
                        combined = np.concatenate([frame, plot_array], axis=1)
                        video_frames.append(combined)
                    else:
                        video_frames.append(frame)

            else:
                video_frames = None
                callback = None

            if self.test_env.batch_size == ():
                rollouts = []
                for eval_episode in range(self.config.evaluation_episodes):
                    rollouts.append(
                        self.test_env.rollout(
                            max_steps=self.max_steps,
                            policy=self.policy,
                            callback=callback if eval_episode == 0 else None,
                            auto_cast_to_device=True,
                            break_when_any_done=True,
                        )
                    )
            else:
                rollouts = self.test_env.rollout(
                    max_steps=self.max_steps,
                    policy=self.policy,
                    callback=callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )
                rollouts = list(rollouts.unbind(0))
        evaluation_time = time.time() - evaluation_start
        self.logger.log(
            {"timers/evaluation_time": evaluation_time}, step=self.n_iters_performed
        )
        self.logger.log_evaluation(
            rollouts,
            video_frames=video_frames,
            step=self.n_iters_performed,
            total_frames=self.total_frames,
        )
        # Callback
        self._on_evaluation_end(rollouts)

    # Saving experiment state
    def state_dict(self) -> OrderedDict:
        """Get the state_dict for the experiment."""
        state = OrderedDict(
            total_time=self.total_time,
            total_frames=self.total_frames,
            n_iters_performed=self.n_iters_performed,
            mean_return=self.mean_return,
        )
        state_dict = OrderedDict(
            state=state,
            **{f"loss_{k}": item.state_dict() for k, item in self.losses.items()},
            **{
                f"buffer_{k}": item.state_dict()
                if len(item) and not self.config.exclude_buffer_from_checkpoint
                else None
                for k, item in self.replay_buffers.items()
            },
        )
        if not self.config.collect_with_grad:
            state_dict.update({"collector": self.collector.state_dict()})
        self._on_state_dict(state_dict)
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load the state_dict for the experiment.

        Args:
            state_dict (dict): the state dict

        """
        for group in self.group_map.keys():
            self.losses[group].load_state_dict(state_dict[f"loss_{group}"])
            if state_dict[f"buffer_{group}"] is not None:
                self.replay_buffers[group].load_state_dict(
                    state_dict[f"buffer_{group}"]
                )
        if not self.config.collect_with_grad:
            self.collector.load_state_dict(state_dict["collector"])
        self.total_time = state_dict["state"]["total_time"]
        self.total_frames = state_dict["state"]["total_frames"]
        self.n_iters_performed = state_dict["state"]["n_iters_performed"]
        self.mean_return = state_dict["state"]["mean_return"]
        self._on_load_state_dict(state_dict)

    def _save_experiment(self) -> None:
        """Checkpoint trainer"""
        if self.config.keep_checkpoints_num is not None:
            while len(self._checkpointed_files) >= self.config.keep_checkpoints_num:
                file_to_delete = self._checkpointed_files.popleft()
                file_to_delete.unlink(missing_ok=False)

        checkpoint_folder = self.folder_name / "checkpoints"
        checkpoint_folder.mkdir(parents=False, exist_ok=True)
        checkpoint_file = checkpoint_folder / f"checkpoint_{self.total_frames}.pt"
        torch.save(self.state_dict(), checkpoint_file)
        self._checkpointed_files.append(checkpoint_file)

    def _load_experiment(self) -> Experiment:
        """Load trainer from checkpoint"""
        loaded_dict: OrderedDict = torch.load(
            self.config.restore_file, map_location=self.config.restore_map_location
        )
        self.load_state_dict(loaded_dict)
        return self

    @staticmethod
    def reload_from_file(
        restore_file: str, experiment_patch: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """
        Restores the experiment from the checkpoint file.

        This method expects the same folder structure created when an experiment is run.
        The checkpoint file (``restore_file``) is in the checkpoints directory and a config.pkl file is
        present a level above at restore_file/../../config.pkl

        Args:
            restore_file (str): The checkpoint file (.pt) of the experiment reload.
            experiment_patch (Optional[Dict[str, Any]]): The patch to apply to the experiment config.

        Returns:
            The reloaded experiment.

        """
        experiment_folder = Path(restore_file).parent.parent.resolve()
        config_file = experiment_folder / "config.pkl"
        if not os.path.exists(config_file):
            raise ValueError("config.pkl file not found in experiment folder.")
        with open(config_file, "rb") as f:
            task = pickle.load(f)
            task_config = pickle.load(f)
            algorithm_config = pickle.load(f)
            model_config = pickle.load(f)
            seed = pickle.load(f)
            experiment_config = pickle.load(f)
            critic_model_config = pickle.load(f)
            callbacks = pickle.load(f)
        task.config = task_config
        experiment_config.restore_file = restore_file
        if experiment_patch is not None:
            for key, value in experiment_patch.items():
                if not hasattr(experiment_config, key):
                    raise ValueError(f"Experiment config does not have attribute {key}")
                setattr(experiment_config, key, value)
        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            seed=seed,
            config=experiment_config,
            callbacks=callbacks,
            critic_model_config=critic_model_config,
        )
        print(f"\nReloaded experiment {experiment.name} from {restore_file}.")
        return experiment
