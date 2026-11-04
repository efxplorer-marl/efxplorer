#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    final_reward: float = MISSING
    agent_collision_penalty: float = MISSING
    passage_collision_penalty: float = MISSING
    obstacle_collision_penalty: float = MISSING
    pos_shaping_factor: float = MISSING
