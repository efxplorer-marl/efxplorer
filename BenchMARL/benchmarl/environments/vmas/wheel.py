#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    line_length: float = MISSING
    line_mass: float = MISSING
    desired_velocity: float = MISSING
    x_semidim: float = 1.0  # World boundary in x direction (same as balance task)
    y_semidim: float = 1.0  # World boundary in y direction (same as balance task)
    boundary_collision_penalty: float = 0.0  # Penalty for hitting world boundaries (default: 0, no penalty)
    min_collision_distance: float = 0.005  # Minimum distance to boundary to trigger collision
