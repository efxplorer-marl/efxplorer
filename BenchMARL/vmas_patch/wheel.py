#  Copyright (c) ProrokLab.
#
#  This source code is licensed under GPLv3; see the LICENSE file
#  in this vmas_patch/ directory.
#  Modified by the EFXPLORER authors, 2026.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Line, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.line_length = kwargs.pop("line_length", 2)
        line_mass = kwargs.pop("line_mass", 30)
        self.desired_velocity = kwargs.pop("desired_velocity", 0.05)
        self.x_semidim = kwargs.pop("x_semidim", 0.8)
        self.y_semidim = kwargs.pop("y_semidim", 0.8)
        self.boundary_collision_penalty = kwargs.pop("boundary_collision_penalty", 0.0)
        self.min_collision_distance = kwargs.pop("min_collision_distance", 0.005)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        world = World(batch_dim, device, x_semidim=self.x_semidim, y_semidim=self.y_semidim)
        for i in range(n_agents):
            agent = Agent(name=f"agent_{i}", u_multiplier=0.6, shape=Sphere(0.03))
            world.add_agent(agent)
        self.line = Landmark(
            name="line",
            collide=True,
            rotatable=True,
            shape=Line(length=self.line_length),
            mass=line_mass,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)
        center = Landmark(
            name="center",
            shape=Sphere(radius=0.02),
            collide=False,
            color=Color.BLACK,
        )
        world.add_landmark(center)

        self.boundary_collision_rew = torch.zeros(batch_dim, device=device)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

        self.line.set_rot(
            torch.zeros(
                (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                -torch.pi / 2,
                torch.pi / 2,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            ang_vel_abs = self.line.state.ang_vel.abs()
            if ang_vel_abs.ndim > 1:
                ang_vel_abs = ang_vel_abs.squeeze(-1)
            diff = ang_vel_abs - self.desired_velocity
            normalized_diff_sq = (diff / self.desired_velocity) ** 2
            self.rew = 1.0 - normalized_diff_sq
            
            self.boundary_collision_rew[:] = 0
            if self.boundary_collision_penalty != 0:
                agent_radius = 0.03
                
                for a in self.world.agents:
                    x_dist_from_center_to_boundary = self.x_semidim - a.state.pos[:, 0].abs()
                    x_dist_from_edge_to_boundary = x_dist_from_center_to_boundary - agent_radius
                    x_near_boundary = x_dist_from_edge_to_boundary <= self.min_collision_distance
                    
                    y_dist_from_center_to_boundary = self.y_semidim - a.state.pos[:, 1].abs()
                    y_dist_from_edge_to_boundary = y_dist_from_center_to_boundary - agent_radius
                    y_near_boundary = y_dist_from_edge_to_boundary <= self.min_collision_distance
                    
                    self.boundary_collision_rew[x_near_boundary | y_near_boundary] += self.boundary_collision_penalty

        reward = self.rew + self.boundary_collision_rew
        if reward.ndim > 1:
            reward = reward.squeeze(-1)
        return reward

    def observation(self, agent: Agent):
        line_end_1 = torch.cat(
            [
                (self.line_length / 2) * torch.cos(self.line.state.rot),
                (self.line_length / 2) * torch.sin(self.line.state.rot),
            ],
            dim=1,
        )
        line_end_2 = -line_end_1

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.line.state.pos - agent.state.pos,
                line_end_1 - agent.state.pos,
                line_end_2 - agent.state.pos,
                torch.cos(self.line.state.rot),
                torch.sin(self.line.state.rot),
                self.line.state.ang_vel.abs(),
                (self.line.state.ang_vel.abs() - self.desired_velocity).abs(),
            ],
            dim=-1,
        )

    def info(self, agent: Agent):
        return {
            "boundary_collision_rew": self.boundary_collision_rew,
        }


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions is True, "Heuristic for continuous actions only"

        index_line_extrema = 6

        pos_agent = observation[:, :2]
        pos_end2_agent = observation[:, index_line_extrema + 2 : index_line_extrema + 4]

        pos_end2 = pos_end2_agent + pos_agent

        pos_end2_shifted = TorchUtils.rotate_vector(
            pos_end2,
            torch.tensor(torch.pi / 4, device=observation.device).expand(
                pos_end2.shape[0]
            ),
        )

        pos_end2_shifted_agent = pos_end2_shifted - pos_agent

        action_agent = torch.clamp(
            pos_end2_shifted_agent,
            min=-u_range,
            max=u_range,
        )

        return action_agent


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        desired_velocity=0.05,
        n_agents=4,
        line_length=2,
        line_mass=30,
    )
