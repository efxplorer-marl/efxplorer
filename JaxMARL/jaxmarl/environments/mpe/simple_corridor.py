import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from jaxmarl.environments.spaces import Box


class SimpleCorridorMPE(SimpleMPE):
    def __init__(
        self,
        num_agents=5,
        num_landmarks=None,
        local_ratio=0.5,
        action_type=DISCRETE_ACT,
        agent_size=0.15,
        landmark_size=0.05,
        agent_wise_distance=False,
        use_goal_conditioned_sd=True,
        sparse_reward=False,
        spawn_mode="garden",
        fixed_spawn_y=True,
        fixed_spawn_x=True,
        hard_collision=True,
        hard_collision_iterations=10,
        collision_elasticity=0.3,
        agent_collision_coef=1.5,
        wall_collision_coef=0.1,
        corridor_width=0.8,
        garden_x_spread=3.0,
        reward_scale=1.0,
        dense_distance_clip=2.5,
        sparse_goal_threshold_multiplier=2.0,
        sparse_team_reward_scale=5.0,
        sparse_team_bonus=10.0,
        **kwargs,
    ):
        if num_landmarks is None:
            num_landmarks = num_agents

        dim_c = 2
        self.agent_collision_coef = agent_collision_coef
        self.wall_collision_coef = wall_collision_coef
        self.corridor_width = corridor_width
        self.garden_x_spread = garden_x_spread
        self.use_goal_conditioned_sd = use_goal_conditioned_sd
        self.sparse_reward = sparse_reward
        assert spawn_mode in ("unbalanced", "balanced", "diagonal", "garden"), \
            f"spawn_mode must be 'unbalanced', 'balanced', 'diagonal', or 'garden', got '{spawn_mode}'"
        self.spawn_mode = spawn_mode
        self.fixed_spawn_y = fixed_spawn_y
        self.fixed_spawn_x = fixed_spawn_x
        self.hard_collision = hard_collision
        self.hard_collision_iterations = hard_collision_iterations
        self.collision_elasticity = collision_elasticity
        self.reward_scale = reward_scale
        self.dense_distance_clip = dense_distance_clip
        self.sparse_goal_threshold_multiplier = sparse_goal_threshold_multiplier
        self.sparse_team_reward_scale = sparse_team_reward_scale
        self.sparse_team_bonus = sparse_team_bonus
        self.map_size_horizontal = 5.0
        self.map_size_vertical = 3.0

        map_size = self.map_size_horizontal

        self.local_ratio = local_ratio
        assert (
            self.local_ratio >= 0.0 and self.local_ratio <= 1.0
        ), "local_ratio must be between 0.0 and 1.0"

        self.agent_size = agent_size
        self.landmark_size = landmark_size
        self.agent_wise_distance = agent_wise_distance
        assert (
            num_landmarks == num_agents
        ), "simple_corridor requires exactly one goal per agent"

        num_walls = 2
        wall_indices = []

        agents = ["agent_{}".format(i) for i in range(num_agents)]
        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        for i in range(num_walls):
            wall_idx = num_landmarks + i
            landmarks.append("wall{}".format(i + 1))
            wall_indices.append(wall_idx)

        half_corridor = corridor_width / 2
        wall1_width = 5.0 - half_corridor
        wall1_height = 2.0

        wall2_width = 5.0 - half_corridor
        wall2_height = 2.0

        wall_widths = [wall1_width, wall2_width]
        wall_heights = [wall1_height, wall2_height]

        total_landmarks = num_landmarks + num_walls
        goal_pos_dims = 2 if use_goal_conditioned_sd else 0

        base_dims = 4 + goal_pos_dims

        observation_spaces = {
            i:Box(-jnp.inf, jnp.inf, (base_dims + (num_agents-1)*4+(num_landmarks*2),))
            for i in agents
        }

        pair_colours_palette = [
            (235, 64, 52),
            (52, 152, 219),
            (46, 204, 113),
            (155, 89, 182),
            (241, 196, 15),
            (26, 188, 156),
            (230, 126, 34),
            (149, 165, 166),
        ]
        agent_colours = [pair_colours_palette[i % len(pair_colours_palette)] for i in range(num_agents)]
        goal_colours = [pair_colours_palette[i % len(pair_colours_palette)] for i in range(num_landmarks)]
        colour = agent_colours + goal_colours
        wall_colour = (191, 64, 64)
        colour = colour + [wall_colour] * num_walls

        rad = jnp.concatenate(
            [jnp.full((num_agents), agent_size),
             jnp.full((num_landmarks), landmark_size)]
        )
        wall_rad = jnp.full((num_walls,), 1e-6)
        rad = jnp.concatenate([rad, wall_rad])

        collide = jnp.concatenate(
            [jnp.full((num_agents), True),
             jnp.full((num_landmarks), False)]
        )
        collide = jnp.concatenate([collide, jnp.full((num_walls), False)])

        self.num_walls = num_walls
        self.wall_indices_jnp = jnp.array(wall_indices, dtype=jnp.int32)
        self.wall_widths_jnp = jnp.array(wall_widths)
        self.wall_heights_jnp = jnp.array(wall_heights)
        self.pair_colours_palette = pair_colours_palette

        self.num_regular_landmarks = num_landmarks

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=total_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=dim_c,
            colour=colour,
            rad=rad,
            collide=collide,
            map_size=map_size,
            wall_indices=self.wall_indices_jnp,
            wall_widths=self.wall_widths_jnp,
            wall_heights=self.wall_heights_jnp,
            hard_collision=hard_collision,
            hard_collision_iterations=hard_collision_iterations,
            collision_elasticity=collision_elasticity,
            **kwargs,
        )

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0))
        def _common_stats(aidx: int):
            regular_landmark_start = self.num_agents
            regular_landmark_end = self.num_agents + self.num_regular_landmarks
            landmark_pos = state.p_pos[regular_landmark_start : regular_landmark_end] - state.p_pos[aidx]

            other_pos = state.p_pos[: self.num_agents] - state.p_pos[aidx]

            other_pos = jnp.roll(other_pos, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]
            comm = jnp.roll(
                state.c[: self.num_agents], shift=self.num_agents - aidx - 1, axis=0
            )[: self.num_agents - 1]

            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            comm = jnp.roll(comm, shift=aidx, axis=0)

            return landmark_pos, other_pos, comm

        landmark_pos, other_pos, comm = _common_stats(self.agent_range)

        def _obs(aidx: int):
            goal_idx = self.num_agents + aidx
            goal_pos = state.p_pos[goal_idx]

            if self.use_goal_conditioned_sd:
                obs_vec = jnp.concatenate(
                    [
                        state.p_vel[aidx].flatten(),
                        state.p_pos[aidx].flatten(),
                        goal_pos.flatten(),
                        landmark_pos[aidx].flatten(),
                        other_pos[aidx].flatten(),
                        comm[aidx].flatten(),
                    ]
                )
            else:
                obs_vec = jnp.concatenate(
                    [
                        state.p_vel[aidx].flatten(),
                        state.p_pos[aidx].flatten(),
                        landmark_pos[aidx].flatten(),
                        other_pos[aidx].flatten(),
                        comm[aidx].flatten(),
                    ]
                )
            obs_vec = jnp.where(jnp.isfinite(obs_vec), obs_vec, 0.0)
            obs_vec = jnp.clip(obs_vec, -1e6, 1e6)
            return obs_vec

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        key_side, key_shuffle = jax.random.split(key)

        if self.spawn_mode in ("balanced", "garden"):
            num_top = self.num_agents // 2
        else:
            num_top = self.num_agents // 2 + 1
        num_bottom = self.num_agents - num_top

        base_agent_spawn_y = jnp.concatenate([
            jnp.full((num_top,), 2.0),
            jnp.full((num_bottom,), -2.0)
        ])
        base_goal_spawn_y = -base_agent_spawn_y

        if self.fixed_spawn_y:
            agent_spawn_y = base_agent_spawn_y
            goal_spawn_y = base_goal_spawn_y
        else:
            key_side, key_y = jax.random.split(key_side)
            spawn_top = jax.random.bernoulli(key_y, 0.5, (self.num_agents,))
            agent_spawn_y = jnp.where(spawn_top, 2.0, -2.0)
            goal_spawn_y = -agent_spawn_y

        if self.spawn_mode == "garden":
            x_spread = self.garden_x_spread
            top_x_positions = (
                jnp.linspace(-x_spread, x_spread, num_top)
                if num_top > 1
                else jnp.array([0.0])
            )
            bottom_x_positions = (
                jnp.linspace(-x_spread, x_spread, num_bottom)
                if num_bottom > 1
                else jnp.array([0.0])
            )
            base_x_positions = jnp.concatenate([top_x_positions, bottom_x_positions])
        else:
            num_candidates = self.num_regular_landmarks
            base_x_positions = (
                jnp.linspace(
                    -self.map_size_horizontal + 1.0,
                    self.map_size_horizontal - 1.0,
                    num_candidates,
                )
                if num_candidates > 1
                else jnp.array([0.0])
            )

        if self.fixed_spawn_x:
            agent_x_positions = jnp.clip(
                base_x_positions[: self.num_agents],
                -self.map_size_horizontal,
                self.map_size_horizontal
            )
        else:
            key_side, key_x = jax.random.split(key_side)
            shuffled_positions = jax.random.permutation(key_x, base_x_positions)
            agent_x_positions = jnp.clip(
                shuffled_positions[: self.num_agents],
                -self.map_size_horizontal,
                self.map_size_horizontal
            )

        if self.spawn_mode == "diagonal":
            goal_x_positions = -agent_x_positions[: self.num_regular_landmarks]
        elif self.spawn_mode == "garden":
            goal_x_positions = agent_x_positions[: self.num_regular_landmarks][::-1]
        else:
            goal_x_positions = agent_x_positions[: self.num_regular_landmarks]

        agent_positions = jnp.stack([agent_x_positions, agent_spawn_y], axis=1)

        landmark_positions = jnp.stack(
            [
                goal_x_positions,
                goal_spawn_y[: self.num_regular_landmarks],
            ],
            axis=1,
        )

        half_corridor = self.corridor_width / 2
        wall1_center_x = (-5.0 - half_corridor) / 2
        wall2_center_x = (half_corridor + 5.0) / 2
        wall1_pos = jnp.array([wall1_center_x, 0.0])
        wall2_pos = jnp.array([wall2_center_x, 0.0])

        wall_positions = jnp.stack([wall1_pos, wall2_pos])

        landmark_positions = jnp.concatenate([landmark_positions, wall_positions])

        p_pos = jnp.concatenate([agent_positions, landmark_positions])

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def rewards(self, state: State) -> Dict[str, float]:
        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: int, other_idx: int):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx,
                other_idx,
                state,
            )

        c = _collisions(
            self.agent_range,
            self.agent_range,
        )

        @partial(jax.vmap, in_axes=(0, None))
        def _wall_collisions(agent_idx: int, state: State):
            agent_pos = state.p_pos[agent_idx]
            agent_size = self.rad[agent_idx]

            def check_wall_collision(wall_idx: int):
                wall_entity_idx = self.num_agents + wall_idx
                wall_center = state.p_pos[wall_entity_idx]
                wall_width = self.wall_widths_jnp[wall_idx]
                wall_height = self.wall_heights_jnp[wall_idx]

                left = wall_center[0] - wall_width/2
                right = wall_center[0] + wall_width/2
                bottom = wall_center[1] - wall_height/2
                top = wall_center[1] + wall_height/2

                return (left <= agent_pos[0] + agent_size) & (agent_pos[0] - agent_size <= right) & \
                       (bottom <= agent_pos[1] + agent_size) & (agent_pos[1] - agent_size <= top)

            wall_collisions = jax.vmap(check_wall_collision)(self.wall_indices_jnp)
            return -jnp.sum(wall_collisions) * self.wall_collision_coef

        wall_collision_penalties = _wall_collisions(self.agent_range, state)

        regular_landmark_start = self.num_agents
        regular_landmark_end = self.num_agents + self.num_regular_landmarks
        goal_positions = state.p_pos[regular_landmark_start:regular_landmark_end]

        agent_positions = state.p_pos[:self.num_agents]
        paired_goal_positions = goal_positions[:self.num_agents]

        pair_distances = jnp.linalg.norm(agent_positions - paired_goal_positions, axis=1)
        pair_distances = jnp.where(jnp.isfinite(pair_distances), pair_distances, 1e6)

        if not self.sparse_reward:
            pair_distances_for_penalty = jnp.clip(pair_distances, 0.0, self.dense_distance_clip)
        else:
            pair_distances_for_penalty = pair_distances

        normalized_distances = pair_distances_for_penalty / self.map_size_horizontal

        if self.sparse_reward:
            goal_threshold = self.sparse_goal_threshold_multiplier * (self.agent_size + self.landmark_size)
        else:
            goal_threshold = self.agent_size + self.landmark_size
        goal_reached = pair_distances < goal_threshold

        if self.sparse_reward:
            num_agents_reached = jnp.sum(goal_reached.astype(jnp.float32))
            team_goal_reward = self.sparse_team_reward_scale * (num_agents_reached / self.num_agents)
            goal_bonus_per_agent = jnp.zeros(self.num_agents)
            all_reached = jnp.all(goal_reached)
            team_bonus = jnp.where(all_reached, self.sparse_team_bonus, 0.0)
        else:
            goal_bonus_per_agent = jnp.where(goal_reached, 0.5, 0.0)
            all_reached = jnp.all(goal_reached)
            team_bonus = jnp.where(all_reached, 2.0, 0.0)
            team_goal_reward = 0.0

        def _agent_local_reward(aidx: int, collisions: chex.Array):
            collision_penalty = -self.agent_collision_coef * jnp.sum(collisions[aidx])
            wall_penalty = wall_collision_penalties[aidx]
            goal_bonus = goal_bonus_per_agent[aidx]
            return collision_penalty + wall_penalty + goal_bonus

        global_distance_penalty = jnp.where(
            self.sparse_reward,
            0.0,
            -jnp.sum(normalized_distances)
        )
        global_rew = global_distance_penalty + team_bonus

        rew = {
            a: _agent_local_reward(i, c) * self.local_ratio + global_rew * (1 - self.local_ratio)
            for i, a in enumerate(self.agents)
        }

        rew = {k: jnp.clip(jnp.where(jnp.isfinite(v), v, 0.0), -1e6, 1e6) * self.reward_scale for k, v in rew.items()}
        return rew

    @partial(jax.jit, static_argnums=[0])
    def _handle_walls_and_boundaries_with_walls(self, prev_pos: chex.Array, new_pos: chex.Array, vel: chex.Array):
        def check_wall_collision(pos: chex.Array, size: float, wall_center: chex.Array, wall_width: float, wall_height: float):
            pos_flat = jnp.ravel(pos)
            wall_center_flat = jnp.ravel(wall_center)

            pos_x = pos_flat[0]
            pos_y = pos_flat[1]
            wall_center_x = wall_center_flat[0]
            wall_center_y = wall_center_flat[1]

            left = wall_center_x - wall_width/2
            right = wall_center_x + wall_width/2
            bottom = wall_center_y - wall_height/2
            top = wall_center_y + wall_height/2

            return (left <= pos_x + size) & (pos_x - size <= right) & \
                   (bottom <= pos_y + size) & (pos_y - size <= top)

        def handle_wall_collision_single(prev: chex.Array, new: chex.Array, v: chex.Array,
                                        wall_center: chex.Array, wall_width: float, wall_height: float, size: float):
            prev_flat = jnp.ravel(prev)
            new_flat = jnp.ravel(new)
            v_flat = jnp.ravel(v)
            wall_center_flat = jnp.ravel(wall_center)

            prev_x = prev_flat[0]
            prev_y = prev_flat[1]
            new_x = new_flat[0]
            new_y = new_flat[1]
            v_x = v_flat[0]
            v_y = v_flat[1]
            wall_center_x = wall_center_flat[0]
            wall_center_y = wall_center_flat[1]

            left = wall_center_x - wall_width/2
            right = wall_center_x + wall_width/2
            bottom = wall_center_y - wall_height/2
            top = wall_center_y + wall_height/2

            hit_left = (prev_x + size <= left) & (new_x + size > left)
            hit_right = (prev_x - size >= right) & (new_x - size < right)
            hit_bottom = (prev_y + size <= bottom) & (new_y + size > bottom)
            hit_top = (prev_y - size >= top) & (new_y - size < top)

            hit_x = hit_left | hit_right
            hit_y = hit_bottom | hit_top

            safe_pos_x = jnp.where(hit_left, left - size,
                                  jnp.where(hit_right, right + size, new_x))
            safe_pos_y = jnp.where(hit_bottom, bottom - size,
                                  jnp.where(hit_top, top + size, new_y))

            new_vel_x = jnp.where(hit_x, -self.reflect_damping * v_x, v_x)
            new_vel_x = jnp.where(hit_x & (v_x == 0),
                                  jnp.where(prev_x < left, 1e-3, -1e-3),
                                  new_vel_x)

            new_vel_y = jnp.where(hit_y, -self.reflect_damping * v_y, v_y)
            new_vel_y = jnp.where(hit_y & (v_y == 0),
                                  jnp.where(prev_y < bottom, 1e-3, -1e-3),
                                  new_vel_y)

            safe_pos = jnp.stack([safe_pos_x, safe_pos_y], axis=0)
            safe_vel = jnp.stack([new_vel_x, new_vel_y], axis=0)

            return safe_pos, safe_vel

        def reflect_boundary_rectangular(pos: chex.Array, v: chex.Array, size: float):
            map_size_x = self.map_size_horizontal
            map_size_y = self.map_size_vertical

            hit_left = pos[0] - size <= -map_size_x
            hit_right = pos[0] + size >= map_size_x
            hit_x = hit_left | hit_right

            pos_x_left = -map_size_x + size
            pos_x_right = map_size_x - size
            new_pos_x = jnp.where(hit_left, pos_x_left,
                                  jnp.where(hit_right, pos_x_right, pos[0]))

            vel_reflect_x = jnp.where(hit_left, v[0] < 0, v[0] > 0)
            vel_x_reflected = -self.reflect_damping * v[0]
            vel_x_stuck = jnp.where(hit_left, 1e-3, -1e-3)

            new_vel_x = jnp.where(hit_x & (v[0] == 0), vel_x_stuck,
                                  jnp.where(hit_x & vel_reflect_x, vel_x_reflected, v[0]))

            hit_bottom = pos[1] - size <= -map_size_y
            hit_top = pos[1] + size >= map_size_y
            hit_y = hit_bottom | hit_top

            pos_y_bottom = -map_size_y + size
            pos_y_top = map_size_y - size
            new_pos_y = jnp.where(hit_bottom, pos_y_bottom,
                                  jnp.where(hit_top, pos_y_top, pos[1]))

            vel_reflect_y = jnp.where(hit_bottom, v[1] < 0, v[1] > 0)
            vel_y_reflected = -self.reflect_damping * v[1]
            vel_y_stuck = jnp.where(hit_bottom, 1e-3, -1e-3)

            new_vel_y = jnp.where(hit_y & (v[1] == 0), vel_y_stuck,
                                  jnp.where(hit_y & vel_reflect_y, vel_y_reflected, v[1]))

            new_pos = jnp.array([new_pos_x, new_pos_y])
            new_vel = jnp.array([new_vel_x, new_vel_y])

            return new_pos, new_vel

        def process_agent_with_walls(carry, agent_idx):
            agent_pos_all, agent_vel_all = carry

            agent_idx_scalar = agent_idx
            size = self.rad[agent_idx_scalar]

            current_pos = agent_pos_all[agent_idx_scalar]
            current_vel = agent_vel_all[agent_idx_scalar]
            prev_agent_pos = prev_pos[agent_idx_scalar]

            prev_flat = jnp.ravel(prev_agent_pos)
            pos_flat = jnp.ravel(current_pos)
            vel_flat = jnp.ravel(current_vel)

            prev_pos_2d = jnp.array([prev_flat[0], prev_flat[1]])
            pos_2d = jnp.array([pos_flat[0], pos_flat[1]])
            vel_2d = jnp.array([vel_flat[0], vel_flat[1]])

            def check_and_handle_wall(carry, wall_data):
                pos, vel = carry
                wall_idx, wall_width, wall_height = wall_data

                wall_width_scalar = jnp.ravel(jnp.asarray(wall_width))[0]
                wall_height_scalar = jnp.ravel(jnp.asarray(wall_height))[0]

                wall_idx_int = jnp.ravel(jnp.asarray(wall_idx, dtype=jnp.int32))[0]
                wall_entity_idx = self.num_agents + wall_idx_int
                wall_center_raw = prev_pos[wall_entity_idx]

                wall_center_flat = jnp.ravel(wall_center_raw)
                wall_center = jnp.array([wall_center_flat[0], wall_center_flat[1]])

                size_scalar = jnp.ravel(jnp.asarray(size))[0]

                is_colliding = check_wall_collision(pos, size_scalar, wall_center, wall_width_scalar, wall_height_scalar)
                safe_pos, safe_vel = handle_wall_collision_single(
                    prev_pos_2d, pos, vel, wall_center, wall_width_scalar, wall_height_scalar, size_scalar
                )

                is_colliding_bool = jnp.any(is_colliding)
                pos = jnp.where(jnp.broadcast_to(is_colliding_bool, (2,)), safe_pos, pos)
                vel = jnp.where(jnp.broadcast_to(is_colliding_bool, (2,)), safe_vel, vel)
                return (pos, vel), None

            wall_data = jnp.stack([
                self.wall_indices_jnp.astype(float),
                self.wall_widths_jnp,
                self.wall_heights_jnp
            ], axis=1)

            (pos_2d, vel_2d), _ = jax.lax.scan(
                check_and_handle_wall,
                (pos_2d, vel_2d),
                wall_data
            )

            pos_2d, vel_2d = reflect_boundary_rectangular(pos_2d, vel_2d, size)

            agent_pos_all = agent_pos_all.at[agent_idx_scalar].set(pos_2d)
            agent_vel_all = agent_vel_all.at[agent_idx_scalar].set(vel_2d)

            return (agent_pos_all, agent_vel_all), None

        agent_pos_all = new_pos[:self.num_agents]
        agent_vel_all = vel[:self.num_agents]

        agent_indices = jnp.arange(self.num_agents)

        (new_agent_pos, new_agent_vel), _ = jax.lax.scan(
            process_agent_with_walls,
            (agent_pos_all, agent_vel_all),
            agent_indices
        )

        final_pos = jnp.concatenate([new_agent_pos, new_pos[self.num_agents:]])
        final_vel = jnp.concatenate([new_agent_vel, vel[self.num_agents:]])
        final_pos = jnp.where(jnp.isfinite(final_pos), final_pos, 0.0)
        final_vel = jnp.where(jnp.isfinite(final_vel), final_vel, 0.0)

        return final_pos, final_vel

    @partial(jax.jit, static_argnums=[0])
    def _world_step(self, key: chex.PRNGKey, state: State, u: chex.Array):
        p_pos, p_vel = super()._world_step(key, state, u)

        if self.hard_collision:
            p_pos = p_pos.at[:self.num_agents, 0].set(
                jnp.clip(p_pos[:self.num_agents, 0],
                        -self.map_size_horizontal + self.rad[:self.num_agents],
                        self.map_size_horizontal - self.rad[:self.num_agents])
            )
            p_pos = p_pos.at[:self.num_agents, 1].set(
                jnp.clip(p_pos[:self.num_agents, 1],
                        -self.map_size_vertical + self.rad[:self.num_agents],
                        self.map_size_vertical - self.rad[:self.num_agents])
            )

        return p_pos, p_vel
