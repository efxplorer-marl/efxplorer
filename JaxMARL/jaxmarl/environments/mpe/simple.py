""" 
Base class for MPE PettingZoo envs.

TODO: viz for communication env, e.g. crypto
"""

import jax
import jax.numpy as jnp
import numpy as onp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.mpe.default_params import *
import chex
from jaxmarl.environments.spaces import Box, Discrete
from flax import struct
from typing import Tuple, Optional, Dict
from functools import partial

import matplotlib.pyplot as plt
import matplotlib

@struct.dataclass
class State:
    """Basic MPE State"""

    p_pos: chex.Array  # [num_entities, [x, y]]
    p_vel: chex.Array  # [n, [x, y]]
    c: chex.Array  # communication state [num_agents, [dim_c]]
    done: chex.Array
    step: int
    goal: int = None  # index of target landmark, used in: SimpleSpeakerListenerMPE, SimpleReferenceMPE, SimplePushMPE, SimpleAdversaryMPE


class SimpleMPE(MultiAgentEnv):
    def __init__(
        self,
        num_agents=1,
        action_type=DISCRETE_ACT,
        agents=None,
        num_landmarks=1,
        landmarks=None,
        action_spaces=None,
        observation_spaces=None,
        colour=None,
        dim_c=0,
        dim_p=2,
        max_steps=MAX_STEPS,
        dt=DT,
        **kwargs,
        ):
        # Agent and entity constants
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_entities = num_agents + num_landmarks
        self.agent_range = jnp.arange(num_agents)
        self.entity_range = jnp.arange(self.num_entities)

        # Setting, and sense checking, entity names and agent action spaces
        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert (
                len(agents) == num_agents
            ), f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}
        self.classes = self.create_agent_classes()

        if landmarks is None:
            self.landmarks = [f"landmark {i}" for i in range(num_landmarks)]
        else:
            assert (
                len(landmarks) == num_landmarks
            ), f"Number of landmarks {len(landmarks)} does not match number of landmarks {num_landmarks}"
            self.landmarks = landmarks
        self.l_to_i = {l: i + self.num_agents for i, l in enumerate(self.landmarks)}

        if action_spaces is None:
            if action_type == DISCRETE_ACT:
                self.action_spaces = {i: Discrete(5) for i in self.agents}
            elif action_type == CONTINUOUS_ACT:
                self.action_spaces = {i: Box(0.0, 1.0, (5,)) for i in self.agents}
        else:
            assert (
                len(action_spaces.keys()) == num_agents
            ), f"Number of action spaces {len(action_spaces.keys())} does not match number of agents {num_agents}"
            self.action_spaces = action_spaces

        if observation_spaces is None:
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (4,)) for i in self.agents
            }
        else:
            assert (
                len(observation_spaces.keys()) == num_agents
            ), f"Number of observation spaces {len(observation_spaces.keys())} does not match number of agents {num_agents}"
            self.observation_spaces = observation_spaces

        self.colour = (
            colour
            if colour is not None
            else [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks
        )

        # Action type
        if action_type == DISCRETE_ACT:
            self.action_decoder = self._decode_discrete_action
        elif action_type == CONTINUOUS_ACT:
            self.action_decoder = self._decode_continuous_action
        else:
            raise NotImplementedError(f"Action type: {action_type} is not supported")

        # World dimensions
        self.dim_c = dim_c  # communication channel dimensionality
        self.dim_p = dim_p  # position dimensionality

        # Environment parameters
        self.max_steps = max_steps
        self.dt = dt
        if "rad" in kwargs:
            self.rad = kwargs["rad"]
            assert (
                len(self.rad) == self.num_entities
            ), f"Rad array length {len(self.rad)} does not match number of entities {self.num_entities}"
            assert jnp.all(self.rad > 0), f"Rad array must be positive, got {self.rad}"
        else:
            self.rad = jnp.concatenate(
                [jnp.full((self.num_agents), 0.15), jnp.full((self.num_landmarks), 0.2)]
            )

        if "moveable" in kwargs:
            self.moveable = kwargs["moveable"]
            assert (
                len(self.moveable) == self.num_entities
            ), f"Moveable array length {len(self.moveable)} does not match number of entities {self.num_entities}"
            assert (
                self.moveable.dtype == bool
            ), f"Moveable array must be boolean, got {self.moveable}"
        else:
            self.moveable = jnp.concatenate(
                [
                    jnp.full((self.num_agents), True),
                    jnp.full((self.num_landmarks), False),
                ]
            )

        if "silent" in kwargs:
            self.silent = kwargs["silent"]
            assert (
                len(self.silent) == self.num_agents
            ), f"Silent array length {len(self.silent)} does not match number of agents {self.num_agents}"
        else:
            self.silent = jnp.full((self.num_agents), 1)

        if "collide" in kwargs:
            self.collide = kwargs["collide"]
            assert (
                len(self.collide) == self.num_entities
            ), f"Collide array length {len(self.collide)} does not match number of entities {self.num_entities}"
        else:
            self.collide = jnp.full((self.num_entities), False)

        if "mass" in kwargs:
            self.mass = kwargs["mass"]
            assert (
                len(self.mass) == self.num_entities
            ), f"Mass array length {len(self.mass)} does not match number of entities {self.num_entities}"
            assert jnp.all(
                self.mass > 0
            ), f"Mass array must be positive, got {self.mass}"
        else:
            self.mass = jnp.full((self.num_entities), 1.0)

        if "accel" in kwargs:
            self.accel = kwargs["accel"]
            assert (
                len(self.accel) == self.num_agents
            ), f"Accel array length {len(self.accel)} does not match number of agents {self.num_agents}"
            assert jnp.all(
                self.accel > 0
            ), f"Accel array must be positive, got {self.accel}"
        else:
            self.accel = jnp.full((self.num_agents), 5.0)

        if "max_speed" in kwargs:
            self.max_speed = kwargs["max_speed"]
            assert (
                len(self.max_speed) == self.num_entities
            ), f"Max speed array length {len(self.max_speed)} does not match number of entities {self.num_entities}"
        else:
            self.max_speed = jnp.concatenate(
                [jnp.full((self.num_agents), -1), jnp.full((self.num_landmarks), 0.0)]
            )

        if "u_noise" in kwargs:
            self.u_noise = kwargs["u_noise"]
            assert (
                len(self.u_noise) == self.num_agents
            ), f"U noise array length {len(self.u_noise)} does not match number of agents {self.num_agents}"
        else:
            self.u_noise = jnp.full((self.num_agents), 0)

        if "c_noise" in kwargs:
            self.c_noise = kwargs["c_noise"]
            assert (
                len(self.c_noise) == self.num_agents
            ), f"C noise array length {len(self.c_noise)} does not match number of agents {self.num_agents}"
        else:
            self.c_noise = jnp.full((self.num_agents), 0)

        if "damping" in kwargs:
            self.damping = kwargs["damping"]
            assert (
                self.damping >= 0
            ), f"Damping must be non-negative, got {self.damping}"
        else:
            self.damping = DAMPING

        if "contact_force" in kwargs:
            self.contact_force = kwargs["contact_force"]
        else:
            self.contact_force = CONTACT_FORCE

        if "contact_margin" in kwargs:
            self.contact_margin = kwargs["contact_margin"]
        else:
            self.contact_margin = CONTACT_MARGIN

        if "map_size" in kwargs:
            self.map_size = kwargs["map_size"]
        else:
            self.map_size = jnp.inf

        if "reflect_damping" in kwargs:
            self.reflect_damping = kwargs["reflect_damping"]
        else:
            self.reflect_damping = 0.75

        self.debug_trace = kwargs.get("debug_trace", False)

        self.hard_collision = kwargs.get("hard_collision", False)
        self.hard_collision_iterations = kwargs.get("hard_collision_iterations", 10)
        self.collision_elasticity = kwargs.get("collision_elasticity", 0.3)

        if self.hard_collision:
            min_agent_radius = float(jnp.min(self.rad[:num_agents]))
            self.adaptive_safe_speed = 0.8 * min_agent_radius / dt
        else:
            self.adaptive_safe_speed = 50.0

        if "wall_indices" in kwargs:
            self.wall_indices = kwargs["wall_indices"]
            if len(self.wall_indices) > 0:
                assert (
                    len(self.wall_indices) <= self.num_landmarks
                ), "wall_indices must be within landmark range"
        else:
            self.wall_indices = jnp.array([])

        if "wall_widths" in kwargs:
            self.wall_widths = kwargs["wall_widths"]
            if len(self.wall_indices) > 0:
                assert (
                    len(self.wall_widths) == len(self.wall_indices)
                ), "wall_widths length must match wall_indices length"
        else:
            self.wall_widths = jnp.array([])

        if "wall_heights" in kwargs:
            self.wall_heights = kwargs["wall_heights"]
            if len(self.wall_indices) > 0:
                assert (
                    len(self.wall_heights) == len(self.wall_indices)
                ), "wall_heights length must match wall_indices length"
        else:
            self.wall_heights = jnp.array([])

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        u, c = self.set_actions(actions)
        if (
            c.shape[1] < self.dim_c
        ):  # This is due to the MPE code carrying around 0s for the communication channels
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1
            )

        key, key_w = jax.random.split(key)
        prev_p_pos, prev_p_vel = state.p_pos, state.p_vel
        p_pos, p_vel = self._world_step(key_w, state, u)

        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, self.c_noise, self.silent)
        done = jnp.full((self.num_agents), state.step >= self.max_steps)

        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)

        obs = self.get_obs(state)

        obs = {a: jnp.where(jnp.isfinite(v), v, 0.0) for a, v in obs.items()}
        reward = {a: jnp.clip(jnp.where(jnp.isfinite(v), v, -100.0), -1e6, 1e6) for a, v in reward.items()}

        def nf_mask(x):
            return jnp.logical_not(jnp.isfinite(x))
        p_pos_nf = jax.vmap(lambda v: jnp.sum(nf_mask(v)))(state.p_pos[: self.num_agents])
        p_vel_nf = jax.vmap(lambda v: jnp.sum(nf_mask(v)))(state.p_vel[: self.num_agents])
        obs_nf = jnp.array([jnp.sum(nf_mask(obs[a])) for a in self.agents])
        rew_nf = jnp.array([jnp.sum(nf_mask(reward[a])) for a in self.agents])
        info = {
            "nan/p_pos_nonfinite_count": p_pos_nf,
            "nan/p_vel_nonfinite_count": p_vel_nf,
            "nan/obs_nonfinite_count": obs_nf,
            "nan/reward_nonfinite_count": rew_nf,
            "nan/any_state_nonfinite": (p_pos_nf + p_vel_nf) > 0,
        }

        if self.debug_trace and (
            jnp.any(p_pos_nf > 0) | jnp.any(p_vel_nf > 0) | jnp.any(obs_nf > 0) | jnp.any(rew_nf > 0)
        ):
            trace = {
                "step": state.step,
                "prev_p_pos": prev_p_pos,
                "prev_p_vel": prev_p_vel,
                "action_u": u,
                "action_c": c,
                "new_p_pos": p_pos,
                "new_p_vel": p_vel,
                "obs": {a: obs[a] for a in self.agents},
                "reward": {a: reward[a] for a in self.agents},
            }
            info["nan/trace"] = trace

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Initialise with random positions"""

        key_a, key_l = jax.random.split(key)

        p_pos = jnp.concatenate(
            [
                jax.random.uniform(
                    key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
                ),
                jax.random.uniform(
                    key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0
                ),
            ]
        )

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return dictionary of agent observations"""

        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> jnp.ndarray:
            """Return observation for agent i."""
            landmark_rel_pos = state.p_pos[self.num_agents :] - state.p_pos[aidx]

            return jnp.concatenate(
                [state.p_vel[aidx].flatten(), landmark_rel_pos.flatten()]
            )

        obs = _observation(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    def rewards(self, state: State) -> Dict[str, float]:
        """Assign rewards for all agents"""

        @partial(jax.vmap, in_axes=[0, None])
        def _reward(aidx: int, state: State):
            return -1 * jnp.sum(
                jnp.square(state.p_pos[aidx] - state.p_pos[self.num_agents :])
            )

        r = _reward(self.agent_range, state)
        return {agent: r[i] for i, agent in enumerate(self.agents)}

    def set_actions(self, actions: Dict):
        """Extract u and c actions for all agents from actions Dict."""

        actions = jnp.array([actions[i] for i in self.agents]).reshape(
            (self.num_agents, -1)
        )

        return self.action_decoder(self.agent_range, actions)

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_continuous_action(
        self, a_idx: int, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.array([action[1] - action[2], action[3] - action[4]])
        u = u * self.accel[a_idx] * self.moveable[a_idx]
        c = action[5:]
        return u, c

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _decode_discrete_action(
        self, a_idx: int, action: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        u = jnp.zeros((self.dim_p,))
        idx = jax.lax.select(action <= 2, 0, 1)
        u_val = jax.lax.select(action % 2 == 0, 1.0, -1.0) * (action != 0)
        u = u.at[idx].set(u_val)
        u = u * self.accel[a_idx] * self.moveable[a_idx]
        return u, jnp.zeros((self.dim_c,))

    def _world_step(self, key: chex.PRNGKey, state: State, u: chex.Array):
        p_force = jnp.zeros((self.num_agents, 2))

        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_agents)
        p_force = self._apply_action_force(
            key_noise, p_force, u, self.u_noise, self.moveable[: self.num_agents]
        )
        # jax.debug.print('jax p_force post agent {p_force}', p_force=p_force)

        # apply environment forces
        p_force = jnp.concatenate([p_force, jnp.zeros((self.num_landmarks, 2))])
        p_force = self._apply_environment_force(p_force, state)

        SAFE_FORCE_MAX = 1e3
        p_force = jnp.clip(p_force, -SAFE_FORCE_MAX, SAFE_FORCE_MAX)
        # print('p_force post apply env force', p_force)
        # jax.debug.print('jax p_force final: {p_force}', p_force=p_force)

        # integrate physical state
        p_pos, p_vel = self._integrate_state(
            p_force, state.p_pos, state.p_vel, self.mass, self.moveable, self.max_speed
        )

        speed = jnp.sqrt(jnp.sum(jnp.square(p_vel), axis=1))
        scale = jnp.minimum(1.0, self.adaptive_safe_speed / (speed + 1e-6))
        p_vel = p_vel * scale[:, None]

        has_walls = jnp.logical_and(jnp.isfinite(self.map_size), self.wall_indices.shape[0] > 0)
        p_pos, p_vel = jax.lax.cond(
            has_walls,
            lambda: self._handle_walls_and_boundaries_with_walls(state.p_pos, p_pos, p_vel),
            lambda: self._handle_walls_and_boundaries_without_walls(state.p_pos, p_pos, p_vel)
        )

        def clamp_pos_bounds(pos):
            return jnp.clip(pos, -self.map_size, self.map_size)
        p_pos = jax.lax.cond(
            self.map_size is not None,
            lambda: jax.vmap(clamp_pos_bounds)(p_pos),
            lambda: p_pos,
        )

        p_pos = jnp.where(jnp.isfinite(p_pos), p_pos, 0.0)
        p_vel = jnp.where(jnp.isfinite(p_vel), p_vel, 0.0)

        if self.hard_collision:
            p_pos, p_vel = self._resolve_hard_agent_collisions(p_pos, p_vel)
            p_pos, p_vel = self._resolve_hard_wall_collisions(p_pos, p_vel)

        return p_pos, p_vel

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def _apply_comm_action(
        self, key: chex.PRNGKey, c: chex.Array, c_noise: int, silent: int
    ) -> chex.Array:
        silence = jnp.zeros(c.shape)
        noise = jax.random.normal(key, shape=c.shape) * c_noise
        return jax.lax.select(silent, silence, c + noise)

    # gather agent action forces
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def _apply_action_force(
        self,
        key: chex.PRNGKey,
        p_force: chex.Array,
        u: chex.Array,
        u_noise: int,
        moveable: bool,
    ):
        noise = jax.random.normal(key, shape=u.shape) * u_noise
        return jax.lax.select(moveable, u + noise, p_force)

    def _apply_environment_force(self, p_force_all: chex.Array, state: State):
        """gather physical forces acting on entities"""

        @partial(jax.vmap, in_axes=[0])
        def __env_force_outer(idx: int):
            @partial(jax.vmap, in_axes=[None, 0])
            def __env_force_inner(idx_a: int, idx_b: int):
                l = idx_b <= idx_a
                l_a = jnp.zeros((2, 2))

                collision_force = self._get_collision_force(idx_a, idx_b, state)

                xx = jax.lax.select(l, l_a, collision_force)
                # jax.debug.print('{a} {b} {f}', a=idx_a, b=idx_b, f=xx)
                return xx

            p_force_t = __env_force_inner(idx, self.entity_range)

            p_force_a = jnp.sum(p_force_t[:, 0], axis=0)  # ego force from other agents
            p_force_o = p_force_t[:, 1]
            p_force_o = p_force_o.at[idx].set(p_force_a)

            return p_force_o

        p_forces = __env_force_outer(self.entity_range)
        p_forces = jnp.sum(p_forces, axis=0)

        return p_forces + p_force_all

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0, 0])
    def _integrate_state(self, p_force, p_pos, p_vel, mass, moveable, max_speed):
        """integrate physical state"""

        p_pos += p_vel * self.dt
        p_vel = p_vel * (1 - self.damping)

        p_vel += (p_force / mass) * self.dt * moveable

        speed = jnp.sqrt(jnp.square(p_vel[0]) + jnp.square(p_vel[1]))
        inv_speed = jnp.where(speed > 0, 1.0 / speed, 0.0)
        over_max = p_vel * inv_speed * max_speed

        p_vel = jax.lax.select((speed > max_speed) & (max_speed >= 0), over_max, p_vel)

        return p_pos, p_vel

    def _handle_walls_and_boundaries_without_walls(self, prev_pos: chex.Array, new_pos: chex.Array, vel: chex.Array):
        def reflect_boundary(pos: chex.Array, v: chex.Array, size: float):
            map_size_val = self.map_size
            
            hit_left = pos[0] - size <= -map_size_val
            hit_right = pos[0] + size >= map_size_val
            hit_x = hit_left | hit_right
            
            pos_x_left = -map_size_val + size
            pos_x_right = map_size_val - size
            new_pos_x = jnp.where(hit_left, pos_x_left,
                                  jnp.where(hit_right, pos_x_right, pos[0]))
            
            vel_reflect_x = jnp.where(hit_left, v[0] < 0, v[0] > 0)
            vel_x_reflected = -self.reflect_damping * v[0]
            vel_x_stuck = jnp.where(hit_left, 1e-3, -1e-3)
            
            new_vel_x = jnp.where(hit_x & (v[0] == 0), vel_x_stuck,
                                  jnp.where(hit_x & vel_reflect_x, vel_x_reflected, v[0]))
            
            hit_bottom = pos[1] - size <= -map_size_val
            hit_top = pos[1] + size >= map_size_val
            hit_y = hit_bottom | hit_top
            
            pos_y_bottom = -map_size_val + size
            pos_y_top = map_size_val - size
            new_pos_y = jnp.where(hit_bottom, pos_y_bottom,
                                  jnp.where(hit_top, pos_y_top, pos[1]))
            
            vel_reflect_y = jnp.where(hit_bottom, v[1] < 0, v[1] > 0)
            vel_y_reflected = -self.reflect_damping * v[1]
            vel_y_stuck = jnp.where(hit_bottom, 1e-3, -1e-3)
            
            new_vel_y = jnp.where(hit_y & (v[1] == 0), vel_y_stuck,
                                  jnp.where(hit_y & vel_reflect_y, vel_y_reflected, v[1]))
            
            new_pos = jnp.stack([new_pos_x, new_pos_y], axis=0)
            new_vel = jnp.stack([new_vel_x, new_vel_y], axis=0)
            
            return new_pos, new_vel

        def process_agent_without_walls(carry, agent_idx):
            agent_pos_all, agent_vel_all = carry
            
            size = self.rad[agent_idx]
            
            current_pos = agent_pos_all[agent_idx]
            current_vel = agent_vel_all[agent_idx]
            
            pos_flat = jnp.ravel(current_pos)
            vel_flat = jnp.ravel(current_vel)
            
            pos_2d = jnp.array([pos_flat[0], pos_flat[1]])
            vel_2d = jnp.array([vel_flat[0], vel_flat[1]])
            
            pos_2d, vel_2d = reflect_boundary(pos_2d, vel_2d, size)
            
            agent_pos_all = agent_pos_all.at[agent_idx].set(pos_2d)
            agent_vel_all = agent_vel_all.at[agent_idx].set(vel_2d)
            
            return (agent_pos_all, agent_vel_all), None
        
        agent_pos_all = new_pos[:self.num_agents]
        agent_vel_all = vel[:self.num_agents]
        
        agent_indices = jnp.arange(self.num_agents)
        
        (new_agent_pos, new_agent_vel), _ = jax.lax.scan(
            process_agent_without_walls,
            (agent_pos_all, agent_vel_all),
            agent_indices
        )
        
        final_pos = jnp.concatenate([new_agent_pos, new_pos[self.num_agents:]])
        final_vel = jnp.concatenate([new_agent_vel, vel[self.num_agents:]])
        
        return final_pos, final_vel

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

        def reflect_boundary(pos: chex.Array, v: chex.Array, size: float):
            map_size_val = self.map_size
            
            hit_left = pos[0] - size <= -map_size_val
            hit_right = pos[0] + size >= map_size_val
            hit_x = hit_left | hit_right
            
            pos_x_left = -map_size_val + size
            pos_x_right = map_size_val - size
            new_pos_x = jnp.where(hit_left, pos_x_left,
                                  jnp.where(hit_right, pos_x_right, pos[0]))
            
            vel_reflect_x = jnp.where(hit_left, v[0] < 0, v[0] > 0)
            vel_x_reflected = -self.reflect_damping * v[0]
            vel_x_stuck = jnp.where(hit_left, 1e-3, -1e-3)
            
            new_vel_x = jnp.where(hit_x & (v[0] == 0), vel_x_stuck,
                                  jnp.where(hit_x & vel_reflect_x, vel_x_reflected, v[0]))
            
            hit_bottom = pos[1] - size <= -map_size_val
            hit_top = pos[1] + size >= map_size_val
            hit_y = hit_bottom | hit_top
            
            pos_y_bottom = -map_size_val + size
            pos_y_top = map_size_val - size
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
                self.wall_indices.astype(float),
                self.wall_widths,
                self.wall_heights
            ], axis=1)
            
            (pos_2d, vel_2d), _ = jax.lax.scan(
                check_and_handle_wall,
                (pos_2d, vel_2d),
                wall_data
            )
            
            pos_2d, vel_2d = reflect_boundary(pos_2d, vel_2d, size)
            
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
        
        return final_pos, final_vel

    # get collision forces for any contact between two entities BUG
    def _get_collision_force(self, idx_a: int, idx_b: int, state: State):
        dist_min = self.rad[idx_a] + self.rad[idx_b]
        delta_pos = state.p_pos[idx_a] - state.p_pos[idx_b]

        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))

        # softmax penetration (clamped to avoid overflow)
        k = self.contact_margin
        t = -(dist - dist_min) / k
        t = jnp.clip(t, -60.0, 60.0)
        penetration = jnp.logaddexp(0.0, t) * k
        eps = 1e-2
        dist_safe = jnp.maximum(dist, eps)
        force_dir = delta_pos / dist_safe
        rel_vel = state.p_vel[idx_a] - state.p_vel[idx_b]
        rel_speed = jnp.sqrt(jnp.sum(jnp.square(rel_vel)))
        rel_dir = rel_vel / jnp.maximum(rel_speed, 1e-6)
        fallback_dir = jnp.array([1.0, 0.0])
        overlap_dir = jnp.where(rel_speed > 1e-3, rel_dir, fallback_dir)
        force_dir = jnp.where((dist < eps)[...], overlap_dir, force_dir)
        force = self.contact_force * force_dir * penetration
        FMAX = 1e2
        force = jnp.clip(force, -FMAX, FMAX)
        force_a = +force * self.moveable[idx_a]
        force_b = -force * self.moveable[idx_b]
        force = jnp.array([force_a, force_b])

        c = (~self.collide[idx_a]) | (~self.collide[idx_b]) | (idx_a == idx_b)
        c_force = jnp.zeros((2, 2))
        return jax.lax.select(c, c_force, force)

    def create_agent_classes(self):
        if hasattr(self, "leader"):
            return {
                "leadadversary": self.leader,
                "adversaries": self.adversaries,
                "agents": self.good_agents,
            }
        elif hasattr(self, "adversaries"):
            return {
                "adversaries": self.adversaries,
                "agents": self.good_agents,
            }
        else:
            return {
                "agents": self.agents,
            }

    def agent_classes(self) -> Dict[str, list]:
        return self.classes

    ### === UTILITIES === ###
    def is_collision(self, a: int, b: int, state: State):
        """check if two entities are colliding"""
        dist_min = self.rad[a] + self.rad[b]
        delta_pos = state.p_pos[a] - state.p_pos[b]
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        return (dist < dist_min) & (self.collide[a] & self.collide[b]) & (a != b)

    @partial(jax.vmap, in_axes=(None, 0))
    def map_bounds_reward(self, x: float):
        """vmap over x, y coodinates"""
        w = x < 0.9
        m = x < 1.0
        mr = (x - 0.9) * 10
        br = jnp.min(jnp.array([jnp.exp(2 * x - 2), 10]))
        return jax.lax.select(m, mr, br) * ~w


    @partial(jax.jit, static_argnums=[0])
    def _resolve_hard_agent_collisions(self, p_pos: chex.Array, p_vel: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """

        """
        agent_pos = p_pos[:self.num_agents]
        agent_vel = p_vel[:self.num_agents]
        agent_rad = self.rad[:self.num_agents]

        def resolve_iteration(_, carry):
            pos, vel = carry

            def resolve_pair(carry, pair_idx):
                pos, vel = carry
                i = pair_idx // self.num_agents
                j = pair_idx % self.num_agents

                valid_pair = i < j

                delta = pos[i] - pos[j]
                dist_sq = jnp.sum(delta ** 2)
                dist = jnp.sqrt(dist_sq + 1e-8)
                min_dist = agent_rad[i] + agent_rad[j]

                overlap = min_dist - dist
                is_overlapping = (overlap > 0) & valid_pair

                direction = delta / (dist + 1e-8)

                correction = direction * overlap * 0.5

                pos_i_new = pos[i] + jnp.where(is_overlapping, correction, 0.0)
                pos_j_new = pos[j] - jnp.where(is_overlapping, correction, 0.0)

                vel_i_normal = jnp.sum(vel[i] * direction) * direction
                vel_j_normal = jnp.sum(vel[j] * direction) * direction

                elasticity = self.collision_elasticity
                vel_i_new = vel[i] - vel_i_normal + vel_j_normal * elasticity
                vel_j_new = vel[j] - vel_j_normal + vel_i_normal * elasticity

                vel_i_final = jnp.where(is_overlapping, vel_i_new, vel[i])
                vel_j_final = jnp.where(is_overlapping, vel_j_new, vel[j])

                pos = pos.at[i].set(pos_i_new)
                pos = pos.at[j].set(pos_j_new)
                vel = vel.at[i].set(vel_i_final)
                vel = vel.at[j].set(vel_j_final)

                return (pos, vel), None

            pair_indices = jnp.arange(self.num_agents * self.num_agents)

            (pos, vel), _ = jax.lax.scan(resolve_pair, (pos, vel), pair_indices)

            return (pos, vel)

        (agent_pos, agent_vel) = jax.lax.fori_loop(
            0, self.hard_collision_iterations,
            resolve_iteration,
            (agent_pos, agent_vel)
        )

        final_pos = jnp.concatenate([agent_pos, p_pos[self.num_agents:]])
        final_vel = jnp.concatenate([agent_vel, p_vel[self.num_agents:]])

        return final_pos, final_vel

    @partial(jax.jit, static_argnums=[0])
    def _resolve_hard_wall_collisions(self, p_pos: chex.Array, p_vel: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """

        """
        if self.wall_indices.shape[0] == 0:
            return p_pos, p_vel

        agent_pos = p_pos[:self.num_agents]
        agent_vel = p_vel[:self.num_agents]

        def resolve_agent_walls(agent_idx, carry):
            pos_all, vel_all = carry
            pos = pos_all[agent_idx]
            vel = vel_all[agent_idx]
            agent_size = self.rad[agent_idx]

            def resolve_single_wall(carry, wall_data):
                pos, vel = carry
                wall_idx, wall_width, wall_height = wall_data

                wall_idx_int = jnp.int32(wall_idx)
                wall_entity_idx = self.num_agents + wall_idx_int
                wall_center = p_pos[wall_entity_idx]

                left = wall_center[0] - wall_width / 2
                right = wall_center[0] + wall_width / 2
                bottom = wall_center[1] - wall_height / 2
                top = wall_center[1] + wall_height / 2

                left_exp = left - agent_size
                right_exp = right + agent_size
                bottom_exp = bottom - agent_size
                top_exp = top + agent_size

                inside_x = (pos[0] > left_exp) & (pos[0] < right_exp)
                inside_y = (pos[1] > bottom_exp) & (pos[1] < top_exp)
                inside_wall = inside_x & inside_y

                pen_left = pos[0] - left_exp
                pen_right = right_exp - pos[0]
                pen_bottom = pos[1] - bottom_exp
                pen_top = top_exp - pos[1]

                min_pen_x = jnp.minimum(pen_left, pen_right)
                min_pen_y = jnp.minimum(pen_bottom, pen_top)

                exit_x = min_pen_x < min_pen_y

                new_x_left = left_exp
                new_x_right = right_exp
                new_y_bottom = bottom_exp
                new_y_top = top_exp

                new_pos_x = jnp.where(
                    exit_x,
                    jnp.where(pen_left < pen_right, new_x_left, new_x_right),
                    pos[0]
                )
                new_pos_y = jnp.where(
                    ~exit_x,
                    jnp.where(pen_bottom < pen_top, new_y_bottom, new_y_top),
                    pos[1]
                )

                new_vel_x = jnp.where(
                    exit_x & inside_wall,
                    -vel[0] * self.collision_elasticity,
                    vel[0]
                )
                new_vel_y = jnp.where(
                    ~exit_x & inside_wall,
                    -vel[1] * self.collision_elasticity,
                    vel[1]
                )

                pos = jnp.where(
                    inside_wall,
                    jnp.array([new_pos_x, new_pos_y]),
                    pos
                )
                vel = jnp.where(
                    inside_wall,
                    jnp.array([new_vel_x, new_vel_y]),
                    vel
                )

                return (pos, vel), None

            wall_data = jnp.stack([
                self.wall_indices.astype(float),
                self.wall_widths,
                self.wall_heights
            ], axis=1)

            (pos, vel), _ = jax.lax.scan(resolve_single_wall, (pos, vel), wall_data)

            pos_all = pos_all.at[agent_idx].set(pos)
            vel_all = vel_all.at[agent_idx].set(vel)

            return (pos_all, vel_all)

        (agent_pos, agent_vel) = jax.lax.fori_loop(
            0, self.num_agents,
            lambda i, c: resolve_agent_walls(i, c),
            (agent_pos, agent_vel)
        )

        final_pos = jnp.concatenate([agent_pos, p_pos[self.num_agents:]])
        final_vel = jnp.concatenate([agent_vel, p_vel[self.num_agents:]])

        return final_pos, final_vel


if __name__ == "__main__":
    from jaxmarl.environments.mpe import MPEVisualizer

    num_agents = 3
    key = jax.random.PRNGKey(0)

    env = SimpleMPE(num_agents)

    obs, state = env.reset(key)

    mock_action = jnp.array([[1.0, 1.0, 0.1, 0.1, 0.0]])

    actions = jnp.repeat(mock_action[None], repeats=num_agents, axis=0).squeeze()

    actions = {agent: mock_action for agent in env.agents}
    a = env.agents
    a.reverse()
    print("a", a)
    actions = {agent: mock_action for agent in a}
    print("actions", actions)

    # env.enable_render()

    state_seq = []
    print("state", state)
    print("action spaces", env.action_spaces)

    for _ in range(25):
        state_seq.append(state)
        key, key_act = jax.random.split(key)
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }

        obs, state, rew, dones, _ = env.step_env(key, state, actions)

    viz = MPEVisualizer(env, state_seq)
    viz.animate(None, view=True)
