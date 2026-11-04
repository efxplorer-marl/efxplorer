import jax
import jax.numpy as jnp
import chex
from typing import Dict
from functools import partial
from jaxmarl.environments.mpe.simple_tag import SimpleTagMPE
from jaxmarl.environments.mpe.simple import State
from jaxmarl.environments.spaces import Box, Discrete
from jaxmarl.environments.mpe.default_params import DISCRETE_ACT


class SimpleTagScriptedPreyMPE(SimpleTagMPE):

    def __init__(
        self,
        num_good_agents=1,
        num_adversaries=3,
        num_obs=2,
        score_function="min",
        reward_mode="sparse",
        dense_reward_scale=1.0,
        **kwargs,
    ):
        super().__init__(
            num_good_agents=num_good_agents,
            num_adversaries=num_adversaries,
            num_obs=num_obs,
            action_type=DISCRETE_ACT,
            **kwargs,
        )

        self._all_agents = list(self.agents)
        self._learning_agents = list(self.adversaries)

        self.agents = self._learning_agents

        adv_obs_dim = 6 + self.num_landmarks * 2 + (self.num_agents - 1) * 2
        self.observation_spaces = {
            a: Box(-jnp.inf, jnp.inf, (adv_obs_dim,)) for a in self._learning_agents
        }

        self.action_spaces = {a: Discrete(5) for a in self._learning_agents}

        self.score_function = score_function

        self.reward_mode = reward_mode
        self.dense_reward_scale = dense_reward_scale

        self._prev_min_dist_to_prey = None

    def set_actions(self, actions: Dict):
        actions_array = jnp.array([actions[i] for i in self._all_agents]).reshape(
            (self.num_agents, -1)
        )
        return self.action_decoder(self.agent_range, actions_array)

    def _prey_policy(self, key: chex.PRNGKey, state: State, aidx: int) -> jnp.ndarray:
        n = 100

        key, _key = jax.random.split(key)
        length = jnp.sqrt(jax.random.uniform(_key, (n,), minval=0., maxval=1.))
        key, _key = jax.random.split(key)
        angle = jnp.pi * jnp.sqrt(jax.random.uniform(_key, (n,), minval=0., maxval=2.))
        x = length * jnp.cos(angle)
        y = length * jnp.sin(angle)

        scores = jnp.zeros(n, dtype=jnp.float32)

        if self.score_function == "sum":
            n_iter = 5
            for i in range(n_iter):
                waypoints_length = (length / float(n_iter)) * (i + 1)
                x_wp = waypoints_length * jnp.cos(angle)
                y_wp = waypoints_length * jnp.sin(angle)
                proj_pos = jnp.vstack((x_wp, y_wp)).transpose() + state.p_pos[aidx]
                delta_pos = state.p_pos[None, :, :] - proj_pos[:, None, :]
                dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=2))
                dist_min = self.rad + self.rad[aidx]
                collision_mask = (dist < dist_min[None]).sum(axis=1) > 0
                scores = jnp.where(collision_mask, -9999999.0, scores)
                if i == n_iter - 1:
                    scores += dist[:, :self.num_adversaries].sum(axis=1)
        elif self.score_function == "min":
            proj_pos = jnp.vstack((x, y)).transpose() + state.p_pos[aidx]
            rel_dis = jnp.sqrt(jnp.sum(jnp.square(
                state.p_pos[aidx] - state.p_pos[:self.num_adversaries]
            ), axis=1))
            min_dist_adv_idx = jnp.argmin(rel_dis)
            delta_pos = state.p_pos[:self.num_adversaries][None, :, :] - proj_pos[:, None, :]
            dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=2))
            dist_min = self.rad[:self.num_adversaries] + self.rad[aidx]
            collision_mask = (dist < dist_min[None]).sum(axis=1) > 0
            scores = jnp.where(collision_mask, -9999999.0, scores)
            scores += dist[:, min_dist_adv_idx]
        else:
            return jnp.zeros(2, dtype=jnp.float32)

        best_idx = jnp.argmax(scores)
        chosen_action = jnp.array([x[best_idx], y[best_idx]], dtype=jnp.float32)
        chosen_action = jax.lax.cond(
            scores[best_idx] < 0,
            lambda: chosen_action * 0.0,
            lambda: chosen_action
        )
        return chosen_action

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        full_actions = dict(actions)
        sample_action_shape = next(iter(actions.values())).shape
        for prey_name in self.good_agents:
            full_actions[prey_name] = jnp.zeros(sample_action_shape, dtype=jnp.int32)

        u, c = self.set_actions(full_actions)

        u_adversaries = u[:self.num_adversaries]

        prey_actions = []
        for i in range(self.num_good_agents):
            prey_idx = self.num_adversaries + i
            key, _key = jax.random.split(key)
            prey_action = self._prey_policy(_key, state, prey_idx)
            prey_actions.append(prey_action)

        u_prey = jnp.stack(prey_actions, axis=0)
        u = jnp.concatenate([u_adversaries, u_prey], axis=0)

        if c.shape[1] < self.dim_c:
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))],
                axis=1
            )

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, self.c_noise, self.silent)

        done = jnp.full((self.num_agents,), state.step >= self.max_steps)

        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step + 1,
        )

        all_rewards = self.rewards(state)
        all_obs = self.get_obs(state)

        reward = {a: all_rewards[a] for a in self._learning_agents}
        obs = {a: all_obs[a] for a in self._learning_agents}

        dones = {a: done[i] for i, a in enumerate(self._learning_agents)}
        dones.update({"__all__": jnp.all(done[:self.num_adversaries])})

        info = {}

        return obs, state, reward, dones, info

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey):
        key, _key = jax.random.split(key)

        key_a, key_l = jax.random.split(_key)
        p_pos = jnp.concatenate([
            jax.random.uniform(key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0),
            jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0),
        ])

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents,), False),
            step=0,
        )

        all_obs = self.get_obs(state)
        obs = {a: all_obs[a] for a in self._learning_agents}

        return obs, state

    def rewards(self, state: State) -> Dict[str, float]:
        sparse_rewards = super().rewards(state)

        if self.reward_mode == "sparse":
            return sparse_rewards

        adv_pos = state.p_pos[:self.num_adversaries]
        prey_pos = state.p_pos[self.num_adversaries:self.num_adversaries + self.num_good_agents]

        delta = adv_pos[:, None, :] - prey_pos[None, :, :]
        dists = jnp.sqrt(jnp.sum(jnp.square(delta), axis=-1))

        min_dist_per_adv = jnp.min(dists, axis=-1)

        team_min_dist = jnp.min(min_dist_per_adv)

        dense_shaping = -team_min_dist * self.dense_reward_scale

        dense_rewards = {}
        for a in self.adversaries:
            dense_rewards[a] = sparse_rewards[a] + dense_shaping

        for a in self.good_agents:
            dense_rewards[a] = sparse_rewards[a]

        return dense_rewards

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        return super().get_obs(state)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]


SimpleTagScriptedPrey6v2 = lambda **kwargs: SimpleTagScriptedPreyMPE(
    num_adversaries=6, num_good_agents=2, num_obs=3, **kwargs
)
