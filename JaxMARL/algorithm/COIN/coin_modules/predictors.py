from typing import Dict, Tuple, Any
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax


class CuriosityPredictorOS(nn.Module):
    hidden_dim: int
    obs_output_dim: int
    state_output_dim: int

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        h = nn.relu(h)
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(h)
        h = nn.relu(h)

        pred_obs = nn.Dense(
            self.obs_output_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(h)

        pred_state = nn.Dense(
            self.state_output_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(h)

        return pred_obs, pred_state


class InfluencePredictorWithAction(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        h = nn.relu(h)
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(h)
        h = nn.relu(h)
        pred = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(h)
        return pred


class InfluencePredictorWithoutAction(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        h = nn.relu(h)
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(h)
        h = nn.relu(h)
        pred = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(h)
        return pred


def get_log_pi(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    mse = jnp.square(pred - target)
    log_prob = -1.0 * jnp.sum(mse, axis=-1)
    return log_prob


class COINPredictorTrainState(TrainState):
    target_params: Any = None


def create_coin_predictor_states(
    rng: jax.random.PRNGKey,
    config: Dict,
    n_agents: int,
    rnn_hidden_dim: int,
    n_actions: int,
    obs_dim: int,
    state_dim: int,
) -> Tuple[TrainState, TrainState, TrainState]:
    hidden_dim = config.get("COIN_HIDDEN_DIM", 64)
    predictor_lr = config.get("COIN_PREDICTOR_LR", 3e-4)
    max_grad_norm = config.get("MAX_GRAD_NORM", 25)
    optim = (
        config.get("COIN_PREDICTOR_OPTIMIZER") or config.get("OPTIMIZER") or "adam"
    ).lower()

    if optim == "rmsprop":
        rms_decay = float(config.get("RMSPROP_DECAY", 0.99))
        rms_eps = float(config.get("RMSPROP_EPS", 1e-5))
        opt = optax.rmsprop(predictor_lr, decay=rms_decay, eps=rms_eps)
    else:
        opt = optax.adam(predictor_lr)

    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), opt)

    rng_os, rng_with, rng_without = jax.random.split(rng, 3)

    os_input_dim = (rnn_hidden_dim + n_actions) * n_agents
    os_obs_output_dim = obs_dim * n_agents
    os_state_output_dim = state_dim

    predict_os = CuriosityPredictorOS(
        hidden_dim=hidden_dim,
        obs_output_dim=os_obs_output_dim,
        state_output_dim=os_state_output_dim,
    )
    dummy_os_input = jnp.zeros((1, os_input_dim))
    os_params = predict_os.init(rng_os, dummy_os_input)
    predict_os_state = COINPredictorTrainState.create(
        apply_fn=predict_os.apply,
        params=os_params,
        tx=tx,
        target_params=os_params,
    )

    with_input_dim = rnn_hidden_dim + n_actions + n_actions
    predict_with = InfluencePredictorWithAction(
        hidden_dim=hidden_dim,
        output_dim=obs_dim,
    )
    dummy_with_input = jnp.zeros((1, with_input_dim))
    with_params = predict_with.init(rng_with, dummy_with_input)
    predict_with_state = COINPredictorTrainState.create(
        apply_fn=predict_with.apply,
        params=with_params,
        tx=tx,
        target_params=with_params,
    )

    without_input_dim = rnn_hidden_dim + n_actions
    predict_without = InfluencePredictorWithoutAction(
        hidden_dim=hidden_dim,
        output_dim=obs_dim,
    )
    dummy_without_input = jnp.zeros((1, without_input_dim))
    without_params = predict_without.init(rng_without, dummy_without_input)
    predict_without_state = COINPredictorTrainState.create(
        apply_fn=predict_without.apply,
        params=without_params,
        tx=tx,
        target_params=without_params,
    )

    return predict_os_state, predict_with_state, predict_without_state


def update_predictor_targets(
    predict_os_state: COINPredictorTrainState,
    predict_with_state: COINPredictorTrainState,
    predict_without_state: COINPredictorTrainState,
) -> Tuple[COINPredictorTrainState, COINPredictorTrainState, COINPredictorTrainState]:
    predict_os_state = predict_os_state.replace(target_params=predict_os_state.params)
    predict_with_state = predict_with_state.replace(target_params=predict_with_state.params)
    predict_without_state = predict_without_state.replace(target_params=predict_without_state.params)
    return predict_os_state, predict_with_state, predict_without_state
