# EFXPLORER: Cooperative Multi-Agent Reinforcement Learning via Epigraph-Form Guided Exploration

Code for the EFXPLORER paper.

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate efxplorer
```
### 2. Clone VMAS
```bash
git clone --branch 1.5.2 https://github.com/proroklab/VectorizedMultiAgentSimulator.git
cp BenchMARL/vmas_patch/*.py VectorizedMultiAgentSimulator/vmas/scenarios/
```

### 3. Install Dependencies 

```bash
pip install -e ./VectorizedMultiAgentSimulator
pip install -e ./BenchMARL
pip install -e ./JaxMARL
```

## BenchMARL Configurations

Run from the BenchMARL directory. Override any config value on the command line (e.g. `seed=0 num_envs=256`). Use comma-separated values for Hydra multirun.

```bash
python benchmarl/run.py algorithm=<algo> task=<task> experiment=<experiment>
```

**Example (single run):**
```bash
python benchmarl/run.py algorithm=efxplorer_mappo task=vmas/balance experiment=efxplorer_experiment
```

**Example (multirun over algorithms and tasks):**
```bash
python benchmarl/run.py -m algorithm=efxplorer_mappo,efxplorer_ippo,mappo task=vmas/balance,vmas/transport experiment=efxplorer_experiment
```

**Available configurations:**

| Option | Key | Values |
|--------|-----|--------|
| Algorithm | `algorithm` | `efxplorer_mappo`, `efxplorer_ippo`, `mappo_lagrangian`, `mappo_linear`, `hasac`, `mappo`, `ippo`, `coin` |
| Task | `task` | `vmas/balance`, `vmas/give_way`, `vmas/transport`, `vmas/wheel` |
| Experiment | `experiment` | `efxplorer_experiment`, `hasac_experiment`, `others_experiment` |

In the paper, `efxplorer_mappo`, `efxplorer_ippo` and `mappo_lagrangian` use `efxplorer_experiment`, `hasac` uses `hasac_experiment`, and the other algorithms use `others_experiment`. For `efxplorer_mappo` on `transport` and `give_way`, add `algorithm.z_ema_alpha=0.4`, and on `give_way` also `algorithm.critic_int_lr=1e-4`.


## JaxMARL

### Configurations

Configs are YAML files under each algorithm directory. Select one via `--config-name <name>` (no `.yaml`), or for COIN via Hydra's `+alg=<name>` group, and for HASAC via `--config=<name>`.

| Algorithm | Config directory | Available configs |
|-----------|------------------|--------------------|
| MAPPO-based | `algorithm/MAPPO/config/` | `efxplorer_corridor`, `efxplorer_tag`, `mappo_corridor`, `mappo_tag`, `mappo_linear_corridor`, `mappo_linear_tag`, `mappo_lagrangian_corridor`, `mappo_lagrangian_tag`, `efxplorer_smax`, `mappo_rnn_smax` |
| IPPO-based | `algorithm/IPPO/config/` | `ippo_corridor`, `ippo_tag`, `efxplorer_ippo_corridor`, `efxplorer_ippo_tag`, `efxplorer_ippo_smax`, `ippo_rnn_smax` |
| HASAC | `algorithm/HASAC/config/` | `HASAC_corridor`, `HASAC_tag`, `HASAC_smax` |
| COIN  | `algorithm/COIN/config/alg/` | `COIN_mpe`, `COIN_smax` |

### Running Experiments (JaxMARL MPE)

Run from the JaxMARL directory. Override any config value on the command line (e.g. `SEED=42 NUM_ENVS=256`).

#### MAPPO-based Algorithms

```bash
# 1. EFXPLORER-MAPPO
python algorithm/MAPPO/efxplorer_mpe.py --config-name efxplorer_corridor
python algorithm/MAPPO/efxplorer_mpe.py --config-name efxplorer_tag

# 2. MAPPO baseline
python algorithm/MAPPO/mappo_rnn_mpe.py --config-name mappo_corridor
python algorithm/MAPPO/mappo_rnn_mpe.py --config-name mappo_tag

# 3. MAPPO-Linear
python algorithm/MAPPO/mappo_linear_mpe.py --config-name mappo_linear_corridor
python algorithm/MAPPO/mappo_linear_mpe.py --config-name mappo_linear_tag

# 4. MAPPO-Lagrangian
python algorithm/MAPPO/mappo_lagrangian_mpe.py --config-name mappo_lagrangian_corridor
python algorithm/MAPPO/mappo_lagrangian_mpe.py --config-name mappo_lagrangian_tag
```

#### IPPO-based Algorithms

```bash
# 5. IPPO baseline
python algorithm/IPPO/ippo_rnn_mpe.py --config-name ippo_corridor
python algorithm/IPPO/ippo_rnn_mpe.py --config-name ippo_tag

# 6. EFXPLORER-IPPO
python algorithm/IPPO/efxplorer_ippo_mpe.py --config-name efxplorer_ippo_corridor
python algorithm/IPPO/efxplorer_ippo_mpe.py --config-name efxplorer_ippo_tag
```

#### HASAC (Actor-Critic)

```bash
# 7. HASAC (use --config=HASAC_corridor or --config=HASAC_tag)
python algorithm/HASAC/HASAC_mpe.py --config=HASAC_corridor
python algorithm/HASAC/HASAC_mpe.py --config=HASAC_tag
```

#### COIN (Q-learning + curiosity / influence intrinsic rewards)

```bash
# 8. COIN on MPE
python algorithm/COIN/COIN_mpe.py +alg=COIN_mpe
```

### Running Experiments (SMAX)

SMAX maps are resolved from `MAP_NAME` via `jaxmarl.environments.smax.map_name_to_scenario`. The paper uses `3s5z_vs_3s6z`, `5m_vs_6m`, `10m_vs_11m` and `smacv2_10_units` (`MAP_NAME=5m_vs_6m`, or `alg.MAP_NAME=5m_vs_6m` for COIN).

```bash
# COIN
python algorithm/COIN/COIN_smax.py +alg=COIN_smax

# HASAC
python algorithm/HASAC/HASAC_smax.py --config=HASAC_smax

# EFXPLORER-MAPPO
python algorithm/MAPPO/efxplorer_smax.py --config-name efxplorer_smax

# EFXPLORER-IPPO
python algorithm/IPPO/efxplorer_ippo_smax.py --config-name efxplorer_ippo_smax

# MAPPO / IPPO
python algorithm/MAPPO/mappo_rnn_smax.py --config-name mappo_rnn_smax
python algorithm/IPPO/ippo_rnn_smax.py --config-name ippo_rnn_smax
```

#### Overriding config values

```bash
python algorithm/MAPPO/efxplorer_mpe.py --config-name efxplorer_corridor SEED=42 NUM_ENVS=256 TOTAL_TIMESTEPS=1e7
```


## Paper Experiments

Default configs follow Tables 2-6 of the paper. We use 5 seeds (3 for Figure 10). Ablations change only these values:

| Experiment | BenchMARL | JaxMARL |
|------------|-----------|---------|
| Budget rate (Fig. 4, 14, Table 9) | `algorithm.z_ema_alpha` | `ALPHA_Z` |
| Joint-state variant (Fig. 4, 12) | `algorithm.joint_intrinsic_mode=team` | |
| MAPPO-Linear weight (Fig. 5, 14, 15, Table 9) | `algorithm.intrinsic_lambda` | `INTRINSIC_REWARD_BETA` |
| MAPPO-Lagrangian initial multiplier (Fig. 5, Table 9) | `algorithm.lambda_init` | `LAMBDA_INIT` |
| Softmin (Fig. 13) | `algorithm.use_soft_min=true algorithm.soft_min_tau=<tau>` | |
| No budget update (Table 11) | `algorithm.z_ema_alpha=0` | |

Pass JaxMARL values as floats (`LAMBDA_INIT=1.0`, not `1`).

Notes:
- The JaxMARL EFXPLORER, MAPPO-Linear and MAPPO-Lagrangian trainers update on the full rollout (no minibatches).
- SMAX EFXPLORER-IPPO uses one optimizer for the actor and the extrinsic critic, so `CRITIC_LR` is not used.
- In VMAS, 4000 is the minibatch size and the value loss is not clipped.
- In MPE, the layer after the GRU has width `GRU_HIDDEN_DIM` (256).


## License

See LICENSE file for details. BenchMARL (MIT), JaxMARL (Apache-2.0) and `BenchMARL/vmas_patch/` (GPLv3) keep their own licenses; see NOTICE.
