# EFXPLORER: Multi-Agent Reinforcement Learning with Exploration

Code for the EFXPLORER paper.

## Setup

### 1. Create Conda Environment

```bash
conda env create -f ../environment.yml
conda activate efxplorer
```

### 2. Install Dependencies

```bash
pip install -e .
```

## Configurations

Configs are YAML files under each algorithm directory. Select one via `--config-name <name>` (no `.yaml`), or for COIN via Hydra's `+alg=<name>` group, and for HASAC via `--config=<name>`.

| Algorithm | Config directory | Available configs |
|-----------|------------------|--------------------|
| MAPPO-based | `algorithm/MAPPO/config/` | `efxplorer_corridor`, `efxplorer_tag`, `mappo_corridor`, `mappo_tag`, `mappo_linear_corridor`, `mappo_linear_tag`, `mappo_lagrangian_corridor`, `mappo_lagrangian_tag`, `efxplorer_smax`, `mappo_rnn_smax` |
| IPPO-based | `algorithm/IPPO/config/` | `ippo_corridor`, `ippo_tag`, `efxplorer_ippo_corridor`, `efxplorer_ippo_tag`, `efxplorer_ippo_smax`, `ippo_rnn_smax` |
| HASAC | `algorithm/HASAC/config/` | `HASAC_corridor`, `HASAC_tag`, `HASAC_smax` |
| COIN  | `algorithm/COIN/config/alg/` | `COIN_mpe`, `COIN_smax` |

## Running Experiments (JaxMARL MPE)

Run from this directory. Override any config value on the command line (e.g. `SEED=42 NUM_ENVS=256`).

### MAPPO-based Algorithms

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

### IPPO-based Algorithms

```bash
# 5. IPPO baseline
python algorithm/IPPO/ippo_rnn_mpe.py --config-name ippo_corridor
python algorithm/IPPO/ippo_rnn_mpe.py --config-name ippo_tag

# 6. EFXPLORER-IPPO
python algorithm/IPPO/efxplorer_ippo_mpe.py --config-name efxplorer_ippo_corridor
python algorithm/IPPO/efxplorer_ippo_mpe.py --config-name efxplorer_ippo_tag
```

### HASAC (Actor-Critic)

```bash
# 7. HASAC (use --config=HASAC_corridor or --config=HASAC_tag)
python algorithm/HASAC/HASAC_mpe.py --config=HASAC_corridor
python algorithm/HASAC/HASAC_mpe.py --config=HASAC_tag
```

### COIN (Q-learning + curiosity / influence intrinsic rewards)

```bash
# 8. COIN on MPE
python algorithm/COIN/COIN_mpe.py +alg=COIN_mpe
```

## Running Experiments (SMAX)

SMAX maps are resolved from `MAP_NAME` via `jaxmarl.environments.smax.map_name_to_scenario`.

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

### Overriding config values

```bash
python algorithm/MAPPO/efxplorer_mpe.py --config-name efxplorer_corridor SEED=42 NUM_ENVS=256 TOTAL_TIMESTEPS=1e7
```


## License

See LICENSE file for details.
