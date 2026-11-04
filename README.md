# EFXPLORER: Cooperative Multi-Agent Reinforcement Learning via Epigraph-Form Guided Exploration

This repository provides the official implementation of EFXPLORER. [Project page](https://efxplorer-marl.github.io/).

## Environments

<table>
  <tr>
    <td align="center" width="25%"><img src="assets/balance.gif" width="176" alt="VMAS-balance"></td>
    <td align="center" width="25%"><img src="assets/give_way.gif" width="176" alt="VMAS-give way"></td>
    <td align="center" width="25%"><img src="assets/transport.gif" width="176" alt="VMAS-transport"></td>
    <td align="center" width="25%"><img src="assets/wheel.gif" width="176" alt="VMAS-wheel"></td>
  </tr>
  <tr>
    <td align="center">VMAS-balance</td>
    <td align="center">VMAS-give way</td>
    <td align="center">VMAS-transport</td>
    <td align="center">VMAS-wheel</td>
  </tr>
  <tr>
    <td align="center"><img src="assets/mpe_corridor.gif" width="176" alt="MPE-corridor"></td>
    <td align="center"><img src="assets/mpe_tag.gif" width="176" alt="MPE-tag"></td>
    <td align="center"><img src="assets/smax_3s5z.gif" width="176" alt="SMAX 3s5z_vs_3s6z"></td>
    <td align="center"><img src="assets/smax_5m.gif" width="176" alt="SMAX 5m_vs_6m"></td>
  </tr>
  <tr>
    <td align="center">MPE-corridor</td>
    <td align="center">MPE-tag</td>
    <td align="center">SMAX 3s5z_vs_3s6z</td>
    <td align="center">SMAX 5m_vs_6m</td>
  </tr>
</table>

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


## JaxMARL

### Configurations

Configs are YAML files under each algorithm directory. Select one via `--config-name <name>`, with the corresponding code.

| Algorithm | Config directory | Available configs |
|-----------|------------------|--------------------|
| MAPPO-based | `algorithm/MAPPO/config/` | `efxplorer_corridor`, `efxplorer_tag`, `mappo_corridor`, `mappo_tag`, `mappo_linear_corridor`, `mappo_linear_tag`, `mappo_lagrangian_corridor`, `mappo_lagrangian_tag`, `efxplorer_smax`, `mappo_rnn_smax` |
| IPPO-based | `algorithm/IPPO/config/` | `ippo_corridor`, `ippo_tag`, `efxplorer_ippo_corridor`, `efxplorer_ippo_tag`, `efxplorer_ippo_smax`, `ippo_rnn_smax` |
| HASAC | `algorithm/HASAC/config/` | `HASAC_corridor`, `HASAC_tag`, `HASAC_smax` |
| COIN  | `algorithm/COIN/config/alg/` | `COIN_mpe`, `COIN_smax` |

### Running Experiments (MPE)

```bash
# EFXPLORER-MAPPO
python algorithm/MAPPO/efxplorer_mpe.py --config-name efxplorer_corridor
python algorithm/MAPPO/efxplorer_mpe.py --config-name efxplorer_tag

# EFXPLORER-IPPO
python algorithm/IPPO/efxplorer_ippo_mpe.py --config-name efxplorer_ippo_corridor
python algorithm/IPPO/efxplorer_ippo_mpe.py --config-name efxplorer_ippo_tag
```

### Running Experiments (SMAX)

SMAX maps are resolved from `MAP_NAME`

```bash

# EFXPLORER-MAPPO
python algorithm/MAPPO/efxplorer_smax.py --config-name efxplorer_smax

# EFXPLORER-IPPO
python algorithm/IPPO/efxplorer_ippo_smax.py --config-name efxplorer_ippo_smax

```

#### Overriding config values

```bash
python algorithm/MAPPO/efxplorer_mpe.py --config-name efxplorer_corridor SEED=42 NUM_ENVS=256 TOTAL_TIMESTEPS=1e7
```

## License

See LICENSE file for details. 
