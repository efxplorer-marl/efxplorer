from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jaxmarl import make
from jaxmarl.wrappers.baselines import MPELogWrapper

from COIN._coin_core import single_run, tune


def env_factory(config):
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = MPELogWrapper(env)
    return env, config["ENV_NAME"]


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config.get("HYP_TUNE", False):
        tune(config, env_factory)
    else:
        single_run(config, env_factory)


if __name__ == "__main__":
    main()
