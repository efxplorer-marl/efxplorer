#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Algorithm, AlgorithmConfig
from .ensemble import EnsembleAlgorithm, EnsembleAlgorithmConfig
from .ippo import Ippo, IppoConfig
from .mappo import Mappo, MappoConfig
from .masac import Masac, MasacConfig
from .efxplorer_mappo import EfxplorerMappo, EfxplorerMappoConfig
from .mappo_lagrangian import MappoLagrangian, MappoLagrangianConfig
from .efxplorer_ippo import EfxplorerIppo, EfxplorerIppoConfig
from .mappo_linear import MappoLinear, MappoLinearConfig
from .hasac import Hasac, HasacConfig
from .coin import Coin, CoinConfig
from .qmix import Qmix, QmixConfig

classes = [
    "Mappo",
    "MappoConfig",
    "EfxplorerMappo",
    "EfxplorerMappoConfig",
    "MappoLagrangian",
    "MappoLagrangianConfig",
    "MappoLinear",
    "MappoLinearConfig",
    "Ippo",
    "IppoConfig",
    "Masac",
    "MasacConfig",
    "Qmix",
    "QmixConfig",
    "Coin",
    "CoinConfig",
    "Hasac",
    "HasacConfig",
    "EfxplorerIppo",
    "EfxplorerIppoConfig",
]

# A registry mapping "algoname" to its config dataclass
# This is used to aid loading of algorithms from yaml
algorithm_config_registry = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "hasac": HasacConfig,
    "efxplorer_mappo": EfxplorerMappoConfig,
    "mappo_lagrangian": MappoLagrangianConfig,
    "mappo_linear": MappoLinearConfig,
    "coin": CoinConfig,
    "efxplorer_ippo": EfxplorerIppoConfig,
    "efxplorer_ippo_joint_individual": EfxplorerIppoConfig,
    "efxplorer_ippo_joint_team": EfxplorerIppoConfig,
    "efxplorer_mappo_joint_individual": EfxplorerMappoConfig,
    "efxplorer_mappo_joint_team": EfxplorerMappoConfig,
}
