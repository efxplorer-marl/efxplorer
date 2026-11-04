from .environments import (
    SimpleTagScriptedPrey6v2,
    SimpleCorridorMPE,
    SMAX,
    HeuristicEnemySMAX,
    LearnedPolicyEnemySMAX,
)


def make(env_id: str, **env_kwargs):
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    if env_id == "MPE_simple_tag_scripted_prey_6v2":
        env = SimpleTagScriptedPrey6v2(**env_kwargs)
    elif env_id == "MPE_simple_corridor_v3":
        env = SimpleCorridorMPE(**env_kwargs)
    elif env_id == "SMAX":
        env = SMAX(**env_kwargs)
    elif env_id == "HeuristicEnemySMAX":
        env = HeuristicEnemySMAX(**env_kwargs)
    elif env_id == "LearnedPolicyEnemySMAX":
        env = LearnedPolicyEnemySMAX(**env_kwargs)

    return env


registered_envs = [
    "MPE_simple_tag_scripted_prey_6v2",
    "MPE_simple_corridor_v3",
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
]
