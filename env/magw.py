from .grid_env.magw import MultiAgentGridWorldEnv

env_map = {
    'staghunt': MultiAgentGridWorldEnv,
    'escalation': MultiAgentGridWorldEnv,
}
