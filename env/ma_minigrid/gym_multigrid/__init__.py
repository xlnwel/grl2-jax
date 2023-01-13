from .envs import CollectGameEnv, SoccerGameEnv, TreasureGameEnv

REGISTRY = {
    "collect": CollectGameEnv,
    "soccer": SoccerGameEnv,
    "treasure": TreasureGameEnv,
}