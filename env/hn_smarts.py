import pathlib

import gym

import smarts

gym.logger.set_level(40)

from functools import partial
from typing import Dict, Sequence, Tuple

import argparse

from smarts import sstudio
from smarts.core.agent_interface import (
  OGM,
  RGB,
  AgentInterface,
  DoneCriteria,
  DrivableAreaGridMap,
  NeighborhoodVehicles,
  Waypoints,
)
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from smarts.env.hiway_env import HiWayEnv
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.parallel_env import ParallelEnv
from smarts.zoo.agent_spec import AgentSpec
from env.smarts_rc.wrappers import baseline
from tools import pkg


SCENARIO_DIR = 'env/smarts_rc/scenarios'

def get_scenario(config):
  scenario = '/'.join([SCENARIO_DIR, config["scenario"]])
  scenario = pathlib.Path(scenario).absolute()
  return scenario


def build_agent_specs(config):
  scenario_path = '/'.join([SCENARIO_DIR, config["scenario"]])
  pkg_name = scenario_path.replace('/', '.')
  scenario_module = pkg.import_module('scenario', pkg_name)

  # total_mission = Scenario.discover_agent_missions_count(scenario)
  missions = scenario_module.missions
  total_mission = len(missions)
  
  agent_ids = [f'AGENT-{i}' for i in range(total_mission)]

  interface = {
    "max_episode_steps": config.setdefault('max_episode_steps', 1000), 
    "neighborhood_vehicles": NeighborhoodVehicles(
      **config.setdefault("neighborhood_vehicles", {'radius': 50})
    ), 
    "waypoints": Waypoints(**config.setdefault("waypoints", {'lookahead': 50})), 
    "action": ActionSpaceType(config.setdefault('action_type', 1))
  }

  if config.get("rgb"):
    interface["rgb"] = RGB(**config["rgb"])

  if config.get("ogm"):
    interface["ogm"] = OGM(**config["ogm"])

  agent_specs = {
    aid: AgentSpec(
      interface=AgentInterface(**interface),
    )
    for aid in agent_ids
  }
  return agent_specs


def make(config):
  scenario = get_scenario(config)
  agent_specs = build_agent_specs(config)
  sim_name = f'{config.env_name}_{config.seed}'
  env = HiWayEnv(
    scenarios=[scenario], 
    agent_specs=agent_specs, 
    sim_name=sim_name, 
    headless=config.get('headless', True), 
    sumo_headless=config.get('sumo_headless', True), 
  )
  env = baseline.FrameStack(env, config)
  env = baseline.NormalizeAndFlatten(env)

  return env

class ChaseViaPointsAgent(Agent):
  def act(self, obs: Sequence[Observation]) -> Tuple[float, int]:
    # Here, we only utilise the newest frame from the stacked observations.
    newest_obs = obs[-1]
    speed_limit = newest_obs.waypoint_paths[0][0].speed_limit
    return (speed_limit, 0)

def main(
  scenarios: Sequence[str],
  sim_name: str,
  headless: bool,
  seed: int,
  num_agents: int,
  num_stack: int,
  num_env: int,
  auto_reset: bool,
  max_episode_steps: int = 128,
  num_steps: int = 1280,
  num_episodes: int = 10,
):
  from core.typing import AttrDict, dict2AttrDict
  config = AttrDict({
    'env_name': 'smarts-intersections', 
    'scenario': 'intersections/4lane', 
  })
  def _make(i):
    new_config = dict2AttrDict(config, to_copy=True)
    new_config.seed = i
    env = make(config)
    return env
  
  # A list of env constructors of type `Callable[[], gym.Env]`
  env_constructors = [
    partial(_make, i) for i in range(num_env)
  ]

  # Build multiple agents
  agent_ids = [f"Agent_{i}" for i in range(num_agents)]
  agents = {
    agent_id: ChaseViaPointsAgent()
    for agent_id in agent_ids
  }

  # Create parallel environments
  env = ParallelEnv(
    env_constructors=env_constructors,
    auto_reset=auto_reset,
    seed=seed,
  )

  if auto_reset:
    parallel_env_async(agents, env, num_env, num_steps)
  else:
    parallel_env_sync(agents, env, num_env, num_episodes)

import numpy as np
def parallel_env_async(
  agents: Dict[str, Agent], env: gym.Env, num_env: int, num_steps: int
):
  """Parallel environments with asynchronous episodes. Run multiple environments
  in parallel with `auto_reset=True`. Individual environments will automatically
  reset when their episode ends. Episodes start asynchronously in each environment.
  Args:
    agents (Dict[str, Agent]): Ego agents.
    env (gym.Env): Gym env.
    num_env (int): Number of environments.
    num_steps (int): Number of steps to step the environment.
  """

  batched_dones = [{"__all__": False} for _ in range(num_env)]
  batched_observations = env.reset()

  for _ in range(num_steps):
    # Compute actions for all active(i.e., not done) agents
    batched_actions = [np.random.randint(0, 4, (4)) for _ in batched_observations]
    # for observations, dones in zip(batched_observations, batched_dones):
    #   actions = {
    #     agent_id: agents[agent_id].act(agent_obs)
    #     for agent_id, agent_obs in observations.items()
    #     if not dones.get(agent_id, False)
    #     or dones[
    #       "__all__"
    #     ]  # `dones[__all__]==True` implies the env was auto-reset in previous iteration
    #   }
    #   batched_actions.append(actions)

    # Step all environments in parallel
    batched_observations, batched_rewards, batched_dones, batched_infos = env.step(
      batched_actions
    )

  env.close()


def parallel_env_sync(
  agents: Dict[str, Agent], env: gym.Env, num_env: int, num_episodes: int
):
  """Parallel environments with synchronous episodes. Run multiple environments
  in parallel with `auto_reset=False`. All environments are reset together when
  all their episodes have finished. New episodes start synchronously in all
  environments.
  Args:
    agents (Dict[str, Agent]): Ego agents.
    env (gym.Env): Gym env.
    num_env (int): Number of parallel environments.
    num_episodes (int): Number of episodes.
  """

  for _ in range(num_episodes):
    batched_dones = [{"__all__": False} for _ in range(num_env)]
    batched_observations = env.reset()

    # Iterate until all environments complete an episode each.
    while not all(dones["__all__"] for dones in batched_dones):
      # Compute actions for all active(i.e., not done) agents
      batched_actions = []
      for observations, dones in zip(batched_observations, batched_dones):
        actions = {
          agent_id: agents[agent_id].act(agent_obs)
          for agent_id, agent_obs in observations.items()
          if not dones.get(agent_id, False)
        }
        batched_actions.append(actions)

      # Step all environments in parallel
      (
        batched_observations,
        batched_rewards,
        batched_dones,
        batched_infos,
      ) = env.step(batched_actions)

  env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--scenarios",
    default=[],
    type=list,
  )
  parser.add_argument(
    "--sim_name",
    default='par_env',
    type=str,
  )
  parser.add_argument(
    "--headless",
    default=True,
    type=bool,
    help="Number of ego agents to simulate in each environment.",
  )
  parser.add_argument(
    "--num-agents",
    default=2,
    type=int,
    help="Number of consecutive frames to stack in each environment's observation.",
  )
  parser.add_argument(
    "--num-stack",
    default=2,
    type=int,
    help="Number of consecutive frames to stack in each environment's observation.",
  )
  parser.add_argument(
    "--num-env",
    default=2,
    type=int,
    help="Number of parallel environments to simulate.",
  )
  parser.add_argument(
    "--max-episode-steps",
    default=128,
    type=int,
    help="Maximum number of steps per episode.",
  )
  parser.add_argument(
    "--num-steps",
    default=1280,
    type=int,
    help="Total number of steps to simulate per environment in parallel asynchronous simulation.",
  )
  parser.add_argument(
    "--seed",
    default=0,
    type=int,
  )
  args = parser.parse_args()

  if not args.sim_name:
    args.sim_name = "par_env"

  if not args.scenarios:
    print(pathlib.Path('env/smarts_rc').absolute())
    print(pathlib.Path(smarts.__file__).parents[0])
    # breakpoint()
    args.scenarios = [
      str(
        pathlib.Path('env/smarts_rc').absolute()
        / "scenarios"
        / "sumo"
        / "intersections"
        / "2lane"
      )
    ]
    print(args.scenarios)
    # breakpoint()

  sstudio.build_scenario(args.scenarios)

  print("\nParallel environments with asynchronous episodes.")
  main(
    scenarios=args.scenarios,
    sim_name=f"{args.sim_name}_async",
    headless=args.headless,
    seed=args.seed,
    num_agents=args.num_agents,
    num_stack=args.num_stack,
    num_env=args.num_env,
    auto_reset=True,
    max_episode_steps=args.max_episode_steps,
    num_steps=args.num_steps,
  )

  print("\nParallel environments with synchronous episodes.\n")
  main(
    scenarios=args.scenarios,
    sim_name=f"{args.sim_name}_sync",
    headless=args.headless,
    seed=args.seed,
    num_agents=args.num_agents,
    num_stack=args.num_stack,
    num_env=args.num_env,
    auto_reset=False,
    max_episode_steps=args.max_episode_steps,
    num_episodes=args.episodes,
  )
