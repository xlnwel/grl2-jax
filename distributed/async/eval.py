import numpy as np
import ray

from tools.timer import Timer

from .local.runner_manager import RunnerManager
from tools.ray_setup import sigint_shutdown_ray


def main(configs, n, **kwargs):
  ray.init()
  sigint_shutdown_ray()

  if configs[0].env.env_name.startswith('grf'):
    for c in configs:
      c.env.env_name = 'grf-11_vs_11_stochastic'
      c.env.number_of_right_players_agent_controls = 0
      c.env.render = True
      c.runner.n_steps = c.env.max_episode_steps = 3000

  runner = RunnerManager()
  runner.build_runners(configs, store_data=False, evaluation=True)
  
  runner.set_weights_from_configs(configs, wait=True)
  with Timer('eval') as et:
    steps, n_episodes, video, rewards, stats = runner.evaluate_and_return_stats(n)
  print(stats)

  # for f, r in zip(video[-100:], rewards[-100:]):
  #   print(f)
  #   print(r)

  config = configs[0]
  n_agents = config.n_agents
  n_runners = config.runner.n_runners
  n = n_episodes - n_episodes % n_runners
  for k, v in stats.items():
    for aid in range(n_agents):
      v = np.array(v[:n])
      pstd = np.std(np.mean(v.reshape(n_runners, -1), axis=-1)) * np.sqrt(n // n_runners)
      print(f'Agent{aid}: {k} averaged over {n_episodes} episodes: mean({np.mean(v):3g}), std({np.std(v):3g}), pstd({pstd:3g})')

  print(f'Evaluation time: total({et.total():3g}),',
    f'episode per second({et.total() / n_episodes:3g}),',
    f'steps per second({et.total() / steps})')

  ray.shutdown()
