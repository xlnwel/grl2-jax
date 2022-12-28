from pathlib import Path
import collections
import numpy as np

from env.make import make_smarts
from tools.display import print_dict_info
from core.typing import AttrDict



def main(config):
    env = make_smarts(config)
    all_obs = collections.defaultdict(list)
    obs = env.reset()
    # for aid, o in obs.items():
    #     # for o in oo:
    #     for k, v in o.items():
    #         all_obs[k].append(v)

    for i in range(1000):
        action = env.random_action()
        obs, reward, discount, reset = env.step(action)
        # for o in obs.values():
        #     for k, v in o.items():
        #         all_obs[k].append(v)
        if np.all(reset):
            print(i, 'reset', discount)
            print(env.info())
            env.reset()
    all_obs = {k: np.stack(v) for k, v in all_obs.items()}

    print_dict_info(obs)
    print(env.info())


if __name__ == '__main__':
    config = AttrDict({
        'env_name': 'smarts', 
        'scenario': 'intersections/4lane', 

        'neighborhood_vehicles': {'radius': 50}, 
        'waypoints': {'lookahead': 50}, 
        # 'frame_stack': 3, 

        # 'goal_relative_pos': True, 
        # # distance to the center of lane
        # 'distance_to_center': True, 
        # # speed
        # 'speed': True, 
        # # steering
        # 'steering': True, 
        # # a list of heading errors
        # 'heading_errors': [20, 'continuous'], 
        # # at most eight neighboring vehicles' driving states
        # 'neighbor': 8, 

        # 'action_type': 1, 
    })
    main(config)
