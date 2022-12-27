from .environment import MultiAgentEnv
from .scenarios import load
from .pretrained import PretrainedTag, RandomTag, FrozenTag

def MPEEnv(config):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    name = config['env_name']
    scenario = load(name + ".py").Scenario()
    # create world
    world = scenario.make_world(config)
    # create multiagent environment
    env = MultiAgentEnv(
        world, 
        scenario.reset_world,
        scenario.reward, 
        scenario.observation, 
        scenario.info,
        **config)

    if config.get("pretrained_wrapper", None) == "pretrained_tag":
        env = PretrainedTag(env)
    elif config.get("pretrained_wrapper", None) == "random_tag":
        env = RandomTag(env)
    elif config.get("pretrained_wrapper", None) == "frozen_tag":
        env = FrozenTag(env)
    else:
        pass

    return env
