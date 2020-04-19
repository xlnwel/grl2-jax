import importlib


def get_package(algo, place=0, separator='.'):
    algo = algo.split('-', 1)[place]
    pkg = f'algo{separator}{algo}'

    return pkg

def import_module(name=None, *, config=None, algo=None):
    """ import module according to algo or algorithm in config """
    algo = algo or config['algorithm']
    assert isinstance(algo, str), algo
    pkg = get_package(algo)
    m = importlib.import_module(f'{pkg}.{name}')

    return m

def import_agent(config):
    algo = config['algorithm']
    algo = algo.rsplit('-', 1)[-1]
    pkg = get_package(algo=algo)
    nn = importlib.import_module(f'{pkg}.nn')
    agent = importlib.import_module(f'{pkg}.agent')

    return nn.create_model, agent.Agent
