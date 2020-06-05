import importlib


def get_package(algo, place=0, separator='.'):
    algo = algo.split('-', 1)[place]
    pkg = f'algo{separator}{algo}'

    return pkg

def import_module(name=None, algo=None, *, config=None, place=0):
    """ import module according to algo or algorithm in config """
    algo = algo or config['algorithm']
    assert isinstance(algo, str), algo
    pkg = get_package(algo, place=place)
    m = importlib.import_module(f'{pkg}.{name}')

    return m

def import_agent(algo=None, *, config=None):
    algo = algo or config['algorithm']
    assert isinstance(algo, str), algo
    pkg = get_package(algo=algo, place=-1)
    nn = importlib.import_module(f'{pkg}.nn')
    agent = importlib.import_module(f'{pkg}.agent')

    return nn.create_model, agent.Agent

def import_main(module, algo=None, *, config=None):
    algo = algo or config['algorithm']
    assert isinstance(algo, str), algo
    pkg = get_package(algo, place={'train': 0, 'eval': -1}[module])
    m = importlib.import_module(f'{pkg}.{module}')

    return m.main