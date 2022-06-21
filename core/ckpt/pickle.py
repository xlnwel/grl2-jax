import os
import cloudpickle

from core.elements.agent import Agent
from core.log import do_logging
from core.typing import ModelPath


def set_weights_for_agent(
    agent: Agent, 
    model: ModelPath, 
    filename='params.pkl'
):
    path = '/'.join([*model, filename])
    if os.path.exists(path):
        do_logging(f'Find file: {path}', level='pwt', backtrack=3)
        with open(path, 'rb') as f:
            weights = cloudpickle.load(f)
            agent.set_weights(weights)
    else:
        do_logging(f'No such file: {path}', level='pwt', backtrack=3)
