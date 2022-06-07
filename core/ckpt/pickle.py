import cloudpickle

from core.elements.agent import Agent
from core.typing import ModelPath


def set_weights_for_agent(
    agent: Agent, 
    model: ModelPath, 
    filename='params.pkl'
):
    path = '/'.join([model.root_dir, model.model_name, filename])
    with open(path, 'rb') as f:
        weights = cloudpickle.load(f)
        agent.set_weights(weights)
