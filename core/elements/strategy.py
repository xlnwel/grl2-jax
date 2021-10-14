from typing import Union

from core.elements.actor import Actor
from core.elements.model import Model, ModelEnsemble
from core.elements.trainer import Trainer, TrainerEnsemble
from core.mixin.strategy import StepCounter, TrainingLoopBase
from env.typing import EnvOutput
from utility.utils import config_attr


class Strategy:
    """ Initialization """
    def __init__(self, 
                 *, 
                 name,
                 config: dict,
                 model: Union[Model, ModelEnsemble], 
                 trainer: Union[Trainer, TrainerEnsemble], 
                 actor: Actor=None,
                 train_loop: TrainingLoopBase=None,
                 ):
        self._name = name
        config_attr(self, config)
        self.model = model
        self.trainer = trainer
        self.actor = actor
        self.step_counter = StepCounter(config.root_dir, config.model_name)
        self.train_loop = train_loop
        self._post_init()

    def _post_init():
        pass

    @property
    def name(self):
        return self._name

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        if hasattr(self.step_counter, name):
            return getattr(self.step_counter, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    def get_weights(self):
        weights = {
            f'{self._name}_model': self.model.get_weights(),
            f'{self._name}_opt': self.trainer.get_weights(),
            f'{self._name}_aux': self.actor.get_auxiliary_stats()
        }
        return weights

    def set_weights(self, weights):
        self.model.set_weights(weights[f'{self._name}_model'])
        self.trainer.set_weights(weights[f'{self._name}_opt'])
        self.actor.set_auxiliary_stats(weights[f'{self._name}_aux'])

    def train_record(self, step):
        n, stats = self.train_loop.train()
        self.step_counter.set_env_step(step)
        self.step_counter.set_train_step(self.step_counter.get_env_step() + n)

        return stats

    def reset_states(self, states=None):
        pass

    def get_states(self):
        pass

    """ Call """
    def __call__(self, 
                 env_output: EnvOutput, 
                 evaluation: bool=False,
                 return_eval_stats: bool=False):
        inp = self._prepare_input_to_actor(env_output)
        out = self.actor(inp, evaluation=evaluation, 
            return_eval_stats=return_eval_stats)
        self._record_output(out)
        return out[:2]

    def _prepare_input_to_actor(self, env_output):
        """ Extract data from env_output as the input 
        to Actor for inference """
        inp = env_output.obs
        return inp

    def _record_output(self, out):
        """ Record some data in out """
        pass

    """ Checkpoint Ops """
    def restore(self):
        self.trainer.restore()
        self.model.restore()
        self.actor.restore_auxiliary_stats()
        self.step_counter.restore_step()

    def save(self, print_terminal_info=False):
        self.trainer.save(print_terminal_info)
        self.model.save(print_terminal_info)
        self.actor.save_auxiliary_stats()
        self.step_counter.save_step()
