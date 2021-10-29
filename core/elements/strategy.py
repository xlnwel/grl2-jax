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
        self.train_loop = train_loop
        self.step_counter = StepCounter(
            config.root_dir, config.model_name, f'{name}_step_counter')
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

    def get_weights(self, identifier=None):
        if identifier is None:
            identifier = self._name
        weights = {}
        if self.model is not None:
            weights[f'{identifier}_model'] = self.model.get_weights()
        if self.trainer is not None:
            weights[f'{identifier}_opt'] = self.trainer.get_optimizer_weights()
        if self.actor is not None:
            weights[f'{identifier}_aux'] = self.actor.get_auxiliary_stats()

        return weights

    def set_weights(self, weights, identifier=None):
        if identifier is None:
            identifier = self._name
        if f'{identifier}_model' in weights:
            self.model.set_weights(weights[f'{identifier}_model'])
        if f'{identifier}_opt' in weights:
            self.trainer.set_optimizer_weights(weights[f'{identifier}_opt'])
        if f'{identifier}_aux' in weights:
            self.actor.set_auxiliary_stats(weights[f'{identifier}_aux'])

    def train_record(self):
        n, stats = self.train_loop.train()
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
        self.trainer.restore_optimizer()
        self.model.restore()
        self.actor.restore_auxiliary_stats()
        self.step_counter.restore_step()

    def save(self, print_terminal_info=False):
        self.trainer.save_optimizer(print_terminal_info)
        self.model.save(print_terminal_info)
        self.actor.save_auxiliary_stats()
        self.step_counter.save_step()


def create_strategy(
        name, 
        config, 
        model, 
        trainer, 
        actor, 
        dataset=None,
        *,
        strategy_cls,
        training_loop_cls=None
    ):
    if training_loop_cls is not None:
        if dataset is None:
            raise ValueError('Missing dataset')
    
        train_loop = training_loop_cls(
            config=config.train_loop, 
            dataset=dataset, 
            trainer=trainer
        )
    else:
        train_loop = None

    strategy = strategy_cls(
        name=name,
        config=config,
        model=model,
        trainer=trainer,
        actor=actor,
        train_loop=train_loop
    )

    return strategy
