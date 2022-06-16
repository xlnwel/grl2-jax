import copy
from typing import Tuple, Union

from core.elements.actor import Actor
from core.elements.model import Model
from core.elements.trainer import Trainer, TrainerEnsemble
from core.elements.trainloop import TrainingLoopBase
from core.mixin.strategy import StepCounter
from core.typing import ModelPath
from env.typing import EnvOutput
from utility.utils import set_path
from utility.typing import AttrDict


class Strategy:
    """ Initialization """
    def __init__(
        self, 
        *, 
        name: str,
        config: AttrDict,
        env_stats: AttrDict,
        trainer: Union[Trainer, TrainerEnsemble]=None, 
        actor: Actor=None,
        train_loop: TrainingLoopBase=None,
    ):
        self._name = name
        self.config = config
        self.env_stats = env_stats
        if trainer is None and actor is None:
            raise RuntimeError('Neither trainer nor actor is provided')

        self.model: Model = actor.model if trainer is None else trainer.model
        self.trainer: Trainer = trainer
        self.actor: Actor = actor
        self.train_loop: TrainingLoopBase = train_loop

        if self.config.get('root_dir'):
            self._model_path = ModelPath(
                self.config.root_dir, 
                self.config.model_name
            )
            self.step_counter = StepCounter(
                self._model_path, 
                name=f'{self._name}_step_counter'
            )

        self._post_init()

    def _post_init(self):
        pass

    @property
    def is_trainable(self):
        return self.trainer is not None

    @property
    def name(self):
        return self._name

    def reset_model_path(self, model_path: ModelPath):
        self._model_path = model_path
        self.step_counter = StepCounter(
            model_path, 
            name=f'{self._name}_step_counter'
        )
        self.config = set_path(self.config, model_path, max_layer=0)
        if self.model is not None:
            self.model.reset_model_path(model_path)
        if self.actor is not None:
            self.actor.reset_model_path(model_path)
        if self.trainer is not None:
            self.trainer.reset_model_path(model_path)

    def get_model_path(self):
        return self._model_path

    def __getattr__(self, name):
        # Do not expose the interface of independent elements here. 
        # Invoke them directly instead
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        elif hasattr(self.step_counter, name):
            return getattr(self.step_counter, name)
        elif self.train_loop is not None and hasattr(self.train_loop, name):
            return getattr(self.train_loop, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    def get_weights(self, module_name=None, opt_weights=True, aux_stats=True,
            train_step=False, env_step=False):
        weights = {}
        if self.model is not None:
            weights[f'model'] = self.model.get_weights(module_name)
        if self.trainer is not None and opt_weights:
            weights[f'opt'] = self.trainer.get_optimizer_weights()
        if self.actor is not None and aux_stats:
            weights[f'aux'] = self.actor.get_auxiliary_stats()
        if train_step:
            weights[f'train_step'] = self.step_counter.get_train_step()
        if env_step:
            weights[f'env_step'] = self.step_counter.get_env_step()

        return weights

    def set_weights(self, weights):
        if 'model' in weights:
            self.model.set_weights(weights['model'])
        if 'opt' in weights and self.trainer is not None:
            self.trainer.set_optimizer_weights(weights['opt'])
        if 'aux' in weights:
            self.actor.set_auxiliary_stats(weights['aux'])
        if 'train_step' in weights:
            self.step_counter.set_train_step(weights['train_step'])
        if 'env_step' in weights:
            self.step_counter.set_env_step(weights['env_step'])

    def train_record(self):
        n, stats = self.train_loop.train()
        self.step_counter.add_train_step(n)
        return stats

    def reset_states(self, states=None):
        pass

    def get_states(self):
        pass

    """ Call """
    def __call__(
        self, 
        env_output: EnvOutput, 
        evaluation: bool=False,
        return_eval_stats: bool=False
    ):
        inp = self._prepare_input_to_actor(env_output)
        out = self.actor(inp, evaluation=evaluation, 
            return_eval_stats=return_eval_stats)
        
        self._record_output(out)
        return out[:2]

    def _prepare_input_to_actor(self, env_output):
        """ Extract data from env_output as the input 
        to Actor for inference """
        inp = copy.deepcopy(env_output.obs)
        return inp

    def _record_output(self, out: Tuple):
        """ Record some data in out """
        pass

    """ Checkpoint Ops """
    def restore(self, skip_actor=False, skip_trainer=False):
        if self.model is not None:
            self.model.restore()
        if not skip_actor and self.actor is not None:
            self.actor.restore_auxiliary_stats()
        if not skip_trainer and self.trainer is not None:
            self.trainer.restore_optimizer()
        self.step_counter.restore_step()

    def save(self, print_terminal_info=False):
        if self.model is not None:
            self.model.save(print_terminal_info)
        if self.actor is not None:
            self.actor.save_auxiliary_stats()
        if self.trainer is not None:
            self.trainer.save_optimizer(print_terminal_info)
        self.step_counter.save_step()


def create_strategy(
        name, 
        config: AttrDict,
        env_stats: AttrDict, 
        actor: Actor=None,
        trainer: Union[Trainer, TrainerEnsemble]=None, 
        dataset=None,
        *,
        strategy_cls,
        training_loop_cls=None
    ):
    if trainer is not None:
        if dataset is None:
            raise ValueError('Missing dataset')
        if training_loop_cls is None:
            raise ValueError('Missing TrainingLoop Class')
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
        env_stats=env_stats,
        trainer=trainer,
        actor=actor,
        train_loop=train_loop
    )

    return strategy
