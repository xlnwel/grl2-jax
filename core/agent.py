from abc import ABC
import logging

from core.checkpoint import *
from core.decorator import record, step_track
from core.mixin import StepCounter
from core.log import *
from utility.display import pwc
from utility.utils import Every
from utility.timer import Timer

logger = logging.getLogger(__name__)


class AgentImpl(ABC):
    @classmethod
    def as_remote(cls, **kwargs):
        import ray
        return ray.remote(**kwargs)(cls)

    """ Checkpoint Ops """
    def restore(self):
        """ Restore model """
        if getattr(self, 'trainer', None) is not None:
            self.trainer.restore()
        elif getattr(self, 'model', None) is not None:
            self.model.restore()

    def save(self, print_terminal_info=False):
        """ Save model """
        if getattr(self, 'trainer', None) is not None:
            self.trainer.save(print_terminal_info)
        elif getattr(self, 'model', None) is not None:
            self.model.save(print_terminal_info)

    """ Tensorboard Ops """
    def set_summary_step(self, step):
        """ Sets tensorboard step """
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        """ Adds scalar summary to tensorboard """
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        """ Adds histogram summary to tensorboard """
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def graph_summary(self, sum_type, *args, step=None):
        """ Adds graph summary to tensorboard
        Args:
            sum_type str: either "video" or "image"
            args: Args passed to summary function defined in utility.graph,
                of which the first must be a str to specify the tag in Tensorboard
        """
        assert isinstance(args[0], str), f'args[0] is expected to be a name string, but got "{args[0]}"'
        args = list(args)
        args[0] = f'{self.name}/{args[0]}'
        graph_summary(self._writer, sum_type, args, step=step)

    def video_summary(self, video, step=None):
        video_summary(f'{self.name}/sim', video, step=step)

    def save_config(self, config):
        """ Save config.yaml """
        save_config(self._root_dir, self._model_name, config)

    def print_construction_complete(self):
        pwc(f'{self.name.upper()} is constructed...', color='cyan')


class AgentBase(AgentImpl, StepCounter):
    """ Initialization """
    @record
    def __init__(self, *, env_stats, 
            model=None, trainer=None, dataset=None):        
        self.model = model
        self.trainer = trainer
        self.dataset = dataset

        self._post_init(env_stats, dataset)
        self.restore()

    def _post_init(self, env_stats, dataset):
        """ Adds attributes to Agent """
        self._sample_timer = Timer('sample')
        self._learn_timer = Timer('train')

        self._return_stats = getattr(self, '_return_stats', False)

        self.RECORD = getattr(self, 'RECORD', False)
        self.N_EVAL_EPISODES = getattr(self, 'N_EVAL_EPISODES', 1)

        # intervals between calling self._summary
        self._to_summary = Every(self.LOG_PERIOD, self.LOG_PERIOD)
        
        self._initialize_counter()

    def reset_states(self, states=None):
        pass

    def get_states(self):
        pass

    def _summary(self, data, terms):
        """ Adds non-scalar summaries here """
        pass 

    """ Call """
    def __call__(self, env_output, **kwargs):
        return self.model(env_output, **kwargs)

    """ Train """
    @step_track
    def learn_log(self, step):
        n = self._sample_learn()
        self._store_additional_stats()

        return n

    def _sample_learn(self):
        raise NotImplementedError
    
    def _store_additional_stats(self):
        pass

    """ Checkpoint Ops """
    def restore(self):
        """ Restore model """
        super().restore()
        self.restore_step()

    def save(self, print_terminal_info=False):
        """ Save model """
        super().save(print_terminal_info)
        self.save_step()
