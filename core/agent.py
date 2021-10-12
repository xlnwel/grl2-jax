from core.decorator import *
from core.mixin.agent import StepCounter
from core.strategy import Strategy
from core.monitor import Monitor
from core.utils import save_code
from env.typing import EnvOutput
from utility.timer import Every, Timer


class AgentBase:
    """ Initialization """
    @config
    def __init__(self, 
                 *, 
                 env_stats, 
                 strategy: Strategy,
                 step_counter: StepCounter=None,
                 monitor: Monitor=None,
                 dataset=None):
        self.env_stats = env_stats
        self.strategy = strategy
        self.dataset = dataset
        self.step_counter = step_counter
        self.monitor = monitor

        self._post_init(env_stats, dataset)
        self.restore()
        save_code(self._root_dir, self._model_name)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        if hasattr(self.strategy, name):
            return getattr(self.strategy, name)
        elif hasattr(self.monitor, name):
            return getattr(self.monitor, name)
        elif hasattr(self.step_counter, name):
            return getattr(self.step_counter, name)
        raise AttributeError(f"Attempted to get missing attribute '{name}'")

    def _post_init(self, env_stats, dataset):
        """ Adds attributes to Agent """
        self._sample_timer = Timer('sample')
        self._learn_timer = Timer('train')

        self._return_stats = getattr(self, '_return_stats', False)

        self.RECORD = getattr(self, 'RECORD', False)
        self.N_EVAL_EPISODES = getattr(self, 'N_EVAL_EPISODES', 1)

        # intervals between calling self._summary
        self._to_summary = Every(self.LOG_PERIOD, self.LOG_PERIOD)

    def reset_states(self, states=None):
        pass

    def get_states(self):
        pass

    def _summary(self, data, terms):
        """ Adds non-scalar summaries here """
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

    """ Train """
    @step_track
    def train_record(self, step):
        train_step = self._sample_train()
        self._store_additional_stats()

        return train_step

    def _sample_train(self):
        raise NotImplementedError
    
    def _store_additional_stats(self):
        pass

    """ Checkpoint Ops """
    def restore(self):
        self.strategy.restore()
        self.step_counter.restore_step()

    def save(self, print_terminal_info=False):
        self.strategy.save(print_terminal_info)
        self.step_counter.save_step()


class AgentInterface:
    def __init__(self, name):
        self._name = name
        self.strategies = {}

    @property
    def name(self):
        return self._name

    def add_strategy(self, sid, strategy):
        self.strategies[sid] = strategy

    def save(self):
        pass
    
    def restore(self):
        pass
