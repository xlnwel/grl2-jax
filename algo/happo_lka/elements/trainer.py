from functools import partial
import numpy as np
import jax

from core.log import do_logging
from core.elements.trainer import create_trainer
from core.typing import AttrDict
from core import optimizer
from algo.lka_common.elements.model import LOOKAHEAD
from algo.happo.elements.trainer import Trainer as TrainerBase, pop_lookahead


class Trainer(TrainerBase):
    def add_attributes(self):
        super().add_attributes()

    def compile_train(self):
        _jit_train = jax.jit(self.theta_train, 
            static_argnames=['aid', 'compute_teammate_log_ratio'])
        def jit_train(*args, **kwargs):
            self.rng, rng = jax.random.split(self.rng)
            return _jit_train(*args, rng=rng, **kwargs)
        self.jit_train = jit_train
        self.jit_lka_train = jit_train
        
        self.haiku_tabulate()

create_trainer = partial(create_trainer,
    name='happo_lka', trainer_cls=Trainer
)
