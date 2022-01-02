import functools

from core.elements.trainer import Trainer, create_trainer
from core.decorator import override
from core.tf_config import build
from utility import pkg


class PPOTrainer(Trainer):
    @override(Trainer)
    def _build_train(self, env_stats):
        algo = self.config.algorithm.split('-')[-1]
        get_data_format = pkg.import_module(
            'elements.utils', algo=algo).get_data_format
        # Explicitly instantiate tf.function to avoid unintended retracing
        TensorSpecs = get_data_format(self.config, env_stats, self.loss.model)
        self.train = build(self.train, TensorSpecs)
        return True

    def raw_train(self, obs, action, value, traj_ret, 
            advantage, logpi, state=None, mask=None):
        tape, loss, terms = self.loss.loss(
            obs, action, value, traj_ret, 
            advantage, logpi, state, mask)
        terms['norm'] = self.optimizer(tape, loss)

        return terms


create_trainer = functools.partial(create_trainer,
    name='ppo', trainer_cls=PPOTrainer
)
