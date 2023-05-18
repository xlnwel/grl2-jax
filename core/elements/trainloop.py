from core.elements.trainer import TrainerBase
from core.typing import AttrDict, dict2AttrDict
from tools.timer import Timer


class TrainingLoop:
    def __init__(
        self, 
        config: AttrDict, 
        buffer, 
        trainer: TrainerBase, 
        **kwargs
    ):
        self.config = dict2AttrDict(config, to_copy=True)
        self.buffer = buffer
        self.trainer = trainer
        self.model = self.trainer.model
        self.rng = self.model.rng

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.post_init()

    def post_init(self):
        pass

    def train(self, step, warm_up_stage=False, **kwargs):
        self._before_train(step)

        if warm_up_stage:
            train_step = 0
            for _ in range(self.config.n_wp_epochs):
                n, stats = self._warm_up_train()
                train_step += n
            return train_step, stats

        n_epochs = self.config.n_epochs
        if n_epochs:
            train_step = 0
            for _ in range(n_epochs):
                n, stats = self._train(**kwargs)
                train_step += n
        else:
            train_step, stats = self._train(**kwargs)
        if train_step != 0:
            stats = self._after_train(stats)

        return train_step, stats

    def _before_train(self, step):
        pass

    def _warm_up_train(self):
        pass

    def _train(self, **kwargs):
        data = self.sample_data()
        if data is None:
            return 0, AttrDict()
        stats = self._train_with_data(data, **kwargs)

        if isinstance(stats, tuple):
            assert len(stats) == 2, stats
            n, stats = stats
        else:
            n = 1

        return n, stats

    def _after_train(self, stats):
        return stats

    def sample_data(self, batch_size=None, record_data=True):
        with Timer('sample_data'):
            data = self.buffer.sample(batch_size=batch_size)
        if record_data:
            self.training_data = data
        if data is None:
            return None
        data.setdefault('global_state', data.obs)
        if 'next_obs' in data:
            data.setdefault('next_global_state', data.next_obs)

        return data

    def _train_with_data(self, data, **kwargs):
        if data is None:
            return AttrDict()
        stats = self.trainer.train(data, **kwargs)
        return stats

    def change_buffer(self, buffer):
        old_buffer = self.buffer
        self.buffer = buffer
        return old_buffer
