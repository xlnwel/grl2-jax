from core.log import do_logging
from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def _train(self, **kwargs):
        data = self.sample_data()
        stats = self._train_with_data(data)

        if isinstance(stats, tuple):
            assert len(stats) == 2, stats
            n, stats = stats
        else:
            n = self.trainer.config.n_epochs * self.trainer.config.n_mbs

        return n, stats

    def lookahead_train(self, **kwargs):
        data = self.sample_data()
        if data is None:
            do_logging('Bypassing lookahead train')

        return self.trainer.lookahead_train(data, **kwargs)
