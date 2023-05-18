from core.log import do_logging
from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def train(self, step, **kwargs):
        train_step, stats = super().train(step, **kwargs)
        self.trainer.sync_lookahead_params()

        return train_step, stats

    def lookahead_train(self, **kwargs):
        if self.config.n_lka_epochs:
            for _ in range(self.config.n_lka_epochs):
                data = self.sample_data(record_data=False)
                if data is None:
                    do_logging('Bypassing lookahead train')
                    return

                stats = self.trainer.lookahead_train(data, **kwargs)
        else:
            data = self.sample_data(record_data=False)
            if data is None:
                do_logging('Bypassing lookahead train')
                return

            stats = self.trainer.lookahead_train(data, **kwargs)
        return stats
