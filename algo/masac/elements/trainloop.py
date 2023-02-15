from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def train(self, step, **kwargs):
        self._before_train(step)
        for _ in range(self.config.n_epochs):
            _, stats = self._train(**kwargs)

        self._after_train()

        return self.config.n_epochs, stats

    def lookahead_train(self, **kwargs):
        for _ in range(self.config.n_lka_epochs):
            data = self.sample_data()
            if data is None:
                return

            self.trainer.lookahead_train(data, **kwargs)
