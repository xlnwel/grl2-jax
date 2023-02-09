from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def lookahead_train(self, **kwargs):
        data = self.sample_data()

        return self.trainer.lookahead_train(data, **kwargs)
