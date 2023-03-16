from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def lookahead_train(self):
        data = self.sample_data()

        self.trainer.lookahead_train(data)
