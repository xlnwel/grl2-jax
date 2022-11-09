from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def imaginary_train(self):
        data = self.sample_data()

        self.trainer.imaginary_train(data)
