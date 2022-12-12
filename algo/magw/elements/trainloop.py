from core.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def sample_data(self):
        self.data = super().sample_data()
        return self.data
