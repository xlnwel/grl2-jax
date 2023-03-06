from tools.timer import Timer
from algo.masac.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def sample_data(self):
        with Timer('sample'):
            data = self.buffer.sample(primal_percentage=1)
        if data is None:
            return None
        data.setdefault('global_state', data.obs)
        if 'next_obs' in data:
            data.setdefault('next_global_state', data.next_obs)
        return data
