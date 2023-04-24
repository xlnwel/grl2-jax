import numpy as np

from tools.rms import RunningMeanStd


class TemporaryRMS:
    def __init__(self, key, axis):
        self.key = key
        self.rms = RunningMeanStd(axis, name=f'{key}_rms')
    
    def retrieve_rms(self):
        rms = self.rms.get_rms_stats() \
            if self.rms.is_initialized() else None
        self.rms.reset_rms_stats()
        return rms
    
    def update_obs_rms(self, trajs):
        self.rms.update(
            np.stack([traj[self.key] for traj in trajs]), 
        )
