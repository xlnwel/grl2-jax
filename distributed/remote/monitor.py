from core.mixin.strategy import StepCounter
from core.monitor import Monitor
from distributed.remote.base import RayBase
from utility.utils import AttrDict2dict, config_attr


class RemoteMonitor(RayBase):
    def __init__(self, config, name):
        self.config = config_attr(self, config)
        self.monitor = Monitor(
            self.config.root_dir, 
            self.config.model_name, 
            name,
        )
        self.step_counter = StepCounter(
            self.config.root_dir, self.config.model_name, f'{name}_step_counter'
        )
        self.step_counter.restore_step()

    def store_stats(self, **stats):
        if 'env_steps' in stats:
            self.step_counter.add_env_step(stats['env_steps'])
        if 'train_steps' in stats:
            self.step_counter.add_train_step(stats['train_steps'])

        self.monitor.store(**stats)

    def record(self):
        self.monitor.record(self.step_counter.get_env_step(), adaptive=False)
        self.step_counter.save_step()

    def is_over(self):
        return self.step_counter.get_env_step() > self.config.monitor.MAX_STEPS


def create_central_monitor(config):
    config = AttrDict2dict(config)
    buffer = RemoteMonitor.as_remote().remote(
        config,
        name='monitor'
    )
    return buffer
