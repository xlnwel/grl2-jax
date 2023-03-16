from core.elements.actor import Actor as ActorBase, apply_rms_to_inp
from core.mixin.actor import RMS
from tools.utils import batch_dicts
from algo.ma_common.run import concat_along_unit_dim


class Actor(ActorBase):
    def setup_checkpoint(self):
        if self.rms:
            self.config.rms.model_path = self._model_path
            self.rms = [RMS(self.config.rms)
                for _ in range(self.model.env_stats.n_agents)
            ]

    def __call__(self, inps, evaluation):
        inps = self._process_input(inps, evaluation)
        out = self.model.action(inps, evaluation)
        inp = batch_dicts(inps, concat_along_unit_dim)
        out = self._process_output(inp, out, evaluation)
        return out

    def _process_input(self, inps, evaluation):
        if self.rms is not None:
            inps = [apply_rms_to_inp(
                inp, rms, 
                self.config.get('update_obs_rms_at_execution', False)
            ) for inp, rms in zip(inps, self.rms)]
        return inps

    def get_auxiliary_stats(self):
        if self.rms:
            return [rms.get_rms_stats() for rms in self.rms]

    def set_auxiliary_stats(self, rms_stats):
        if self.rms:
            [rms.set_rms_stats(rms_stats) for rms in self.rms]

    def save_auxiliary_stats(self):
        """ Save the RMS and the model """
        if self.rms:
            for rms in self.rms:
                rms.save_rms()

    def restore_auxiliary_stats(self):
        """ Restore the RMS and the model """
        if self.rms:
            for rms in self.rms:
                rms.restore_rms()


def create_actor(config, model, name='actor'):
    return Actor(config=config, model=model, name=name)
