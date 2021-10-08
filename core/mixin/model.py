import logging
import itertools
import tensorflow as tf

from core.log import *
from utility.timer import Every


logger = logging.getLogger(__name__)


""" Model Mixins """
class TargetNetOps:
    def _setup_target_net_sync(self):
        self._to_sync = Every(self._target_update_period) \
            if hasattr(self, '_target_update_period') else None

    @tf.function
    def _sync_nets(self):
        """ Synchronizes the target net with the online net """
        ons = self.get_online_nets()
        tns = self.get_target_nets()
        do_logging(f"Sync Networks | Online networks: {[n.name for n in ons]}", logger=logger)
        do_logging(f"Sync Networks | Target networks: {[n.name for n in tns]}", logger=logger)
        ovars = list(itertools.chain(*[v.variables for v in ons]))
        tvars = list(itertools.chain(*[v.variables for v in tns]))
        logger.info(f"Sync Networks | Online network parameters:\n\t" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in ovars]))
        logger.info(f"Sync Networks | Target network parameters:\n\t" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in tvars]))
        assert len(tvars) == len(ovars), f'{tvars}\n{ovars}'
        [tvar.assign(ovar) for tvar, ovar in zip(tvars, ovars)]

    @tf.function
    def _update_nets(self):
        """ Updates the target net towards online net using exponentially moving average """
        ons = self.get_online_nets()
        tns = self.get_target_nets()
        do_logging(f"Update Networks | Online networks: {[n.name for n in ons]}", logger=logger)
        do_logging(f"Update Networks | Target networks: {[n.name for n in tns]}", logger=logger)
        ovars = list(itertools.chain(*[v.variables for v in ons]))
        tvars = list(itertools.chain(*[v.variables for v in tns]))
        logger.info(f"Update Networks | Online network parameters:\n" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in ovars]))
        logger.info(f"Update Networks | Target network parameters:\n" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in tvars]))
        assert len(tvars) == len(ovars), f'{tvars}\n{ovars}'
        [tvar.assign(self._polyak * tvar + (1. - self._polyak) * mvar) 
            for tvar, mvar in zip(tvars, ovars)]

    def get_online_nets(self):
        """ Gets the online networks """
        return [getattr(self, f'{k}') for k in self.model 
            if f'target_{k}' in self.model]

    def get_target_nets(self):
        """ Gets the target networks """
        return [getattr(self, f'target_{k}') for k in self.model 
            if f'target_{k}' in self.model]
