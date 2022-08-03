import logging
import itertools
import tensorflow as tf

from core.log import *


logger = logging.getLogger(__name__)


""" Model Mixins """
class NetworkSyncOps:
    def __init__(self, config={}):
        self.config = config

    @tf.function
    def sync_target_nets(self):
        ons = self.get_online_nets()
        tns = self.get_target_nets()
        self.sync_nets(ons, tns)

    @tf.function
    def sync_nets(self, source, target):
        """ Synchronizes the target net with the online net """
        do_logging(f"Sync Networks | Source Networks: {[n.name for n in source]}", 
            logger=logger, level='print', backtrack=20)
        do_logging(f"Sync Networks | Target Networks: {[n.name for n in target]}", 
            logger=logger, level='print', backtrack=20)
        svars = sum([v.variables for v in source], ())
        tvars = sum([v.variables for v in target], ())
        self.sync_vars(svars, tvars)

    @tf.function
    def sync_vars(self, svars, tvars):
        do_logging(f"Sync Parameters | Source Parameters:\n\t" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in svars]), 
            logger=logger, level='print', backtrack=20)
        do_logging(f"Sync Parameters | Target Parameters:\n\t" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in tvars]), 
            logger=logger, level='print', backtrack=20)
        assert len(tvars) == len(svars), f'{tvars}\n{svars}'
        [tvar.assign(ovar) for tvar, ovar in zip(tvars, svars)]

    @tf.function
    def update_target_nets(self):
        ons = self.get_online_nets()
        tns = self.get_target_nets()
        self.update_nets(ons, tns)

    @tf.function
    def update_nets(self, source, target):
        """ Updates the target net towards online net using exponentially moving average """
        do_logging(f"Update Networks | Source Networks: {[n.name for n in source]}", 
            logger=logger, level='print', backtrack=20)
        do_logging(f"Update Networks | Target Networks: {[n.name for n in target]}", 
            logger=logger, level='print', backtrack=20)
        svars = list(itertools.chain(*[v.variables for v in source]))
        tvars = list(itertools.chain(*[v.variables for v in target]))
        self.update_vars(svars, tvars)

    @tf.function
    def update_vars(self, svars, tvars):
        do_logging(f"Update Networks | Source Parameters:\n" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in svars]), 
            logger=logger, level='print', backtrack=20)
        do_logging(f"Update Networks | Target Parameters:\n" 
            + '\n\t'.join([f'{n.name}, {n.shape}' for n in tvars]), 
            logger=logger, level='print', backtrack=20)
        assert len(tvars) == len(svars), f'{tvars}\n{svars}'
        [tvar.assign(self._polyak * tvar + (1. - self._polyak) * mvar) 
            for tvar, mvar in zip(tvars, svars)]

    def get_online_nets(self):
        """ Gets the online networks """
        return [getattr(self, f'{k}') for k in self.model 
            if f'target_{k}' in self.model]

    def get_target_nets(self):
        """ Gets the target networks """
        return [getattr(self, f'target_{k}') for k in self.model 
            if f'target_{k}' in self.model]
