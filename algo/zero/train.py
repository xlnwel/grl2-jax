from core.tf_config import configure_gpu, configure_precision, silence_tf_logs

from .run.pbt import pbt_train
from .run.ppo import ppo_train
from .run.bc import bc_train



def main(config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)

    if config['training'] == 'pbt':
        pbt_train(config)
    if config['training'] == 'ppo':
        ppo_train(config)
    elif config['training'] == 'bc':
        bc_train(config)
    else:
        raise ValueError(config['training'])