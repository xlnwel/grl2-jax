from core.tf_config import configure_gpu, configure_precision, silence_tf_logs


def main(config):
    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)

    if config['training'] == 'pbt':
        from .run.pbt import main
        main(config)
    if config['training'] == 'ppo':
        from .run.ppo import main
        main(config)
    elif config['training'] == 'bc':
        from .run.bc import main
        main(config)
    else:
        raise ValueError(config['training'])
