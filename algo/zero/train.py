from core.tf_config import configure_gpu, configure_precision, silence_tf_logs


def main(config):
    # from core.utils import save_config
    # config.name = 'zero'
    # save_config(config.root_dir, config.model_name, config)
    # assert False
    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)

    if config['training'] == 'pbt':
        from .run.pbt import main
        main(config)
    elif config['training'] == 'ppo':
        from .run.ppo import main
        main(config)
    elif config['training'] == 'bc':
        from .run.bc import main
        main(config)
    else:
        raise ValueError(config['training'])
