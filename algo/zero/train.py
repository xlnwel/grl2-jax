from core.tf_config import configure_gpu, configure_precision, silence_tf_logs
from core.typing import ModelPath
from utility import pkg


def main(config):
    # from core.utils import save_config
    # config.name = 'zero'
    # save_config(ModelPath(config.root_dir, config.model_name), config)
    # assert False
    silence_tf_logs()
    configure_gpu()
    configure_precision(config.precision)

    algo = config['algorithm']
    main = pkg.import_module(name=config['training'], pkg=f'algo.{algo}.run').main
    main(config)
