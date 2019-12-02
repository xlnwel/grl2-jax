import tensorflow as tf
from utility.logger import Logger


""" Logging """
def save_config(logger, config):
    logger.save_config(config)

def log(logger, writer, model_name, step, timing='Train', print_terminal_info=True):
    stats = dict(
        model_name=f'{model_name}',
        timing=timing,
        steps=f'{step}'
    )
    stats.update(logger.get_stats())
    log_summary(writer, stats, step)
    log_stats(logger, stats, print_terminal_info=print_terminal_info)

def log_stats(logger, stats, print_terminal_info=True):
    [logger.log_tabular(k, v) for k, v in stats.items()]
    logger.dump_tabular(print_terminal_info=print_terminal_info)

def set_summary_step(step):
    tf.summary.experimental.set_step(step)

def log_summary(writer, stats, step=None):
    with writer.as_default():
        for k, v in stats.items():
            if isinstance(v, str):
                continue
            if tf.rank(v).numpy() == 0:
                tf.summary.scalar(f'stats/{k}', v, step=step)
            else:
                v = tf.convert_to_tensor(v, dtype=tf.float32)
                tf.summary.scalar(f'stats/{k}_mean', tf.reduce_mean(v), step=step)
                tf.summary.scalar(f'stats/{k}_std', tf.math.reduce_std(v), step=step)

    writer.flush()

def store(logger, **kwargs):
    logger.store(**kwargs)

def get_stats(logger, mean=True, std=False, min=False, max=False):
    return logger.get_stats(mean=mean, std=std, min=min, max=max)

def get_value(logger, key, mean=True, std=False, min=False, max=False):
    return logger.get(key, mean=mean, std=std, min=min, max=max)

""" Functions for setup logging """                
def setup_logger(root_dir, model_name):
    # logger save stats in f'{root_dir}/{model_name}/logs/log.txt'
    logger = Logger(f'{root_dir}/{model_name}/logs')
    return logger

def setup_tensorboard(root_dir, model_name):
    # writer for tensorboard summary
    # stats are saved in directory f'{root_dir}/{model_name}'
    writer = tf.summary.create_file_writer(f'{root_dir}/{model_name}/logs')
    return writer
