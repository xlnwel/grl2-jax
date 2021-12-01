import tensorflow as tf

from utility.display import pwc


def restore(ckpt_manager, ckpt, ckpt_path, name='model', version='latest'):
    """ Restores the latest parameter recorded by ckpt_manager

    Args:
        ckpt_manager: An instance of tf.train.CheckpointManager
        ckpt: An instance of tf.train.Checkpoint
        ckpt_path: The directory in which to write checkpoints
        name: optional name for print
    """
    if version == 'latest':
        path = ckpt_manager.latest_checkpoint
    elif version == 'oldest':
        path = ckpt_manager.checkpoints[0]
    else:
        if isinstance(version, int):
            path = ckpt_manager.checkpoints[-version]
        else:
            raise ValueError(f'Invalid version({version})')
    if path:
        ckpt.restore(path)#.assert_consumed()
        pwc(f'Params for {name} are restored from "{path}".', color='cyan')
    else:
        pwc(f'No model for {name} is found at "{ckpt_path}"!', color='cyan')
    return bool(path)

def save(ckpt_manager, print_terminal_info=True):
    """ Saves model

    Args:
        ckpt_manager: An instance of tf.train.CheckpointManager
        message: optional message for print
    """
    path = ckpt_manager.save()
    if print_terminal_info:
        pwc(f'Model saved at {path}', color='cyan')

def setup_checkpoint(ckpt_models, root_dir, model_name, name='model', ckpt_kwargs={}, ckptm_kwargs={}):
    """ Setups checkpoint

    Args:
        ckpt_models: A dict of models to save, including optimizers
        root_dir: The root directory for checkpoint
    """
    if not model_name:
        model_name = 'baseline'
    if 'max_to_keep' not in ckptm_kwargs:
        ckptm_kwargs['max_to_keep'] = 5
    # checkpoint & manager
    ckpt = tf.train.Checkpoint(**ckpt_models, **ckpt_kwargs)
    ckpt_path = f'{root_dir}/{model_name}/{name}'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, **ckptm_kwargs)
    
    return ckpt, ckpt_path, ckpt_manager
