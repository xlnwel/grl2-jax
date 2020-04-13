import tensorflow as tf

from utility.display import pwc


def restore(ckpt_manager, ckpt, ckpt_path, name='model'):
    """ Restore the latest parameter recorded by ckpt_manager

    Args:
        ckpt_manager: An instance of tf.train.CheckpointManager
        ckpt: An instance of tf.train.Checkpoint
        ckpt_path: The directory in which to write checkpoints
        name: optional name for print
    """
    path = ckpt_manager.latest_checkpoint
    if path:
        ckpt.restore(path)#.assert_consumed()
        pwc(f'Params for {name} are restored from "{path}".', color='cyan')
    else:
        pwc(f'No model for {name} is found at "{ckpt_path}"!', 
            f'Start training from scratch.', color='cyan')

def save(ckpt_manager, global_steps, steps, message='', print_terminal_info=True):
    """ Save model

    Args:
        ckpt_manager: An instance of tf.train.CheckpointManager
        global_steps: A tensor that records step
        steps: An int that assigns to global_steps. 
            If it's None, we leave global_steps unchanged
        message: optional message for print
    """
    if steps:
        global_steps.assign(steps)
    path = ckpt_manager.save()
    if print_terminal_info:
        pwc(f'Model saved at {path}: {message}', color='cyan')

def setup_checkpoint(ckpt_models, root_dir, model_name, global_steps, name='model'):
    """ Setup checkpoint

    Args:
        ckpt_models: A dict of models to save, including optimizers
        root_dir: The root directory for checkpoint
        model_name: The name of the model
    """
    # checkpoint & manager
    ckpt = tf.train.Checkpoint(step=global_steps, **ckpt_models)
    ckpt_path = f'{root_dir}/{model_name}/models'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, 5)
    restore(ckpt_manager, ckpt, ckpt_path, name)
    
    return ckpt, ckpt_path, ckpt_manager
