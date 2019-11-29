import tensorflow as tf

from utility.display import pwc


def setup_checkpoint(ckpt_models, model_root_dir, model_name):
    """ Setup checkpoint

    Args:
        ckpt_models: A dict of models to save, including optimizers
        model_root_dir: The root directory for checkpoint
        model_name: The name of the model
    """
    # checkpoint & manager
    global_steps = tf.Variable(1)
    ckpt = tf.train.Checkpoint(step=global_steps, **ckpt_models)
    ckpt_path = f'{model_root_dir}/{model_name}'
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, 5)

    return global_steps, ckpt, ckpt_path, ckpt_manager

def restore(ckpt_manager, ckpt, ckpt_path, name='model'):
    """ Restore the latest parameter recorded by ckpt_manager

    Args:
        ckpt_manager: An instance of tf.train.CheckpointManager
        ckpt: An instance of tf.train.Checkpoint
        ckpt_path: The directory in which to write checkpoints
        name: optional name for print
    """
    path = ckpt_manager.latest_checkpoint
    ckpt.restore(path).assert_consumed()
    if path:
        pwc(f'Params for {name} are restored from "{path}".', color='cyan')
    else:
        pwc(f'No model for {name} is found at "{ckpt_path}"!', color='magenta')
        pwc(f'Continue or Exist (c/e):', color='magenta')
        ans = input()
        if ans.lower() == 'e':
            import sys
            sys.exit()
        else:
            pwc(f'Start training from scratch.', color='magenta')

def save(ckpt_manager, global_steps, steps, message=''):
    """ Save model

    Args:
        ckpt_manager: An instance of tf.train.CheckpointManager
        global_steps: A tensor that records step
        steps: An int that assigns to global_steps
        message: optional message for print
    """
    global_steps.assign(steps)
    path = ckpt_manager.save()
    pwc(f'Model saved at {path}: {message}', color='cyan')
