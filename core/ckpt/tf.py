import tensorflow as tf

from core.typing import ModelPath
from utility.display import pwc


def restore_ckpt(ckpt_manager, ckpt, ckpt_path, name='model', version='latest'):
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

def save_ckpt(ckpt_manager, print_terminal_info=True):
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


class TFCheckpoint:
    def __init__(self, config, ckpt_model, name):
        self._model_path = ModelPath(config.get('root_dir', None), config.get('model_name', None))
        self._ckpt_kwargs = config.get('ckpt_kwargs', {})
        self._ckptm_kwargs = config.get('ckptm_kwargs', {})
        self._ckpt_model = ckpt_model
        self._name = name
        self._has_ckpt = None not in self._model_path

    """ Save & Restore Model """
    def reset_model_path(self, model_path: ModelPath):
        self._model_path = model_path
        self.setup_checkpoint(force=True)
        self._has_ckpt = True

    def setup_checkpoint(self, force=False):
        if force or not hasattr(self, 'ckpt'):
            self.ckpt, self.ckpt_path, self.ckpt_manager = \
                setup_checkpoint(
                    self._ckpt_model, self._model_path.root_dir, 
                    self._model_path.model_name, name=self._name, 
                    ckpt_kwargs=self._ckpt_kwargs,
                    ckptm_kwargs=self._ckptm_kwargs,
                )

    def save(self, print_terminal_info=True):
        if self._has_ckpt:
            self.setup_checkpoint()
            save_ckpt(self.ckpt_manager, print_terminal_info)
        else:
            raise RuntimeError(
                'Cannot perform <save> as either root_dir or model_name was not specified at initialization')

    def restore(self):
        if self._has_ckpt:
            self.setup_checkpoint()
            restore_ckpt(self.ckpt_manager, self.ckpt, self.ckpt_path, self._name)
        else:
            raise RuntimeError(
                'Cannot perform <restore> as either root_dir or model_name was not specified at initialization')
