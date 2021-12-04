from typing import Tuple, Dict
import tensorflow as tf

from run.utils import set_path
from utility.utils import config_attr
from utility.tf_utils import numpy2tensor, tensor2numpy


class Actor:
    def __init__(self, *, config, model, name):
        self._raw_name = name
        self._name = f'{name}_actor'
        self.config = config_attr(self, config, filter_dict=True)
        
        self.model = model
        
        self._post_init()

    @property
    def name(self):
        return self._name

    def _post_init(self):
        pass

    def reset(self, root_dir, model_name):
        self._root_dir = root_dir
        self._model_name = model_name
        self.config = set_path(self.config, root_dir, model_name)
        self.setup_checkpoint()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.model, name)

    def __call__(self, 
                 inp: dict,
                 evaluation: bool=False, 
                 return_eval_stats: bool=False):
        """ The interface to interact with the environment
        
        Args:
            inp: input to the calling method
            evaluation: evaluation mode or not
            return_eval_stats: if return evaluation stats
        Return:
            (action, terms, rnn_state)
        """
        inp, tf_inp = self._process_input(inp, evaluation)
        tf_inp = self._add_eval(
            tf_inp, 
            evaluation=evaluation, 
            return_eval_stats=return_eval_stats)
        
        out = self.model.action(**tf_inp)
        out = self._process_output(inp, out, evaluation)

        return out

    """ Overwrite the following methods if necessary """
    def _process_input(self, inp: dict, evaluation: bool):
        """ Processes input to Model at the algorithmic level 
        
        Args:
            inp: input to the model
            evaluation bool: evaluation mode or not
        Returns: 
            processed input to <model.action>
        """
        return inp, numpy2tensor(inp)

    def _add_eval(self, tf_inp, evaluation, return_eval_stats):
        if not self.model.to_build:
            tf_inp.update({
                'evaluation': evaluation,
                'return_eval_stats': return_eval_stats
            })
        return tf_inp

    def _process_output(self, 
                        inp: dict, 
                        out: Tuple[tf.Tensor, Dict[str, tf.Tensor]], 
                        evaluation: bool):
        """ Post-processes output. By default, 
        we convert tf.Tensor to np.ndarray
        
        Args:
            inp: Pre-processed inputs
            out: Model output
        Returns:
            (action, terms, rnn_state)
        """
        action, terms, state = out
        if state is not None:
            action, terms, prev_state = tensor2numpy((action, terms, inp['state']))
        else:
            action, terms = tensor2numpy((action, terms))
        if not evaluation and state is not None:
            terms.update({
                'mask': inp['mask'], 
                **prev_state._asdict(),
            })
        return action, terms, state

    def get_weights(self, identifier=None):
        if identifier is None:
            identifier = self._raw_name
        weights = {
            f'{identifier}_model': self.model.get_weights(),
            f'{identifier}_aux': self.get_auxiliary_stats()
        }
        return weights

    def set_weights(self, weights, identifier=None):
        if identifier is None:
            identifier = self._raw_name
        self.model.set_weights(weights[f'{identifier}_model'])
        if f'{identifier}_aux' in weights:
            self.set_auxiliary_stats(weights[f'{identifier}_aux'])

    def get_model_weights(self, name: str=None):
        return self.model.get_weights(name)

    def set_model_weights(self, weights):
        self.model.set_weights(weights)

    def get_auxiliary_stats(self):
        pass
    
    def set_auxiliary_stats(self):
        pass

    def save_auxiliary_stats(self):
        pass
    
    def restore_auxiliary_stats(self):
        pass

    def save(self, print_terminal_info=False):
        self.model.save(print_terminal_info)
        self.save_auxiliary_stats()
    
    def restore(self):
        self.model.restore()
        self.restore_auxiliary_stats()
