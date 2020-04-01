import tensorflow as tf


class Module(tf.Module):
    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]

        
class Ensemble(tf.Module):
    """ This class groups all models used by off-policy algorithms together
    so that one can easily get and set all variables """
    def __init__(self, 
                 *,
                 models=None, 
                 model_fn=None, 
                 **kwargs):
        super().__init__()
        if models is None:
            self.models = model_fn(**kwargs)
        else:
            self.models = models

    def get_weights(self, name=None):
        """ Return a list/dict of weights
        Returns:
            If name is provided, it returns a dict of weights for models specified by keys.
            Otherwise it returns a list of all weights
        """
        if name is None:
            return [v.numpy() for v in self.variables]
        elif isinstance(name, str):
            name = [name]
        assert isinstance(name, (tuple, list))

        return dict((n, self.models[n].get_weights()) for n in name)

    def set_weights(self, weights):
        """Set weights 
        Args:
            weights: a dict or list of weights. If it's a dict, 
            it sets weights for models specified by the keys.
            Otherwise, it sets all weights 
        """
        if isinstance(weights, dict):
            for n, w in weights.items():
                self[n].set_weights(w)
        else:
            assert len(self.variables) == len(weights)
            [v.assign(w) for v, w in zip(self.variables, weights)]
    
    """ Auxiliary functions that make Ensemble like a dict """
    def __setitem__(self, key, item):
        self.models[key] = item

    def __getitem__(self, key):
        return self.models[key]

    def __len__(self):
        return len(self.models)
    
    def __iter__(self):
        return self.models.__iter__()

    def keys(self):
        return self.models.keys()

    def values(self):
        return self.models.values()
    
    def items(self):
        return self.models.items()
