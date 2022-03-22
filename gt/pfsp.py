from typing import List, Tuple
import numpy as np

from core.typing import ModelPath


class PFSP:
    def __init__(self, **kwargs):
        pass

    def __call__(
        aid: int, 
        model: ModelPath, 
        payoffs: List[np.ndarray], 
        model2sid: List[Tuple[ModelPath, int]], 
        sid2model: List[Tuple[int, ModelPath]], 
    ):
        """ Prioritized Fictitous Self-Play """
        pass
