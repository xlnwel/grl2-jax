from functools import partial
import numpy as np

from algo.ppo.train import main, train


def training_aids(all_aids, routine_config):
    aids = np.random.choice(
        all_aids, size=len(all_aids), replace=False, 
        p=routine_config.perm)
    return aids


train = partial(train, aids_fn=training_aids)
main = partial(main, train=train)
