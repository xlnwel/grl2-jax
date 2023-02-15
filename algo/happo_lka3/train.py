from functools import partial
import numpy as np

from algo.ppo.train import main, train
from algo.happo_lka2.train import ego_run


def training_aids(all_aids, routine_config):
    aids = np.random.choice(
        all_aids, size=len(all_aids), replace=False, 
        p=routine_config.perm)
    return aids


train = partial(train, aids_fn=training_aids, ego_run_fn=ego_run)
main = partial(main, train=train)
