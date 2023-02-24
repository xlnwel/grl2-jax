from functools import partial

from algo.masac.train import *


def lookahead_optimize(agent):
    return


train = partial(train, lka_opt_fn=lookahead_optimize)
main = partial(main, train=train)
