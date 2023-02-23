from functools import partial

from algo.masac.train import *


def lookahead_train(agent, model, buffer, model_buffer, routine_config, 
        n_runs, run_fn, opt_fn):
    if not model_buffer.ready_to_sample():
        return
    for _ in range(n_runs):
        run_fn(agent, model, buffer, model_buffer, routine_config)


train = partial(train, lka_train_fn=lookahead_train)
main = partial(main, train=train)
