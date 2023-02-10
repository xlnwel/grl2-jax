from functools import partial

from algo.ppo.train import main
from algo.happo_lka3.train import train


def lookahead_optimize(agents, routine_config, aids):
    teammate_log_ratio = None
    for i, aid in enumerate(aids):
        agent = agents[aid]
        if i == 0:
            tlr = agent.fake_lookahead_train(teammate_log_ratio=teammate_log_ratio)
        else:
            tlr = agent.lookahead_train(teammate_log_ratio=teammate_log_ratio)
        if not routine_config.ignore_ratio_for_lookahead:
            teammate_log_ratio = tlr



train = partial(train, lka_opt_fn=lookahead_optimize)
main = partial(main, train=train)
