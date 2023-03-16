from functools import partial

from algo.ma_common.elements.strategy import Strategy as StrategyBase, create_strategy


class Strategy(StrategyBase):
    def lookahead_train(self, **kwargs):
        return self.train_loop.lookahead_train(**kwargs)


create_strategy = partial(create_strategy, strategy_cls=Strategy)
