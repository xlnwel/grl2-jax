from core.log import do_logging
from core.typing import AttrDict
from algo.lka_common.elements.model import LOOKAHEAD
from algo.happo.elements.trainloop import TrainingLoop as TrainingLoopBase


class TrainingLoop(TrainingLoopBase):
    def _train(self, **kwargs):
        data = self.sample_data()
        if data is None:
            return 0, AttrDict()
        lka = data.pop(LOOKAHEAD)
        assert lka == False, lka
        stats = self._train_with_data(data, **kwargs)

        if isinstance(stats, tuple):
            assert len(stats) == 2, stats
            n, stats = stats
        else:
            n = 1

        return n, stats

    def lookahead_train(self, **kwargs):
        if self.config.n_lka_epochs:
            for _ in range(self.config.n_lka_epochs):
                data = self.sample_data()
                if data is None:
                    do_logging('Bypassing lookahead train')
                    return

                self.trainer.lookahead_train(data, **kwargs)
        else:
            data = self.sample_data()
            if data is None:
                do_logging('Bypassing lookahead train')
                return

            lka = data.pop(LOOKAHEAD)
            assert lka == True, lka
            self.trainer.lookahead_train(data, **kwargs)