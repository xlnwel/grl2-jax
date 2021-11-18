import logging
import time
import ray

from core.dataset import *
from core.log import do_logging

logger = logging.getLogger(__name__)


class  RayDataset(Dataset):
    def __init__(self, 
                 buffer, 
                 data_format, 
                 process_fn=None, 
                 batch_size=False, 
                 print_data_format=True, 
                 **kwargs):
        super().__init__(
            buffer, 
            data_format, 
            process_fn=process_fn, 
            batch_size=batch_size, 
            print_data_format=print_data_format, 
            **kwargs)
        self._sleep_time = 0.025
    
    def name(self):
        return ray.get(self._buffer.name.remote())

    def good_to_learn(self):
        return ray.get(self._buffer.good_to_learn.remote())

    def _sample(self):
        while True:
            data = ray.get(self._buffer.sample.remote())
            if data is None:
                time.sleep(self._sleep_time)
            else:
                yield data

    def update_priorities(self, priorities, indices):
        self._buffer.update_priorities.remote(priorities, indices)


def get_dataformat(replay):
    import time
    i = 0
    while not ray.get(replay.good_to_learn.remote()):
        time.sleep(1)
        i += 1
        if i % 60 == 0:
            size = ray.get(replay.size.remote())
            if size == 0:
                import sys
                print('Replay does not collect any data in 60s. Specify data_format for dataset construction explicitly')
                sys.exit()
            print(f'Dataset Construction: replay size = {size}')
    data = ray.get(replay.sample.remote())
    data_format = {k: (v.shape, v.dtype) for k, v in data.items()}
    do_logging('Data format:', logger=logger)
    do_logging(data_format, prefix='\t', logger=logger)

    return data_format
