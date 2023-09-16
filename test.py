import numpy as np

from core.ckpt.pickle import save, restore
from tools.display import print_dict_info
from tools.utils import batch_dicts


if __name__ == '__main__':
  data1 = restore(filedir='data', filename='uniform')
  data2 = restore(filedir='data/happo_mb-magw-escalation', filename='uniform')

  data = batch_dicts([data1, data2], np.concatenate)
  print_dict_info(data)
  save(filedir='data', filename='uniform')
