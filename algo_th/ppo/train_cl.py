import functools

from algo.ma_common.run import CurriculumRunner
from algo.ma_common.train import *

main = functools.partial(main, Runner=CurriculumRunner)
