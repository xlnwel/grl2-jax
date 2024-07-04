import os, sys
# os.environ["XLA_FLAGS"] = '--xla_dump_to=/tmp/foo'
if sys.platform == "linux" or sys.platform == "linux2":
  pass
elif sys.platform == "darwin":
  os.environ["XLA_FLAGS"] = '--xla_gpu_force_compilation_parallelism=1'
# running in a single cpu
# os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
               # "intra_op_parallelism_threads=1")

from datetime import datetime

# try:
#   from tensorflow.python.compiler.mlcompute import mlcompute
#   mlcompute.set_mlc_device(device_name='gpu')
#   print("----------M1----------")
# except:
#   print("----------Not M1-----------")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.log import setup_logging, do_logging
from core import names
from tools import pkg
from run.args import parse_train_args


if __name__ == '__main__':
  cmd_args = parse_train_args()
  if cmd_args.dllib == names.DL_LIB.TORCH:
    names.dllib = names.DL_LIB.TORCH
  do_logging(cmd_args, level='info')
  from run.ops import *

  setup_logging(cmd_args.verbose)
  if not (cmd_args.grid_search and cmd_args.multiprocess) and cmd_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = f",".join([f"{gpu}" for gpu in cmd_args.gpu])
  processes = []
  if cmd_args.directory != '':
    configs = [search_for_config(d) for d in cmd_args.directory]
    for config in configs:
      config.cpu_only = cmd_args.cpu
    main = pkg.import_main(cmd_args.train_entry, config=configs[0])
    main(configs)
  else:
    run_with_configs(cmd_args)
