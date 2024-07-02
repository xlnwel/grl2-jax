import os
import importlib

from core.names import PATH_SPLIT
from tools.log import do_logging


def pkg_str(root_dir, separator, base_name=None):
  if base_name is None:
    return root_dir
  root_dir = root_dir.replace(PATH_SPLIT, separator)
  return f'{root_dir}{separator}{base_name}'


def get_package_from_algo(algo, place=0, separator='.', dllib=None):
  if ':' in algo:
    algo = algo.split(':')[0]
  algo = algo.split('-', 1)[place]

  if dllib == 'jax':
    dllib = None
  pkg = get_package(dllib, 'algo', algo, separator)
  if pkg is None:
    pkg = get_package(None, 'distributed', algo, separator)

  return pkg


def get_package(root_dir, search_prefix, base_name=None, separator='.', backtrack=3):
  if root_dir is None:
    src = os.getcwd()
  else:
    src = os.path.join(os.getcwd(), root_dir)
  for d in os.listdir(src):
    if d.startswith(search_prefix):
      if root_dir is None:
        pkg = '.'.join([d, base_name])
      else:
        pkg = '.'.join([root_dir, d, base_name])
      try:
        if importlib.util.find_spec(pkg) is not None:
          if root_dir is None:
            pkg = f'{separator}'.join([d, base_name])
          else:
            pkg = f'{separator}'.join([root_dir, d, base_name])
          return pkg
      except Exception as e:
        do_logging(f'{e}', backtrack=backtrack, level='info')
        return None
  return None


def import_module(name, pkg=None, algo=None, *, config=None, place=0, dllib=None):
  """ import <name> module from <pkg>, 
  if <pkg> is not provided, import <name> module
  according to <algo> or "algorithm" in <config> 
  """
  if pkg is None:
    algo = algo or config['algorithm']
    assert isinstance(algo, str), algo
    pkg = get_package_from_algo(algo=algo, place=place, dllib=dllib)
  m = importlib.import_module(f'{pkg}.{name}')

  return m


def import_main(module, algo=None, *, config=None, dllib=None):
  algo = algo or config['algorithm']
  # if '-' in algo:
  #   module = '.'.join([algo.split('-')[0], module])
  assert isinstance(algo, str), algo
  if '-' in algo:
    m = importlib.import_module(f'distributed.{module}')
  else:
    place = 0 if module.startswith('train') else -1
    pkg = get_package_from_algo(algo, place=place, dllib=dllib)
    m = importlib.import_module(f'{pkg}.{module}')

  return m.main
