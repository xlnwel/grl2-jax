import os, glob
import types
import pathlib
import importlib
import psutil
import shutil

from tools.log import do_logging
from core.names import PATH_SPLIT


def mkdir(d):
  if not os.path.isdir(d):
    pathlib.Path(d).mkdir(parents=True)


def rm(path):
  if os.path.isfile(path):
    # Delete the file using os.remove
    os.remove(path)
  elif os.path.isdir(path):
    # Delete the directory and its contents using shutil.rmtree
    shutil.rmtree(path)


def is_file_open(file_path):
  for proc in psutil.process_iter(['pid', 'name', 'open_files']):
    try:
      for file in proc.open_files():
        if file.path == file_path:
          return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
      pass
  return False


def source_file(file_path):
  """
  Dynamically "sources" a provided file
  """
  basename = os.path.basename(file_path)
  filename = basename.replace(".py", "")
  # Load the module
  loader = importlib.machinery.SourceFileLoader(filename, file_path)
  mod = types.ModuleType(loader.name)
  loader.exec_module(mod)


def load_files(path='.', recursively_load=True):
  """
  This function takes a path to a local directory
  and imports all the available files in there.
  """
  # for _file_path in glob.glob(os.path.join(local_dir, "*.py")):
  #   source_file(_file_path)
  for f in glob.glob(f'{path}{PATH_SPLIT}*'):
    if os.path.isdir(f) and recursively_load:
      load_files(f)
    elif f.endswith('.py') and not f.endswith('__init__.py') \
        and not f.endswith('utils.py') and not f.endswith('test.py') \
          and not f.endswith('typing.py'):
      source_file(f)


def retrieve_pyfiles(path='.'):
  return [f for f in glob.glob(f'{path}/*') 
    if f.endswith('.py') and not f.endswith('__init__.py')]


def check_make_dir(path):
  _, ext = os.path.splitext(path)
  if ext: # if path is a file path, extract its directory path
    path, _ = os.path.split(path)

  if not os.path.isdir(path):
    os.mkdir(path)


def search_for_all_files(directory, filename, suffix=True):
  if not os.path.exists(directory):
    return []
  directory = directory
  all_target_files = []
  for root, _, files in os.walk(directory):
    if 'src' in root:
      continue
    for f in files:
      if suffix:
        if f.endswith(filename):
          all_target_files.append(os.path.join(root, f))
      else:
        if f.startswith(filename):
          all_target_files.append(os.path.join(root, f))
  
  return all_target_files


def search_for_file(directory, filename, check_duplicates=True):
  if not os.path.exists(directory):
    return None
  do_logging(f'{directory}/{filename}')
  directory = directory
  target_file = None
  for root, _, files in os.walk(directory):
    if 'src' in root:
      continue
    for f in files:
      if f.endswith(filename) and target_file is None:
        target_file = os.path.join(root, f)
        if not check_duplicates:
          break
      elif f.endswith(filename) and target_file is not None:
        do_logging(f'Get multiple "{filename}": "{target_file}" and "{os.path.join(root, f)}"', backtrack=4)
        exit()
    if not check_duplicates and target_file is not None:
      break

  return target_file


def search_for_all_files(directory, filename, is_suffix=True, remove_dir=False):
  if not os.path.exists(directory):
    return []
  directory = directory
  all_target_files = []
  for root, _, files in os.walk(directory):
    if 'src' in root:
      continue
    for f in files:
      if is_suffix:
        if f.endswith(filename):
          all_target_files.append(os.path.join(root, f))
      else:
        if f.startswith(filename):
          all_target_files.append(os.path.join(root, f))
  if remove_dir:
    all_target_files = [f.replace(f'{directory}{PATH_SPLIT}', '') for f in all_target_files]
  return all_target_files


def search_for_dirs(directory, dirname, is_suffix=True, matches=None):
  if not os.path.exists(directory):
    return []
  directory = directory
  n_slashes = dirname.count('/')
  all_target_files = set()
  for root, _, _ in os.walk(directory):
    if 'src' in root:
      continue
    if matches is not None and all([m not in root for m in matches]):
      continue
    endnames = root.rsplit(PATH_SPLIT, n_slashes+1)[1:]
    endname = PATH_SPLIT.join(endnames)
    if is_suffix:
      if endname.endswith(dirname):
        all_target_files.add(root)
    else:
      if endname.startswith(dirname):
        all_target_files.add(root)

  return list(all_target_files)


def yield_dirs(directory, dirnames, is_suffix=True, root_matches=None):
  if not os.path.exists(directory):
    return []
  directory = directory
  
  for root, _, _ in os.walk(directory):
    if 'src' in root:
      continue
    if root_matches is not None and all([m not in root for m in root_matches]):
      continue

    for dirname in dirnames:
      n_slashes = dirname.count(PATH_SPLIT)

      endnames = root.rsplit(PATH_SPLIT, n_slashes+1)[1:]
      endname = PATH_SPLIT.join(endnames)

      if is_suffix:
        if endname.endswith(dirname):
          yield root
      else:
        if endname.startswith(dirname):
          yield root


def write_file(path, content, mode='a'):
  if not path.endswith('.txt'):
    path = path + '.txt'
  d, f = path.rsplit('/', 1)
  if not os.path.isdir(d):
    pathlib.Path(d).mkdir(parents=True)
  with open(path, mode) as f:
    f.write(content + '\n')
