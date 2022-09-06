import os, glob
import types
import importlib


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
    #     source_file(_file_path)
    for f in glob.glob(f'{path}/*'):
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
                print(f'Get multiple "{filename}": "{target_file}" and "{os.path.join(root, f)}"')
                exit()
        if not check_duplicates and target_file is not None:
            break

    return target_file

def search_for_all_files(directory, filename, is_suffix=True):
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
    
    return all_target_files

def search_for_dirs(directory, dirname, is_suffix=True, name=None):
    if not os.path.exists(directory):
        return []
    directory = directory
    n_slashes = dirname.count('/')
    all_target_files = set()
    for root, _, _ in os.walk(directory):
        if 'src' in root:
            continue
        if name is not None and name not in root:
            continue
        endnames = root.rsplit('/', n_slashes+1)[1:]
        endname = '/'.join(endnames)
        if is_suffix:
            if endname.endswith(dirname):
                all_target_files.add(root)
        else:
            if endname.startswith(dirname):
                all_target_files.add(root)

    return list(all_target_files)
