"""
This file defines general CNN architectures used in RL
"""
import os, glob
import types
import importlib
from nn.block.cnns.utils import cnn_mapping


def cnn(cnn_name, **kwargs):
    if cnn_name is None:
        return None
    cnn_name = cnn_name.lower()
    if cnn_name in cnn_mapping:
        return cnn_mapping[cnn_name](**kwargs)
    else:
        raise ValueError(f'Unknown CNN structure: {cnn_name}. Available cnn: {list(cnn_mapping)}')


def _source_file(_file_path):
    """
    Dynamically "sources" a provided file
    """
    basename = os.path.basename(_file_path)
    filename = basename.replace(".py", "")
    # Load the module
    loader = importlib.machinery.SourceFileLoader(filename, _file_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)


def load_cnn(local_dir="."):
    """
    This function takes a path to a local directory
    and looks for a `models` folder, and imports
    all the available files in there.
    """
    for _file_path in glob.glob(os.path.join(
        local_dir, "cnns", "*.py")):
        """
        Sources a file expected to implement a
        custom model.

        The respective files are expected to do a
        `registry.register_env` call to ensure that
        the implemented envs are available in the
        ray registry.
        """
        _source_file(_file_path)


load_cnn(os.path.dirname(os.path.realpath(__file__)))
