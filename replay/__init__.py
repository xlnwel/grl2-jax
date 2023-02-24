import os

from replay.registry import *
from tools.file import load_files


def load_buffer():
    load_files(os.path.dirname(os.path.realpath(__file__)))
    
load_buffer()
