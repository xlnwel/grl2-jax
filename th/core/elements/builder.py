import importlib

from core.elements.builder import ElementsBuilder as ElementsBuilderBase
from tools.log import do_logging
from tools import pkg


class ElementsBuilder(ElementsBuilderBase):
  """ Implementations"""
  def _import_element(self, name, algo=None):
    try:
      module = pkg.import_module(
        f'elements.{name}', algo=algo, algo_package='th')
    except Exception as e:
      level = 'info' if name == 'agent' else 'pwc'
      do_logging(
        f'Switch to default module({name}) due to error: {e}', 
        level=level, backtrack=4)
      do_logging(
        "You are safe to neglect it if it's an intended behavior. ", 
        level=level, backtrack=4)
      name = '.'.join(['th', 'core', 'elements', name])
      module = importlib.import_module(name)
    return module
