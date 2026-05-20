import sys
from importlib import import_module

_target = import_module("agfc.research.journalmix_mineru_baseline")
sys.modules[__name__] = _target
