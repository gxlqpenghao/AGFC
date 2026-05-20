import sys
from importlib import import_module

_target = import_module("agfc.research.journalmix_agfc_fresh_benchmark")
sys.modules[__name__] = _target
