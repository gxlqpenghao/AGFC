import sys
from importlib import import_module

_target = import_module("agfc.core.figure_instances")
sys.modules[__name__] = _target
