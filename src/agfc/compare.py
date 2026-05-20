import sys
from importlib import import_module

_target = import_module("agfc.research.compare")
sys.modules[__name__] = _target
