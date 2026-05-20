import sys
from importlib import import_module

_target = import_module("agfc.runtime.runner")
sys.modules[__name__] = _target
