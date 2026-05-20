import sys
from importlib import import_module

_target = import_module("agfc.runtime.service")
sys.modules[__name__] = _target
