import sys
from importlib import import_module

_target = import_module("agfc.adapters.integrations.mineru")
sys.modules[__name__] = _target
