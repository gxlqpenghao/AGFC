import sys
from importlib import import_module

_target = import_module("agfc.adapters.provider_env")
sys.modules[__name__] = _target
