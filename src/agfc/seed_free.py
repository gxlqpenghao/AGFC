import sys
from importlib import import_module

_target = import_module("agfc.core.seed_free")
sys.modules[__name__] = _target
