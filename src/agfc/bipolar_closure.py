import sys
from importlib import import_module

_target = import_module("agfc.core.bipolar_closure")
sys.modules[__name__] = _target
