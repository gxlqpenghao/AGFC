import sys
from importlib import import_module

_target = import_module("agfc.core.object_export")
sys.modules[__name__] = _target
