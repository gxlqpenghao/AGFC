import sys
from importlib import import_module

_target = import_module("agfc.core.visual_community")
sys.modules[__name__] = _target
