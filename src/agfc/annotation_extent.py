import sys
from importlib import import_module

_target = import_module("agfc.core.annotation_extent")
sys.modules[__name__] = _target
