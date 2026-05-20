import sys
from importlib import import_module

_target = import_module("agfc.research.doclaynet_error_review")
sys.modules[__name__] = _target
