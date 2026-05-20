import sys
from importlib import import_module

_target = import_module("agfc.adapters.integrations.glmocr")
sys.modules[__name__] = _target
