import sys
from importlib import import_module

_target = import_module("agfc.core.pipeline_models")
sys.modules[__name__] = _target
