import sys
from importlib import import_module

_target = import_module("agfc.core.nonraster_content_decomposition")
sys.modules[__name__] = _target
