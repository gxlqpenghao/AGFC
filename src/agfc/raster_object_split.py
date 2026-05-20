import sys
from importlib import import_module

_target = import_module("agfc.core.raster_object_split")
sys.modules[__name__] = _target
