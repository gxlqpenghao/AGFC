import sys
from importlib import import_module

_target = import_module("agfc.adapters.integrations.marker.api_adapter")
sys.modules[__name__] = _target
