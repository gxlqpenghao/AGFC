import sys
from importlib import import_module

_target = import_module("agfc.core.text_anchors")
sys.modules[__name__] = _target
