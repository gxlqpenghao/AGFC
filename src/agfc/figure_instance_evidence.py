import sys
from importlib import import_module

_target = import_module("agfc.core.figure_instance_evidence")
sys.modules[__name__] = _target
