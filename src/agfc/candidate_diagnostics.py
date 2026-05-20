import sys
from importlib import import_module

_target = import_module("agfc.core.candidate_diagnostics")
sys.modules[__name__] = _target
