import sys
from importlib import import_module

_target = import_module("agfc.research.mineru_api_vs_client_audit")
sys.modules[__name__] = _target
