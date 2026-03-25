"""CIL method registry — auto-imports all method modules so @register_method fires."""
from pathlib import Path
import importlib

from .base import CILMethod, register_method, get_method_registry

# Auto-import every .py in this package (except base.py and _-prefixed files)
_pkg_dir = Path(__file__).parent
for _f in sorted(_pkg_dir.glob("*.py")):
    if _f.name.startswith("_") or _f.name == "base.py":
        continue
    importlib.import_module(f".{_f.stem}", package=__name__)

__all__ = ["CILMethod", "register_method", "get_method_registry"]
