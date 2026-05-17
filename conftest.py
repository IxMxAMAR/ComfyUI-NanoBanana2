"""Project-root conftest for pytest.

Without this file at the rootdir, pytest sees `__init__.py` in the
project root and tries to import the package via dotted name, which
breaks because the package uses relative imports designed for ComfyUI's
SourceFileLoader.

Having `conftest.py` at the rootdir AND setting `tests/__init__.py`'s
absence means pytest treats `tests/` as the only collection root.
"""

import os
import sys

# Add project root to sys.path so test modules can `import gemini_client`,
# `from shared.conversions import ...` etc.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

collect_ignore_glob = [
    "__init__.py",
    "nanobanana_node.py",
    "extra_nodes.py",
    "gemini_client.py",
    "shared/*.py",
]
