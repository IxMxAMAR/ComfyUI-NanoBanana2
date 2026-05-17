"""Pytest fixtures + path setup for NanoBanana2 tests.

Tests import the package by name. We register the project root in
sys.path so they can `import gemini_client`, `import shared.conversions`,
etc., without going through ComfyUI's loader.
"""

import os
import sys

# Project root is one level above this conftest.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
