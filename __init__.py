"""ComfyUI-NanoBanana2 — Google Gemini integration.

Aggregates the v2.0 core nodes (nanobanana_node.py) with v2.1's new
feature nodes (extra_nodes.py) into the standard ComfyUI mapping dicts.

Loader compatibility: ComfyUI typically loads custom_node packages via
SourceFileLoader, which gives this file a __package__ name and lets the
relative imports below resolve. When run outside a package (e.g. pytest
adding the parent dir to sys.path), we fall back to absolute imports.
"""

try:
    from .nanobanana_node import (
        NODE_CLASS_MAPPINGS as _CORE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _CORE_NAMES,
    )
    from .extra_nodes import (
        NEW_NODE_CLASS_MAPPINGS as _EXTRA_CLASSES,
        NEW_NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_NAMES,
    )
    from .network_route import (
        NODE_CLASS_MAPPINGS as _NET_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _NET_NAMES,
    )
except ImportError:
    # Flat-import fallback (tests / direct execution).
    from nanobanana_node import (
        NODE_CLASS_MAPPINGS as _CORE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _CORE_NAMES,
    )
    from extra_nodes import (
        NEW_NODE_CLASS_MAPPINGS as _EXTRA_CLASSES,
        NEW_NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_NAMES,
    )
    from network_route import (
        NODE_CLASS_MAPPINGS as _NET_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _NET_NAMES,
    )

NODE_CLASS_MAPPINGS = {**_CORE_CLASSES, **_EXTRA_CLASSES, **_NET_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**_CORE_NAMES, **_EXTRA_NAMES, **_NET_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
