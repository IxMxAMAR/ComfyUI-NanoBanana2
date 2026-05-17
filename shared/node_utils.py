"""Shared node utilities and mixins for ComfyUI nodes."""


class AlwaysExecuteMixin:
    """Mixin that forces re-execution on every queue (no stale cache)."""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class OptionalRerunMixin:
    """Re-execute by default but cache on identical inputs when force_rerun=False.

    Nodes that wrap expensive API calls (Veo, Imagen, Lyria) should mix
    this in and expose an optional `force_rerun` BOOL input. When False
    AND all other inputs are byte-identical to the last run, ComfyUI
    reuses the cached output and skips the API call.
    """

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Default to "always changed" if force_rerun toggle is missing or True.
        if kwargs.get("force_rerun", True):
            return float("nan")
        # Otherwise return a hash of remaining inputs so identical runs cache.
        import hashlib
        import json
        try:
            blob = json.dumps(
                {k: _hashable(v) for k, v in kwargs.items() if k != "force_rerun"},
                sort_keys=True, default=str,
            )
        except Exception:
            return float("nan")
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _hashable(v):
    """Best-effort hashable form for tensors, dicts, lists."""
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return ("tensor", tuple(v.shape), float(v.sum().item()))
    except Exception:
        pass
    if isinstance(v, dict):
        return {k: _hashable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_hashable(x) for x in v]
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return repr(v)[:200]
