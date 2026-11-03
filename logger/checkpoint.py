"""Explicit deserialization policy shared by CPU-staged checkpoint loaders."""
import torch


def load_checkpoint(path, checkpoint_policy='trusted'):
    """Load on CPU without silently retrying unsafe deserialization.

    ``trusted`` retains compatibility with legacy local DDSP/encoder checkpoints
    that contain Python objects or optimizer state. Only use it for trusted files.
    Callers handling tensor-only checkpoints can explicitly choose ``weights_only``;
    unsupported objects then fail rather than triggering an unsafe fallback.
    """
    if checkpoint_policy not in ('trusted', 'weights_only'):
        raise ValueError('checkpoint_policy must be trusted or weights_only')
    return torch.load(path, map_location='cpu',
                      weights_only=checkpoint_policy == 'weights_only')
