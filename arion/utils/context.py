"""Reference to the current default AutoDist context."""

_DEFAULT_AUTODIST = None


def set_default_autodist(o):
    """Set the AutoDist object the scope of which you are in."""
    global _DEFAULT_AUTODIST
    _DEFAULT_AUTODIST = o


def get_default_autodist():
    """Get the AutoDist object the scope of which you are in."""
    global _DEFAULT_AUTODIST
    return _DEFAULT_AUTODIST