"""Kernel."""
from abc import ABC, abstractmethod


class Kernel(ABC):
    """Represents a transformation of a GraphItem."""

    __key = object()

    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply the Kernel transformation."""
        obj = cls(cls.__key, *args, **kwargs)
        return obj._apply(*args, **kwargs)

    def __init__(self, key, *args, **kwargs):
        assert(key == self.__key), "This object should only be called using the `apply` method"

    @abstractmethod
    def _apply(self, *args, **kwargs):
        pass
