# Copyright 2020 Petuum. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
