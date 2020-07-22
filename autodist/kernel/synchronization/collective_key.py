# Copyright 2020 Petuum, Inc. All Rights Reserved.
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

"""Utility functions for `AllReduceSynchronizer`."""
import threading
import hashlib

from autodist.const import MAX_INT32


_collective_keys = None
_collective_keys_lock = threading.Lock()


def get_collective_keys():
    """Return a singleton instance of CollectiveKey."""
    global _collective_keys
    if _collective_keys:
        return _collective_keys
    _collective_keys_lock.acquire()

    try:
        if _collective_keys:
            return _collective_keys
        collective_keys = CollectiveKey()
        _collective_keys = collective_keys
        return _collective_keys
    finally:
        _collective_keys_lock.release()


class CollectiveKey:
    """A hash that generates group key and instance key for AllReduce."""

    def __init__(self, group_key_start=1):
        """Init the collective key."""
        self._group_key = group_key_start
        self._group_key_dict = {}
        self._instance_key_dict = {}

    def get_group_key(self, canonical_devices):
        """Generate or retrieve the group key based on a list of strings of the participating devices."""
        for d in canonical_devices:
            if not isinstance(d, str):
                raise ValueError('Need canonicalized devices')
        key_id = ','.join(canonical_devices)
        if key_id not in self._group_key_dict:
            new_key = self._group_key
            self._group_key += 1
            self._group_key_dict[key_id] = new_key
        return self._group_key_dict[key_id]

    def get_instance_key(self, var_op_name):
        """Generate or retrieve the instance key based on the *original* variable op name."""
        key_id = var_op_name
        if key_id not in self._instance_key_dict:
            new_key = int(hashlib.md5(var_op_name.encode()).hexdigest(), 16) % MAX_INT32
            self._instance_key_dict[key_id] = new_key
        return self._instance_key_dict[key_id]
