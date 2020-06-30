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

"""AutoDist device-setting kernels."""

from collections import defaultdict

from google.protobuf.pyext._message import RepeatedScalarContainer
from tensorflow.python.framework import device_spec

from autodist.resource_spec import DeviceSpec


class DeviceResolver:
    """Resolving AutoDist DeviceSpec to TensorFlow DeviceSpec given a cluster."""

    def __init__(self, cluster):
        """
        Initialize the DeviceResolver.

        Args:
            cluster (Cluster): The cluster with which to resolve devices.
        """
        self._cluster = cluster
        self._address_to_tasks = self._get_address_to_tasks(cluster)

    @staticmethod
    def _get_address_to_tasks(cluster):
        d = defaultdict(list)
        for job_name, tasks in cluster.cluster_spec.items():
            for task_index, full_address in enumerate(tasks):
                address = full_address.split(':')[0]
                d[address].append(dict(job=job_name, task=task_index))
        return d

    def resolve_to_device_spec(self, device):
        """Resolve an AutoDist DeviceSpec or its string to a TensorFlow DeviceSpec."""
        if isinstance(device, (list, set)):
            return type(device)(self.resolve_to_device_spec(d) for d in device)
        if isinstance(device, str):
            device = DeviceSpec.from_string(device)
        t = self._address_to_tasks.get(device.host_address)[0]  # temporarily only use the first one
        return device_spec.DeviceSpecV2(
            job=t['job'],  # temporarily not fully resolving it before memory issue solved
            task=t['task'],
            device_type=device.device_type.name,
            device_index=device.device_index
        )

    def resolve_to_device_str(self, device):
        """Resolve an AutoDist DeviceSpec or its string to a TensorFlow device string."""
        if isinstance(device, (list, set)):
            return type(device)(self.resolve_to_device_spec(d).to_string() for d in device)
        elif isinstance(device, RepeatedScalarContainer):
            return list(self.resolve_to_device_spec(d).to_string() for d in device)
        return self.resolve_to_device_spec(device).to_string()
