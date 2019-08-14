"""AutoDist device-setting kernels."""

from collections import defaultdict

from tensorflow.python.framework import device_spec

from autodist.resource_spec import DeviceSpec


class DeviceResolver:
    """Resolving AutoDist DeviceSpec to TensorFlow DeviceSpec given a cluster."""

    def __init__(self, cluster):

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
        return self.resolve_to_device_spec(device).to_string()
