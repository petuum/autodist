"""Resource Specification."""

import os
from enum import Enum


class Connectivity(Enum):
    """Connectivity."""

    ETHERNET = 0
    CPU_TO_GPU = 1
    GPU_TO_GPU_LOCAL = 2
    GPU_TO_GPU_REMOTE = 3
    SAME = 4


class DeviceType(Enum):
    """Device Type."""

    CPU = 0
    GPU = 1


# TODO(Hao): make it a real GRAPH
class ResourceSpec:
    """Resource Spec."""

    def __init__(self, resource_file=None):
        """
        Construct a device graph containing the connectivity between devices.

        Each device is represented as an autodist.device_spec.
        If resource_file is None, use the local machines and all visible GPUs.

        Args:
            resource_file (string, optional): path to the file containing the resource info. Defaults to None.
        """
        # protected properties
        self.__devices = dict()
        self.__nodes = None
        self.__cpu_devices = None
        self.__num_cpus = None
        self.__gpu_devices = None
        self.__num_gpus = None

        # set self.__devices
        self._from_resource_info(resource_file)

    @property
    def devices(self):
        """Return all devices."""
        return self.__devices.items()

    @property
    def nodes(self):
        """Return all node addresses."""
        if not self.__nodes:
            self.__nodes = {k.split(':')[0] for k in self.__devices}  # set
        return self.__nodes

    @property
    def cpu_devices(self):
        """String-to-device_spec mapping of all cpu devices."""
        if not self.__cpu_devices:
            self.__cpu_devices = {k: v for k, v in self.__devices.items() if v.device_type is DeviceType.CPU}
        return self.__cpu_devices.items()

    @property
    def num_cpus(self):
        """Number of all cpu devices."""
        if not self.__num_cpus:
            self.__num_cpus = len(self.cpu_devices)
        return self.__num_cpus

    @property
    def gpu_devices(self):
        """String-to-device_spec mapping of all gpu devices."""
        if not self.__gpu_devices:
            self.__gpu_devices = {k: v for k, v in self.__devices.items() if v.device_type is DeviceType.GPU}
        return self.__gpu_devices.items()

    @property
    def num_gpus(self):
        """Number of all gpu devices."""
        if not self.__num_gpus:
            self.__num_gpus = len(self.gpu_devices)
        return self.__num_gpus

    def _add_device(self, device_spec):
        if device_spec.name_string() not in self.__devices:
            self.__devices[device_spec.name_string()] = device_spec

    def _from_resource_info(self, resource_file=None):
        if resource_file is None:
            # TODO(Hao): To deal with single-node GPUs
            return
        if not os.path.exists(resource_file):
            raise FileNotFoundError
        lines = [line.rstrip('\n') for line in open(resource_file)]
        for line in lines:
            devices = line.split(':')
            host_address = devices[0]
            host_cpu = DeviceSpec(self, host_address)
            self._add_device(host_cpu)
            # handle GPUs
            if len(devices) > 1:
                gpu_indices = devices[1].split(',')
                for index in gpu_indices:
                    gpu = DeviceSpec(self, host_address,
                                     host_cpu,
                                     DeviceType.GPU,
                                     index)
                    self._add_device(gpu)

    def is_single_node(self):
        """Return True if there is only a single node"""
        return self.num_cpus == 1

    def is_single_node_with_gpu(self):
        """Return True if there is a single node with GPUs"""
        return self.is_single_node() and self.num_gpus > 0


class DeviceSpec:
    """Device specification."""

    def __init__(self,
                 resource_spec, host_address,
                 host_device=None,
                 device_type=DeviceType.CPU,
                 device_index=None):
        # reference to a device graph
        self._resource_spec = resource_spec
        self.host_address = host_address
        self.device_type = device_type
        if self.device_type is DeviceType.GPU:
            self.device_index = device_index
            if host_device is None or \
                    host_device.device_type is not DeviceType.CPU:
                raise ValueError('Host device must be a CPU')
            self.host_device = host_device
        else:
            self.device_index = 0
            self.host_device = self

    def name_string(self):
        """Name string."""
        if self.device_type is DeviceType.CPU:
            return self.host_address
        else:
            return self.host_address + ':' + str(self.device_index)

    def connectivity_with(self, device_spec):
        """Connectivity."""
        if self.host_address is not device_spec.host_address:
            return Connectivity.ETHERNET
        # on the same pyhsical node
        elif self.device_type is not device_spec.device_type:
            return Connectivity.CPU_TO_GPU
        # have the same type of devices
        elif self.device_type is DeviceType.CPU:
            return Connectivity.SAME
        # both are GPUs
        elif self.device_index is device_spec.device_index:
            return Connectivity.SAME
        else:
            return Connectivity.GPU_TO_GPU_LOCAL
