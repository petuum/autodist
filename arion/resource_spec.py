"""Resource Specification."""

from enum import Enum

import re
import yaml

from autodist.network import SSHConfigMap, is_loopback_address
from autodist.utils import logging


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
        self.__chief_address = None
        self.__ssh_config_map = dict()
        self.__ssh_group = dict()

        # set self.__devices
        self._from_resource_info(resource_file)

    @property
    def chief(self):
        """Return chief address."""
        return self.__chief_address

    @property
    def devices(self):
        """Return all devices."""
        return self.__devices.items()

    @property
    def nodes(self):
        """Return all node addresses."""
        if not self.__nodes:
            self.__nodes = {v.host_address for v in self.__devices.values()}  # set
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

    @property
    def ssh_config_map(self):
        """Configurations for SSH."""
        return self.__ssh_config_map

    @property
    def ssh_group(self):
        """SSH Group for each node."""
        return self.__ssh_group

    def _add_device(self, device_spec):
        if device_spec.name_string() not in self.__devices:
            self.__devices[device_spec.name_string()] = device_spec

    def _from_resource_info(self, resource_file=None):
        if resource_file is None:
            # TODO(Hao): To deal with single-node GPUs
            return

        resource_info = yaml.safe_load(open(resource_file, 'r'))
        num_nodes = len(resource_info.get('nodes', {}))

        for node in resource_info.pop('nodes', {}):
            host_address = node['address']

            if is_loopback_address(host_address) and num_nodes > 1:
                raise ValueError("Can't (currently) use a loopback address when there are multiple nodes.")

            if node.get('chief') or num_nodes == 1:
                # 2 cases for marking this node as chief:
                # 1) The node was marked as chief
                # 2) If there is only one node, it is chief by default
                logging.info("%s is CHIEF" % host_address)
                self.__chief_address = host_address

            host_cpu = DeviceSpec(host_address)
            self._add_device(host_cpu)

            # handle any other CPUs
            for cpu_index in node.get('cpus', [])[1:]:
                cpu = DeviceSpec(host_address, host_cpu, DeviceType.CPU, cpu_index)
                self._add_device(cpu)
            # handle GPUs
            for gpu_index in node.get('gpus', []):
                gpu = DeviceSpec(host_address, host_cpu, DeviceType.GPU, gpu_index)
                self._add_device(gpu)

            self.__ssh_group[host_address] = node.get('ssh_config')
            if self.__ssh_group[host_address] is None and self.__chief_address != host_address:
                raise ValueError("Need to define SSH groups for all non-chief nodes.")

        # Make sure there is a chief set
        if not self.__chief_address:
            raise ValueError("Must specify one of the nodes to be chief.")

        # all other configs except nodes are (optional) ssh config
        self.__ssh_config_map = SSHConfigMap(resource_info.pop('ssh', {}), self.__ssh_group)

        # checks
        if self.__chief_address is None:
            raise ValueError('Must provide "chief: true" in one of the nodes in resource spec.')


class DeviceSpec:
    """Device specification."""

    def __init__(
            self,
            host_address,
            host_device=None,
            device_type=DeviceType.CPU,
            device_index=None
    ):
        self.host_address = host_address
        self.device_type = device_type
        if self.device_type is DeviceType.GPU:
            self.device_index = device_index
            if host_device is not None:
                if host_device.device_type is not DeviceType.CPU:
                    raise ValueError('Host device must be a CPU')
            else:
                self.host_device = DeviceSpec(host_address)
        else:
            self.device_index = 0
            self.host_device = self

    def name_string(self):
        """Name string."""
        if self.device_type is DeviceType.CPU:
            return self.host_address + ':' + DeviceType.CPU.name + ':0'
        else:
            return self.host_address + ':' + self.device_type.name + ':' + str(self.device_index)

    def connectivity_with(self, device_spec):
        """
        Connectivity.

        TODO (hao.zhang): why func rather than an precalculated adjacency list.
        """
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

    @classmethod
    def from_string(cls, name_string):
        """
        Construct an AutoDist DeviceSpec based on its name string.

        Args:
            name_string: AutoDist DeviceSpec name string

        Returns:
            DeviceSpec: an instance
        """
        address, device_type, device_index = re.match(r"(\S+):([a-zA-Z]+):(\d+)", name_string).groups()
        obj = cls(
            address,
            device_type=DeviceType[device_type],
            device_index=device_index
        )
        return obj

    def __hash__(self):
        return hash(self.name_string())

    def __eq__(self, other):
        return self.name_string() == other.name_string()

    def __repr__(self):
        return "<DeviceSpec: {}>".format(self.name_string())

    def __str__(self):
        return self.name_string()
