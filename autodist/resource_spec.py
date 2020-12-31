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

"""Resource Specification."""
import os
import re
from enum import Enum
from typing import NamedTuple, Optional, Dict
import yaml
import paramiko

from autodist.utils import logging
from autodist.utils.network import is_loopback_address, is_local_address


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
    """
    Resource Spec.

    Contains node and SSH information found by parsing a `resource_spec.yml`.

    # TODO: Make it a real Graph (a clique), with edge weights being network bandwidth.
        This would allow for even more intelligent strategy generation.
    """

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
        self.__network_bandwidth = dict()

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
    def node_gpu_devices(self):
        """Node_address-to-device_string mapping of all gpu devices."""
        _gpu_devices = dict()
        for device in self.gpu_devices:
            _gpu_devices.setdefault(device[0].split(':')[0], []).append(device[0])
        return _gpu_devices

    @property
    def node_cpu_devices(self):
        """Node_address-to-device_string mapping of all cpu devices."""        
        _cpu_devices = dict()
        for device in self.cpu_devices:
            _cpu_devices.setdefault(device[0].split(':')[0], []).append(device[0])
        return _cpu_devices

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

    @property
    def network_bandwidth(self):
        """Network bandwidth of each node."""
        return self.__network_bandwidth

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
            self._parse_node(node, num_nodes)

        # Make sure there is a chief set
        if not self.__chief_address:
            raise ValueError("Must specify one of the nodes to be chief.")

        # all other configs except nodes are (optional) ssh config
        gpu_devices = self.node_gpu_devices
        if is_local_address(self.__chief_address):
            self.__ssh_config_map = SSHConfigMap(
                resource_info.pop('ssh', {}), self.__ssh_group, gpu_devices)

        # checks
        if self.__chief_address is None:
            raise ValueError('Must provide "chief: true" in one of the nodes in resource spec.')

    def _parse_node(self, node, num_nodes):
        host_address = node['address']
        if is_loopback_address(host_address) and num_nodes > 1:
            raise ValueError("Can't (currently) use a loopback address when there are multiple nodes.")
        if node.get('chief') or num_nodes == 1:
            # 2 cases for marking this node as chief:
            # 1) The node was marked as chief
            # 2) If there is only one node, it is chief by default
            logging.info("Chief: %s" % host_address)
            self.__chief_address = host_address
        host_cpu = DeviceSpec(host_address, device_index=0)
        self._add_device(host_cpu)
        # handle any other CPUs when GPU is unavailable
        if len(node.get('gpus', [])) == 0:
            for cpu_index in set(sorted(node.get('cpus', []))) - {0}:
                cpu = DeviceSpec(host_address, host_cpu, DeviceType.CPU, cpu_index)
                self._add_device(cpu)
        # handle GPUs
        for gpu_index in set(sorted(node.get('gpus', []))):
            gpu = DeviceSpec(host_address, host_cpu, DeviceType.GPU, gpu_index)
            self._add_device(gpu)
        self.__ssh_group[host_address] = node.get('ssh_config')
        if self.__ssh_group[host_address] is None and self.__chief_address != host_address:
            raise ValueError("Need to define SSH groups for all non-chief nodes.")
        # handle network bandwidth (optional)
        if node.get('network_bandwidth'):
            self.__network_bandwidth[host_address] = node.get('network_bandwidth')
        else:
            logging.debug('The bandwidth for {} is undefined and set as default (1 GBE). '
                          'Caution: AutoStrategy might be inaccurate.'.format(host_address))
            self.__network_bandwidth[host_address] = 1


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
            self.device_index = device_index
            self.host_device = self

    def name_string(self):
        """Name string."""
        if self.device_type is DeviceType.CPU:
            return self.host_address + ':' + DeviceType.CPU.name + ':' + str(self.device_index)
        else:
            return self.host_address + ':' + self.device_type.name + ':' + str(self.device_index)

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


class SSHConfig(NamedTuple):
    """Contains any necessary SSH information (e.g. passwords, keyfiles, etc.)."""

    username: str
    port: int
    python_venv: str
    key_file: str
    pkey: Optional[paramiko.RSAKey]
    env: dict


class SSHConfigMap(dict):
    """Contains all necessary SSH configs, grouped by config name."""

    def __init__(self, info: Dict[str, Dict], node_groups: Dict[str, str], gpu_devices: Dict[str, str]):
        """
        Initialize the object with a dictionary of SSH information.

        Args:
            info (dict): any SSH information needed for remote control.
                This dict should map from identifier to dict of SSH info
                (username, port, keyfile, etc.).
            node_groups (dict): mapping from hostnames to SSH group names.
            gpu_devices: GPU devices in each node
        """
        super().__init__()

        # Construct SSH Group to SSH Config mapping
        conf_map = {}
        for key, ssh_info in info.items():
            # Parse out information from sub-dict
            conf_map[key] = SSHConfig(
                username=ssh_info.get('username', ''),
                port=ssh_info.get('port', 22),
                python_venv=ssh_info.get('python_venv', ''),
                key_file=ssh_info.get('key_file', ''),
                pkey=self._gen_rsa_pkey(ssh_info.get('key_file', None)),
                env=dict(
                    TF_CPP_MIN_LOG_LEVEL=0,
                    **ssh_info.get('shared_envs', {})
                )
            )

        # Use conf_map to construct Hostname to SSH Config mapping
        for hostname, group in node_groups.items():
            self[hostname] = conf_map.get(group)

    @staticmethod
    def _gen_rsa_pkey(key_file_path: str):
        if not key_file_path:
            return None
        return paramiko.RSAKey.from_private_key_file(os.path.expanduser(os.path.abspath(key_file_path)))
