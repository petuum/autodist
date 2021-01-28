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

"""
Cluster contains almost all the networking code for AutoDist.

The notable exception is ResourceSpec and a couple functions in `autodist.utils.network`.

Cluster will be used by other modules to handle:
1) Copying files
2) Writing files
3) Running code
on nodes in the cluster defined by the ResourceSpec. 

Prerequisites:
* TensorFlow is already installed in the env of all nodes.
* Only supports graph launching logic. Only one node (the Chief) runs the session client.
* AutoDist is already installed in the env of the worker node Chief, where the main script runs.
* The SSH private key to other nodes is accessible by AutoDist on Chief given a path.
"""
import atexit
import contextlib
import json
import os
import signal
import subprocess
import sys
import warnings
from abc import ABCMeta, abstractmethod

import paramiko

from autodist.const import DEFAULT_PORT_RANGE, DEFAULT_WORKING_DIR, ENV
from autodist.resource_spec import ResourceSpec
from autodist.utils import logging

warnings.filterwarnings(action='ignore', module=paramiko.__name__)


class Cluster(metaclass=ABCMeta):
    """Cluster manager for TensorFlow servers."""

    def __init__(self, resource_spec: ResourceSpec):
        self._resource_spec = resource_spec
        self.cluster_spec = self._get_default_cluster_spec(resource_spec)
        self._cpu_devices = self._get_node_cpu_devices(resource_spec)
        self._gpu_devices = self._get_node_gpu_devices(resource_spec)
        self._chief = resource_spec.chief
        self._full_addresses = [full_address for tasks in self.cluster_spec.values() for full_address in tasks]
        # noinspection PyTypeChecker
        self._address_to_port = dict(a.split(':') for a in self._full_addresses)
        self._task_to_address = {
            (job_name, task_index): a.split(':')[0]
            for job_name, tasks in self.cluster_spec.items()
            for task_index, a in enumerate(tasks)
        }
        self.subprocesses = []
        logging.info('ClusterSpec: {}'.format(self.cluster_spec))

    @staticmethod
    def _get_default_cluster_spec(resource_spec: ResourceSpec):
        """Create list of workers from the resource spec with semi-arbitrarily chosen ports."""
        return {
            'worker': [
                '{ip}:{port}'.format(
                    ip=n,
                    port=next(DEFAULT_PORT_RANGE)
                    # sorted is important.
                    # we need to guarantee the ip-port mapping to be the same in every worker.
                ) for n in sorted(resource_spec.nodes)
            ]
        }

    @staticmethod
    def _get_node_cpu_devices(resource_spec: ResourceSpec):
        _cpu_devices = dict()
        for device in resource_spec.cpu_devices:
            _cpu_devices.setdefault(device[0].split(':')[0], []).append(':'.join(device[0].split(':')[1:]))
        return _cpu_devices

    @staticmethod
    def _get_node_gpu_devices(resource_spec: ResourceSpec):
        _gpu_devices = dict()
        for device in resource_spec.gpu_devices:
            _gpu_devices.setdefault(device[0].split(':')[0], []).append(':'.join(device[0].split(':')[1:]))
        return _gpu_devices

    def get_resource_spec(self):
        return self._resource_spec

    def is_chief(self, address=None):
        """
        Check whether an address is chief or not.

        If the argument `address` is not provided,
        it will check whether the local address is chief.

        Args:
            address (str): node address e.g. ip

        Returns:
            bool: Whether address or self is chief
        """
        address = address or self.get_local_address()
        return address == self._chief

    def get_address_from_task(self, job_name, task_index):
        """
        Given a job name and task index, return the address.

        Args:
            job_name (str): job name
            task_index (int): task index

        Returns:
            str: The address for that task
        """
        return self._task_to_address[(job_name, task_index)]

    def get_local_address(self):
        """
        Get the local (ip) address.

        If labelled as AUTODIST_WORKER by the environment variable,
        the value of it is the address of the local node;
        otherwise the local node is chief.

        Returns:
            str: Worker ip or chief address by default.
        """
        return ENV.AUTODIST_WORKER.val or self._chief

    def get_local_worker_task_index(self):
        """
        Get the (first) TensorFlow task index of the "worker" for the local.

        Returns:
            int: Task index
        """
        return [i for i, a in enumerate(self._full_addresses) if self.get_local_address() in a][0]

    def get_local_session_target(self):
        """
        Get the session target of the local session.

        Returns:
            str: Local session target
        """
        port = self._address_to_port[self.get_local_address()]
        return 'grpc://localhost:' + port

    # pylint: disable=too-many-locals
    def start(self):
        """
        Start tf.servers on all nodes.

        Note that this only runs (and only should run) on the chief node.
        """
        # pylint: disable=import-outside-toplevel
        from autodist.utils import server_starter

        # atexit registration should be placed
        #   - before the beginning of the start
        #   (to ensure the clean termination if the start fails in its half way); and
        #   - at the same module as the start
        #   (to follow the python assumption that
        #   lower level modules will normally be imported
        #   before higher level modules and thus must be cleaned up later).
        atexit.register(self.terminate)
        envs = {ENV.AUTODIST_MIN_LOG_LEVEL.name: 'ERROR'}
        envs = ['{}={}'.format(k, v) for k, v in envs.items()]
        module_name = server_starter.__name__
        module_file = server_starter.__file__

        for job_name, tasks in self.cluster_spec.items():
            for task_index, full_address in enumerate(tasks):
                address = full_address.split(':')[0]
                args = ['--job_name=%s' % job_name, '--task_index=%d' % task_index,
                        '--cpu_device_num=%d' % len(self._cpu_devices[address])]
                if address in self._gpu_devices:
                    envs_cuda = []
                else:
                    envs_cuda = ['CUDA_VISIBLE_DEVICES=""']
                if self.is_chief(address):
                    json.dump(self.cluster_spec, open(os.path.join(DEFAULT_WORKING_DIR, 'cluster_spec.json'), 'w+'))
                    cmd = envs + envs_cuda + [sys.executable, '-m', module_name] + args
                    # pylint: disable=subprocess-popen-preexec-fn
                    proc = subprocess.Popen(' '.join(cmd), shell=True, preexec_fn=os.setsid)
                    self.subprocesses.append(proc)
                    # The above line immediately follows the Popen
                    # to ensure no gap for termination failure due to the empty proc list.
                    logging.info('$ local tf.server started at {}: job_name={} task_index={}'.format(
                        full_address, job_name, task_index
                    ))
                else:  # remote
                    self._remote_pre_start_tf_server(address, tf_server_starter_filepath=module_file)
                    file = os.path.join(DEFAULT_WORKING_DIR, os.path.basename(module_file))
                    bash = envs + envs_cuda + ['python', '-u', file] + args
                    logging.info(f"Launching tf.server on {address} with {bash}")
                    proc = self.remote_exec(bash, hostname=address)
                    # The above line immediately follows the Popen
                    # to ensure no gap for termination failure due to the empty proc list.
                    self.subprocesses.append(proc)

    def terminate(self):
        """Terminate."""
        logging.debug('Terminating cluster...')
        for p in self.subprocesses:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)

    def _remote_pre_start_tf_server(self, hostname, tf_server_starter_filepath, working_dir=DEFAULT_WORKING_DIR):
        """
        Prepare to start a TensorFlow server remotely.

        Args:
            hostname (str): host name or address
            tf_server_starter_filepath (str): local starter file path
            working_dir (str): remote working directory
        """
        logging.info("Copying necessary files to %s" % hostname)
        self.remote_copy(local_path=tf_server_starter_filepath, remote_path=working_dir, hostname=hostname)
        self.remote_file_write(
            remote_path=os.path.join(working_dir, 'cluster_spec.json'),
            data=json.dumps(self.cluster_spec),
            hostname=hostname,
        )

    @abstractmethod
    def remote_exec(self, args, hostname):
        """
        Execute a bash script remotely.

        Args:
            args (list): bash commands
            hostname (str): host name or address

        Returns:
            Process: process handle
        """

    @abstractmethod
    def remote_file_write(self, remote_path, data, hostname):
        """
        Write a remote file.

        Args:
            remote_path (str): remote file path
            data (str): data to be written
            hostname (str): host name or address
        """

    @abstractmethod
    def remote_copy(self, local_path, remote_path, hostname):
        """
        Copy a file to a remote directory.

        Args:
            local_path (str): local file path to be copied
            remote_path (str): remote directory path
            hostname (str): host name or address
        """


class SSHCluster(Cluster):
    """An AutoDist Cluster Based on SSH."""

    def __init__(self, resource_spec):
        self._ssh_conf = resource_spec.ssh_config_map
        super().__init__(resource_spec)

    @contextlib.contextmanager
    def _get_ssh_client(self, hostname):
        """
        Get a Paramiko SSH Client to the given node.

        Args:
            hostname (str): The node to SSH into.

        Returns:
            Yields a Paramiko SSHClient.
        """
        ssh_config = self._ssh_conf[hostname]
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.WarningPolicy)
        client.connect(hostname=hostname, port=ssh_config.port, username=ssh_config.username, pkey=ssh_config.pkey)
        yield client
        client.close()

    @contextlib.contextmanager
    def _get_sftp_client(self, hostname):
        """
        Get a Paramiko SFTP Client to the given node.

        Args:
            hostname (str): The node to SFTP to.

        Returns:
            Yields a Paramiko SFTPClient.
        """
        ssh_config = self._ssh_conf[hostname]
        t = paramiko.Transport((hostname, ssh_config.port))
        t.connect(username=ssh_config.username, pkey=ssh_config.pkey)
        sftp = paramiko.SFTPClient.from_transport(t)
        yield sftp
        sftp.close()
        t.close()

    def remote_exec(self, args, hostname):
        """
        Execute a bash script remotely.

        Args:
            args (list): bash commands
            hostname (str): host name or address

        Returns:
            Process: process handle
        """
        cmd_list = []
        ssh_config = self._ssh_conf[hostname]
        if ssh_config.python_venv:
            cmd_list.append('%s;' % ssh_config.python_venv)
        if ssh_config.env:
            cmd_list.extend(['%s=%s' % (k, v) for k, v in ssh_config.env.items()])
        full_cmd = ' '.join(cmd_list + args)

        remote_cmd = 'ssh -i {} -o StrictHostKeyChecking=no -tt -p {} {}@{} \'bash -c "{}"\' </dev/null' \
            .format(ssh_config.key_file, ssh_config.port, ssh_config.username, hostname, full_cmd)

        logging.debug('$ %s' % remote_cmd)

        if ENV.AUTODIST_DEBUG_REMOTE.val:
            return None

        # pylint: disable=subprocess-popen-preexec-fn
        proc = subprocess.Popen(remote_cmd, shell=True, preexec_fn=os.setsid)
        return proc

    def remote_file_write(self, remote_path, data, hostname):
        """
        Write a remote file.

        Args:
            remote_path (str): remote file path
            data (str): data to be written
            hostname (str): host name or address
        """
        with self._get_sftp_client(hostname) as sftp:
            with sftp.open(remote_path, 'w') as f:
                f.write(data)

    def remote_copy(self, local_path, remote_path, hostname):
        """
        Copy a file to a remote directory.

        Args:
            local_path (str): local file path to be copied
            remote_path (str): remote directory path
            hostname (str): host name or address
        """
        # Make sure directory exists
        with self._get_ssh_client(hostname) as client:
            _ = client.exec_command('mkdir -p %s' % remote_path)

        with self._get_sftp_client(hostname) as sftp:
            sftp.put(localpath=local_path, remotepath=os.path.join(remote_path, os.path.basename(local_path)))


# Ray related contents
import asyncio
import ray
import yaml
import socket
import time


@ray.remote
class NodeActor(object):
    def __init__(self):
        self._ready_event = asyncio.Event()
        self._proc_dict = {}

    def send(self, clear=False):
        self._ready_event.set()
        if clear:
            self._ready_event.clear()

    async def wait(self, should_wait=True):
        if should_wait:
            await self._ready_event.wait()

    def get_node_ip(self):
        return ray._private.services.get_node_ip_address()

    def execute_cmd(self, args):
        cmd_list = []
        full_cmd = ' '.join(cmd_list + args)
        logging.info(f"exec remote cmd {full_cmd}")
        # pylint: disable=subprocess-popen-preexec-fn
        proc = subprocess.Popen(full_cmd, shell=True, preexec_fn=os.setsid)
        pid = proc.pid
        self._proc_dict[pid] = proc
        return pid

    def launch_tf_server(self, args):
        cmd_list = []
        full_cmd = ' '.join(cmd_list + args)
        logging.info(f"launch {full_cmd}")
        # pylint: disable=subprocess-popen-preexec-fn
        proc = subprocess.Popen(full_cmd, shell=True, preexec_fn=os.setsid,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            line = proc.stderr.readline().decode()
            print(f"LOG {line}")
            if "Started server with target" in line:
                break

        pid = proc.pid
        self._proc_dict[pid] = proc
        return pid

    def join(self, pid):
        self._proc_dict[pid].wait()
        del self._proc_dict[pid]

    def kill(self):
        logging.info('Terminating the Ray node...')
        for pid, p in self._proc_dict.items():
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        self._proc_dict.clear()

    def file_write(self, remote_path, data):
        with open(remote_path, 'w') as f:
            f.write(data)


class RayCluster(Cluster):
    """An AutoDist Cluster Based on Ray."""

    def __init__(self, resource_spec=None):
        # this should be the autodist chief
        if not resource_spec:
            # init the ray cluster
            # create the resource spec from ray nodes
            print("Init Ray Cluster")
            self._init_ray_actors()
            resource_spec = ResourceSpec(
                resource_file=os.path.join(DEFAULT_WORKING_DIR, "resource_spec.yml"))

        super().__init__(resource_spec)

    def _init_ray_actors(self):
        ray_head_address = os.getenv("RAY_HEAD_ADDRESS")
        if ray_head_address and not ray.is_initialized():
            ray.init(address=ray_head_address)

        self._actor_dict = {}
        # temporarily just look for GPU resources
        gpu_list = []
        for node in ray.nodes():
            print(f"Node {node}")
            node_ip = node["NodeManagerAddress"]
            gpu_count = node["Resources"].get("GPU")
            if not gpu_count or not node["Alive"]:
               continue
            gpu_list.append((gpu_count, node_ip))

        gpu_list = [t for t in reversed(sorted(gpu_list))]
        print(f"gpu_list {gpu_list}")
        for gpu_count, _ in gpu_list:
            actor = NodeActor.options(num_gpus=gpu_count).remote()
            actor.wait.remote()
            node_ip = ray.get(actor.get_node_ip.remote())
            print(f"node ip {node_ip}")
            self._actor_dict[node_ip] = actor

        chief_address = ray._private.services.get_node_ip_address()
        print(f"chief address: {chief_address} ")
        resource_dict = {}
        resource_dict["nodes"] = []
        for gpu_count, node_ip in gpu_list:
            node_dict = {"address": node_ip, "gpus": [i for i in range(int(gpu_count))] }
            if node_ip == chief_address:
                node_dict["chief"] = True
            resource_dict["nodes"].append(node_dict)

        resource_yaml = yaml.dump(resource_dict)
        print("resource yaml")
        print(resource_yaml)
        for node_ip, actor in self._actor_dict.items():
            ray.get(actor.file_write.remote(
                os.path.join(DEFAULT_WORKING_DIR, "resource_spec.yml"), resource_yaml))

    def remote_exec(self, args, hostname):
        """
        Execute a bash script remotely.

        Args:
            args (list): bash commands
            hostname (str): means ip address in a ray cluster

        Returns:
            Process: process handle
        """
        return ray.get(self._actor_dict[hostname].execute_cmd.remote(args))

    def remote_join(self, pid, hostname):
        return ray.get(self._actor_dict[hostname].join.remote(pid))

    def remote_file_write(self, remote_path, data, hostname):
        """
        Write a remote file.

        Args:
            remote_path (str): remote file path
            data (str): data to be written
            hostname (str): host name or address
        """
        ray.get(self._actor_dict[hostname].file_write.remote(remote_path, data))

    def remote_copy(self, local_path, remote_path, hostname):
        """
        Copy a file to a remote directory.

        Args:
            local_path (str): local file path to be copied
            remote_path (str): remote directory path
            hostname (str): host name or address
        """
        # use remote_file_write internally
        with open(local_path,"r") as f:
            data = f.read()

        self.remote_file_write(os.path.join(remote_path, os.path.basename(local_path)), data, hostname)

    # pylint: disable=too-many-locals
    def start(self):
        """
        Start tf.servers on all nodes.

        Note that this only runs (and only should run) on the chief node.
        """
        # pylint: disable=import-outside-toplevel
        from autodist.utils import server_starter

        # atexit registration should be placed
        #   - before the beginning of the start
        #   (to ensure the clean termination if the start fails in its half way); and
        #   - at the same module as the start
        #   (to follow the python assumption that
        #   lower level modules will normally be imported
        #   before higher level modules and thus must be cleaned up later).
        atexit.register(self.terminate)
        envs = {ENV.AUTODIST_MIN_LOG_LEVEL.name: 'ERROR'}
        envs = ['{}={}'.format(k, v) for k, v in envs.items()]
        module_name = server_starter.__name__
        module_file = server_starter.__file__

        for job_name, tasks in self.cluster_spec.items():
            for task_index, full_address in enumerate(tasks):
                address, port = full_address.split(':')
                args = ['--job_name=%s' % job_name, '--task_index=%d' % task_index,
                        '--cpu_device_num=%d' % len(self._cpu_devices[address])]
                if address in self._gpu_devices:
                    envs_cuda = []
                else:
                    envs_cuda = ['CUDA_VISIBLE_DEVICES=""']

                self._remote_pre_start_tf_server(address, tf_server_starter_filepath=module_file)
                file = os.path.join(DEFAULT_WORKING_DIR, os.path.basename(module_file))
                bash = envs + envs_cuda + ['python', '-u', file] + args
                logging.info(f"Launching tf.server on {address} with {bash}")
                # self.remote_exec(bash, hostname=address)
                ray.get(self._actor_dict[address].launch_tf_server.remote(bash))

        # time.sleep(10)

    def terminate(self):
        # call actor methods to cleanup tf servers
        # cleanup ray actors
        for node_ip, actor in self._actor_dict.items():
            ray.get(actor.kill.remote())
            ray.get(actor.send.remote())

        self._actor_dict.clear()
        ray.shutdown()
