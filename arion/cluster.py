"""
Cluster.

The experimental nodes launcher.

Prerequisite:
* TensorFlow is already installed in the env of all nodes.
* Only support in graph launching logic. Only one node `NODE 0` runs the session client.
* AutoDist is already installed in the env of the worker node `NODE 0`, where the main script runs.
* The open ssh private key to other nodes is accessible on `NODE 0` given a path.
All other nodes are added in the `known_host` of `NODE 0`.
"""
import atexit
import json
import os
import signal
import subprocess
import sys

import autodist.utils.server_starter
from autodist.const import DEFAULT_PORT_RANGE, DEFAULT_WORKING_DIR, Env
from autodist.utils import logging
from autodist.utils.network import remote_pre_start_tf_server, remote_exec, is_local_address, colored


class Cluster:
    """Cluster manager for TensorFlow servers."""

    def __init__(self, resource_spec):

        self._chief = resource_spec.chief
        self.cluster_spec = self._get_default_cluster_spec(resource_spec)
        self._full_addresses = [full_address for tasks in self.cluster_spec.values() for full_address in tasks]
        self._address_to_port = {
            ip: port
            for ip, port in (a.split(':') for a in self._full_addresses)
        }
        self._task_to_address = {
            (job_name, task_index): a.split(':')[0]
            for job_name, tasks in self.cluster_spec.items()
            for task_index, a in enumerate(tasks)
        }

        self.ssh_config = resource_spec.ssh_config
        self.subprocesses = []
        logging.info('ClusterSpec: {}'.format(self.cluster_spec))

    @staticmethod
    def _get_default_cluster_spec(resource_spec):

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

    def is_chief(self, address=None):
        """
        Check whether an address is chief or not.

        If the argument `address` is not provided,
        it will check whether the local address is chief.

        Args:
            address (str): node address e.g. ip

        Returns:
            bool:
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
            str
        """
        return self._task_to_address[(job_name, task_index)]

    def get_local_address(self):
        """
        Get the local (ip) address.

        Returns:
            str: worker ip or chief address by default.
        """
        return os.environ.get(Env.AUTODIST_WORKER.name, self._chief)

    def get_local_worker_task_index(self):
        """
        Get the (first) TensorFlow task index of the "worker" for the local.

        Returns:
            int: task index
        """
        return [i for i, a in enumerate(self._full_addresses) if self.get_local_address() in a][0]

    def get_local_session_target(self):
        """
        Get the session target of the local session.

        Returns:
            str:
        """
        port = self._address_to_port[self.get_local_address()]
        return 'grpc://localhost:' + port

    def start(self):
        """Start."""
        # atexit registration should be placed
        #   - before the beginning of the start
        #   (to ensure the clean termination if the start fails in its half way); and
        #   - at the same module as the start
        #   (to follow the python assumption that
        #   lower level modules will normally be imported
        #   before higher level modules and thus must be cleaned up later).
        atexit.register(self.terminate)

        for job_name, tasks in self.cluster_spec.items():
            for task_index, full_address in enumerate(tasks):
                address = full_address.split(':')[0]
                if is_local_address(address) or self.is_chief(address):  # TODO: more rigorous checking
                    json.dump(self.cluster_spec, open(os.path.join(DEFAULT_WORKING_DIR, 'cluster_spec.json'), 'w+'))

                    module_name = autodist.utils.server_starter.__name__
                    args = [
                        '--job_name=%s' % job_name,
                        '--task_index=%d' % task_index
                    ]
                    cmd = [sys.executable, '-m', module_name] + args

                    # pylint: disable=subprocess-popen-preexec-fn
                    proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
                    self.subprocesses.append(proc)
                    # The above line immediately follows the Popen
                    # to ensure no gap for termination failure due to the empty proc list.
                    print(colored('$ local tf.server started at {}: job_name={} task_index={}'.format(
                        full_address, job_name, task_index
                    )))
                else:  # remote
                    remote_pre_start_tf_server(
                        DEFAULT_WORKING_DIR,
                        tf_server_starter_filepath=autodist.utils.server_starter.__file__,
                        cluster_spec=self.cluster_spec,
                        hostname=address,
                        ssh_config=self.ssh_config
                    )
                    file = os.path.join(DEFAULT_WORKING_DIR, os.path.basename(autodist.utils.server_starter.__file__))
                    args = [
                        '--job_name=%s' % job_name,
                        '--task_index=%d' % task_index
                    ]
                    bash = ['python', '-u', file] + args
                    proc = remote_exec(bash, hostname=address, ssh_config=self.ssh_config)
                    # The above line immediately follows the Popen
                    # to ensure no gap for termination failure due to the empty proc list.
                    self.subprocesses.append(proc)

    def terminate(self):
        """Terminate."""
        logging.info('Terminating cluster...')
        for p in self.subprocesses:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
