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

import os
import signal
from multiprocessing import Process

from autodist.const import DEFAULT_PORT_RANGE, DEFAULT_WORKING_DIR
from autodist.utils import single_server_starter
from autodist.utils.network import remote_pre_start_tf_server, remote_exec, is_local_address, colored


class Cluster:
    """Cluster manager for TensorFlow servers."""

    def __init__(self, resource_spec):

        self.cluster_spec = self._get_default_cluster_spec(resource_spec)
        self.ssh_config = resource_spec.ssh_config
        self.subprocesses = []
        self.processes = []
        print(self.cluster_spec)

    @staticmethod
    def _get_default_cluster_spec(resource_spec):

        return {
            'worker': [
                '{ip}:{port}'.format(
                    ip=n,
                    port=next(DEFAULT_PORT_RANGE)
                ) for n in resource_spec.nodes
            ]
        }

    def start(self):
        """Start."""
        for job_name, tasks in self.cluster_spec.items():
            for task_index, full_address in enumerate(tasks):
                address = full_address.split(':')[0]
                if is_local_address(address):  # TODO: more rigorous checking
                    proc = Process(target=single_server_starter.start_server,
                                   args=(self.cluster_spec, job_name, task_index), daemon=True)
                    self.processes.append(proc)
                    proc.start()
                    print(colored('$ local tf.server started at {}: job_name={} task_index={}'.format(
                        full_address, job_name, task_index
                    )))
                else:  # remote
                    remote_pre_start_tf_server(
                        DEFAULT_WORKING_DIR,
                        tf_server_starter_filepath=single_server_starter.__file__,
                        cluster_spec=self.cluster_spec,
                        hostname=address,
                        ssh_config=self.ssh_config
                    )

                    file = os.path.join(DEFAULT_WORKING_DIR, os.path.basename(single_server_starter.__file__))
                    args = [
                        '--job_name=%s' % job_name,
                        '--task_index=%d' % task_index
                    ]
                    bash = ['python', '-u', file] + args
                    proc = remote_exec(
                        bash,
                        hostname=address,
                        ssh_config=self.ssh_config
                    )

                    p = Process(target=proc.wait, daemon=True)
                    p.start()

                    self.subprocesses.append(proc)
                    self.processes.append(p)

    def terminate(self):
        """Terminate."""
        for proc in self.subprocesses:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for p in self.processes:
            p.terminate()
