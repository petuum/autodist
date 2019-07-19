"""Coordinator."""

import subprocess
import sys
import threading

from autodist.cluster import Cluster
from autodist.const import Env
from autodist.resource_spec import DeviceSpec
from autodist.utils.network import is_local_address, remote_exec, colored


class Coordinator:
    """Coordinator to manager TF cluster and processes of workers."""

    def __init__(self, strategy, resource_spec):

        self._strategy = strategy
        self._resource_spec = resource_spec

        self.cluster = Cluster(resource_spec)

    def launch_cluster(self):
        """Launch."""
        # Adapter, TF Clusters start
        self.cluster.start()  # start the tf cluster
        return self

    def launch_clients(self):
        """Launch."""
        replica_devices = [
            DeviceSpec.from_string(device_string)
            for device_string in self._strategy.graph_config.get('replicas', {})
        ]
        replica_hosts = {d.host_address for d in replica_devices}

        threads = []
        for replica_host in replica_hosts:
            is_local = is_local_address(replica_host)

            # Build the command
            env = {
                Env.AUTODIST_WORKER.name: 'true',
                Env.AUTODIST_STRATEGY_ID.name: self._strategy.get_id()
            }
            cmd_env = ['{}={}'.format(k, v) for k, v in env.items()]
            cmd_main = [sys.executable if is_local else "python"] + sys.argv
            cmd = cmd_env + cmd_main

            # Run the process
            if is_local:
                proc = subprocess.Popen(' '.join(cmd), shell=True)
                print(colored('$ ' + ' '.join(cmd)))
            else:
                # remote_copy(
                #     local_path=self._strategy.path,
                #     remote_path=DEFAULT_SERIALIZATION_DIR,
                #     hostname=replica_host,
                #     ssh_config=self._resource_spec.ssh_config
                # )
                proc = remote_exec(
                    cmd,
                    hostname=replica_host,
                    ssh_config=self._resource_spec.ssh_config
                )
            threads.append(self._proc_wait_async(proc))

    def terminate(self):
        """Terminate."""
        pass

    @staticmethod
    def _proc_wait_async(proc, on_exit=lambda: None):
        """Creates a thread to wait on the given proc finishing."""
        def run_subprocess_in_thread(proc, on_exit):
            # proc = subprocess.Popen(*popen_args)
            proc.wait()
            on_exit()

        thread = threading.Thread(target=run_subprocess_in_thread, args=(proc, on_exit))
        thread.start()
        # returns immediately after the thread starts
        return thread
