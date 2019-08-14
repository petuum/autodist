"""Coordinator."""

import sys
import threading

from autodist.const import Env, DEFAULT_SERIALIZATION_DIR
from autodist.resource_spec import DeviceSpec
from autodist.utils.network import is_local_address, remote_exec, remote_copy


class Coordinator:
    """Coordinator to manager TF cluster and processes of workers."""

    def __init__(self, strategy, cluster):

        self._strategy = strategy
        self.cluster = cluster
        self.threads = []

    def launch_clients(self):
        """Launch."""
        replica_devices = [
            DeviceSpec.from_string(device_string)
            for device_string in self._strategy.graph_config.get('replicas', {})
        ]
        replica_hosts = {d.host_address for d in replica_devices}

        # Assumption: Master node must run one replica.
        # assert any([is_local_address(h) for h in replica_hosts])

        for replica_host in replica_hosts:
            # Run the process
            if not is_local_address(replica_host) and not self.cluster.is_chief(replica_host):
                # Build the command
                env = {
                    Env.AUTODIST_WORKER.name: replica_host,
                    Env.AUTODIST_STRATEGY_ID.name: self._strategy.get_id()
                }
                cmd_env = ['{}={}'.format(k, v) for k, v in env.items()]
                cmd_main = ["python"] + sys.argv
                cmd = cmd_env + cmd_main

                remote_copy(
                    local_path=self._strategy.path,
                    remote_path=DEFAULT_SERIALIZATION_DIR,
                    hostname=replica_host,
                    ssh_config=self.cluster.ssh_config
                )
                proc = remote_exec(
                    cmd,
                    hostname=replica_host,
                    ssh_config=self.cluster.ssh_config
                )
                self.threads.append(self._proc_wait_async(proc))

    def join(self):
        """Wait for all subprocesses of remote workers to be completed."""
        for t in self.threads:
            t.join()

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
