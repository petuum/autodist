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

"""Coordinator."""

import sys
import threading
import atexit
import os

from autodist.const import ENV, DEFAULT_SERIALIZATION_DIR
from autodist.resource_spec import DeviceSpec
from autodist.utils import logging


class Coordinator:
    """
    Coordinator is responsible for running user code on a cluster.

    Has one main method, `launch_clients`, which runs the user's code
    on every node of the cluster. Since all we pass to each worker are
    their worker id and the distribution strategy, this means that each
    worker will do its own full graph transformation based on the strategy.
    Then the workers will sync up when they run the graphs.

    `join` is called by `atexit` so that we can try to make sure the
    remote processes are killed with the chief's process is ended.
    """

    def __init__(self, strategy, cluster):
        self._strategy = strategy
        self.cluster = cluster
        self.threads = []

    def launch_clients(self):
        """
        Launch the user's code on each worker.

        Sets environment variables so that we run the correct AutoDist code paths on workers.
        (i.e., the non-chief code-paths).

        Store each new process created into the class so they can be monitored with `join`.
        """
        atexit.register(self.join)

        replica_devices = [
            DeviceSpec.from_string(device_string)
            for device_string in self._strategy.graph_config.replicas
        ]
        replica_hosts = {d.host_address for d in replica_devices}

        # Assumption: Master node must run one replica.
        # assert any([is_local_address(h) for h in replica_hosts])

        for replica_host in replica_hosts:
            # Run the process
            if not self.cluster.is_chief(replica_host):
                # Build the command
                env = {
                    ENV.AUTODIST_WORKER.name: replica_host,
                    ENV.AUTODIST_STRATEGY_ID.name: self._strategy.id,
                    ENV.AUTODIST_MIN_LOG_LEVEL.name: ENV.AUTODIST_MIN_LOG_LEVEL.val,
                    ENV.AUTODIST_IS_TESTING.name: ENV.AUTODIST_IS_TESTING.val,
                    ENV.AUTODIST_PATCH_TF.name: ENV.AUTODIST_PATCH_TF.val,
                    ENV.AUTODIST_INTERNAL_TF.name: ENV.AUTODIST_INTERNAL_TF.val,
                    ENV.SYS_DATA_PATH.name: ENV.SYS_DATA_PATH.val,
                    ENV.SYS_RESOURCE_PATH.name: ENV.SYS_RESOURCE_PATH.val,
                }
                cmd_env = ['{}={}'.format(k, v) for k, v in env.items()]
                cmd_main = ["python"] + sys.argv
                cmd = cmd_env + cmd_main

                self.cluster.remote_copy(
                    local_path=self._strategy.path,
                    remote_path=DEFAULT_SERIALIZATION_DIR,
                    hostname=replica_host
                )
                proc = self.cluster.remote_exec(cmd, hostname=replica_host)
                self.threads.append(self._proc_wait_async(proc))

    def join(self):
        """Wait for all subprocesses of remote workers to be completed."""
        logging.debug('Joining workers...')
        for t in self.threads:
            t.join()

    @staticmethod
    def _proc_wait_async(proc, on_exit=lambda: os._exit(1)):
        """Creates a thread to wait on the given proc finishing."""
        def run_subprocess_in_thread(proc, on_exit):
            proc.communicate()
            if proc.poll():
                print('RuntimeError: A remote AutoDist worker raised an exception. See Above.')
                on_exit()

        thread = threading.Thread(target=run_subprocess_in_thread, args=(proc, on_exit))
        thread.start()
        # returns immediately after the thread starts
        return thread
