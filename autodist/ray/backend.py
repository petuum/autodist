# Copyright 2021 Petuum, Inc. All Rights Reserved.
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


import os
import ray
import tensorflow as tf
import tensorflow.compat.v1 as v1
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training.server_lib import ClusterSpec, Server

from autodist import AutoDist
from autodist.const import ENV, DEFAULT_GROUP_LEADER
from autodist.resource_spec import ResourceSpec
from autodist.resource_spec import DeviceSpec
from autodist.cluster import Cluster


@ray.remote
class TFServer:
    def launch(self, cluster_spec, job_name, task_index, num_cpu_device):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        experimental = config_pb2.ConfigProto.Experimental(
            collective_nccl=True,
            collective_group_leader=DEFAULT_GROUP_LEADER)
        s = Server(
            ClusterSpec(cluster_spec),
            job_name=job_name,
            task_index=task_index,
            config=config_pb2.ConfigProto(
                experimental=experimental,
                device_count={"CPU": num_cpu_device},
                inter_op_parallelism_threads=0,
                intra_op_parallelism_threads=0,
            )
        )
        s.join()


class TFRunner:
    def __init__(self,
                 strategy_builder,
                 strategy,
                 model,
                 data_creator,
                 train_step,
                 env,
                 resource_spec):
        # Setup environment vars for the new runner
        for var, val in env.items():
            if type(val) == bool:
                os.environ[var] = "True" if val else "False"
            else:
                os.environ[var] = val

        # We either pass a strategy_builder or directly a strategy
        self._autodist = AutoDist(strategy_builder=strategy_builder,
                                  strategy=strategy,
                                  resource_spec=resource_spec)
        self._g = v1.Graph()
        with self._g.as_default(), self._autodist.scope():
            self._fetches = train_step(model(), *data_creator())
            self._session = self._autodist.create_distributed_session()

    def step(self):
        with self._g.as_default(), self._autodist.scope():
            l, t, b = self._session.run(self._fetches)
            print(f"loss: {l}\tb:{b}")

    def get_strategy(self):
        return self._autodist._strategy


class TFTrainer:
    def __init__(self, strategy_builder, model, data_creator, train_step):

        # Go from resource_info -> ResourceSpec -> ClusterSpec
        self._resource_spec = ResourceSpec(
            resource_info=self._get_resource_info())

        self._replicas = []   # Replica actors, also contains master

        # Start TF Servers on each node of the cluster
        self._servers = self._start_tf_servers(self._resource_spec)

        def spawn_replica(replica_host, strategy_builder, strategy=None, env={}):
            # Enforce actor placement on the provided host
            Runner = ray.remote(resources={f"node:{replica_host}": 0.01},
                                num_cpus=1)(TFRunner)
            return Runner.remote(strategy_builder,
                                 strategy,
                                 model,
                                 data_creator,
                                 train_step,
                                 env,
                                 self._resource_spec)


        # Start the master worker, let it build a strategy from the strategy builder
        master = spawn_replica(ray._private.services.get_node_ip_address(), strategy_builder)

        # Add master to replicas list because it also acts as one of the clients
        self._replicas.append(master)

        # Fetch the strategy directly from the master
        strategy = ray.get(master.get_strategy.remote())

        assert strategy is not None

        # Spawn clients based on the strategy built by master
        replica_devices = [
            DeviceSpec.from_string(device_string)
            for device_string in strategy.graph_config.replicas
        ]

        replica_hosts = {d.host_address for d in replica_devices}
        for replica_host in replica_hosts:
            if replica_host != ray._private.services.get_node_ip_address():
                # Only non-master replicas
                env = {
                    ENV.AUTODIST_WORKER.name: replica_host,
                    ENV.AUTODIST_MIN_LOG_LEVEL.name: ENV.AUTODIST_MIN_LOG_LEVEL.val,
                    ENV.AUTODIST_IS_TESTING.name: ENV.AUTODIST_IS_TESTING.val,
                    ENV.AUTODIST_PATCH_TF.name: ENV.AUTODIST_PATCH_TF.val,
                    ENV.AUTODIST_INTERNAL_TF.name: ENV.AUTODIST_INTERNAL_TF.val,
                    ENV.SYS_DATA_PATH.name: ENV.SYS_DATA_PATH.val,
                    ENV.SYS_RESOURCE_PATH.name: ENV.SYS_RESOURCE_PATH.val,
                }
                self._replicas.append(spawn_replica(replica_host, None, strategy, env))

    def _start_tf_servers(self, resource_spec):
        cluster_spec = Cluster._get_default_cluster_spec(resource_spec)
        cpu_devices = Cluster._get_node_cpu_devices(resource_spec)
        gpu_devices = Cluster._get_node_gpu_devices(resource_spec)

        servers = []
        for job_name, tasks in cluster_spec.items():
            for task_index, full_address in enumerate(tasks):
                node_ip, _ = full_address.split(':')
                # Make sure we spawn one server per Ray node
                # Give it all the GPUs on that node
                server = TFServer.options(resources={f"node:{node_ip}": 0.01},
                                          num_gpus=gpu_devices.get('node_ip', 0)).remote()
                servers.append(server)
                server.launch.remote(cluster_spec, 
                                     job_name, 
                                     task_index,
                                     len(cpu_devices[node_ip]))
        return servers

    def _get_resource_info(self):
        resource_info = {}
        resource_info["nodes"] = []
        chief_address = ray._private.services.get_node_ip_address()
        for node in ray.nodes():
            node_ip = node["NodeManagerAddress"]
            cpu_count = node["Resources"].get("CPU")
            gpu_count = node["Resources"].get("GPU")
            if not node["Alive"] or (cpu_count is None and gpu_count is None):
                continue
            node = {"address": node_ip, 
                    "cpus": [0] if cpu_count else [], 
                    "gpus": list(range(int(gpu_count))) if gpu_count else []}
            if node_ip == chief_address:
                node["chief"] = True
            resource_info["nodes"].append(node)
        return resource_info

    def train(self):
        """Runs a training epoch."""
        ray.get([replica.step.remote() for replica in self._replicas])

    def validate(self):
        pass

    def shutdown(self):
        for server in self._servers:
            ray.kill(server)
        for replica in self._replicas:
            ray.kill(replica)

    def save(self):
        pass

    def restore(self):
        pass
