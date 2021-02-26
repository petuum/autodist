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
from autodist.const import ENV, DEFAULT_PORT_RANGE, DEFAULT_WORKING_DIR, DEFAULT_GROUP_LEADER
from autodist.resource_spec import ResourceSpec


@ray.remote
class TFServerActor(object):
    def launch(self, cluster_spec, job_name, task_index, cpu_device_num):
        experimental = config_pb2.ConfigProto.Experimental(
            collective_nccl=False,
            collective_group_leader=DEFAULT_GROUP_LEADER)
        s = Server(
            ClusterSpec(cluster_spec),
            job_name=job_name,
            task_index=task_index,
            config=config_pb2.ConfigProto(
                experimental=experimental,
                device_count={"CPU": cpu_device_num},
                inter_op_parallelism_threads=0,
                intra_op_parallelism_threads=0,
            )
        )
        s.join()


class TFRunner:
    def __init__(self, model, data_creator, train_step, env, resource_spec, strategy):
        for k, v in env.items():
            if type(v) == bool:
                os.environ[k] = "True" if v else "False"
            else:
                os.environ[k] = v

        self.autodist = AutoDist(resource_spec=resource_spec,
                                 strategy_builder=strategy)

        self.g = v1.Graph()
        with self.g.as_default(), self.autodist.scope():
            self.fetches = train_step(model(), *data_creator())
            self.session = self.autodist.create_distributed_session()

    def step(self):
        with self.g.as_default(), self.autodist.scope():
            l, t, b = self.session.run(self.fetches)
            print(f"loss: {l}  b:{b}")

    def get_strategy_id(self):
        pass


class TFTrainer:
    def __init__(self, model, data_creator, train_step, strategy):
        self._resource_spec = ResourceSpec(resource_info=self._get_resource_info())
        self._cluster_spec = self._get_default_cluster_spec(self._resource_spec)
        self._servers = []
        self._workers = []

        print(self._cluster_spec)
        for job_name, tasks in self._cluster_spec.items():
            for task_index, full_address in enumerate(tasks):
                node_ip, _ = full_address.split(':')
                server = TFServerActor.options(resources={f"node:{node_ip}": 0.01},
                                               num_cpus=1).remote()
                self._servers.append(server)
                server.launch.remote(self._cluster_spec, job_name, task_index, 1)

        for job_name, tasks in self._cluster_spec.items():
            for task_index, full_address in enumerate(tasks):
                node_ip, _ = full_address.split(':')
                Runner = ray.remote(resources={f"node:{node_ip}": 0.01},
                                    num_cpus=1)(TFRunner)
                ischief = node_ip == ray._private.services.get_node_ip_address()
                env = {
                    ENV.AUTODIST_WORKER.name: "" if ischief else node_ip,
                    #ENV.AUTODIST_STRATEGY_ID.name: "20210224T233038M775422",
                    ENV.AUTODIST_STRATEGY_ID.name: "",
                    ENV.AUTODIST_MIN_LOG_LEVEL.name: ENV.AUTODIST_MIN_LOG_LEVEL.val,
                    ENV.AUTODIST_IS_TESTING.name: ENV.AUTODIST_IS_TESTING.val,
                    ENV.AUTODIST_PATCH_TF.name: ENV.AUTODIST_PATCH_TF.val,
                    ENV.AUTODIST_INTERNAL_TF.name: ENV.AUTODIST_INTERNAL_TF.val,
                    ENV.SYS_DATA_PATH.name: ENV.SYS_DATA_PATH.val,
                    ENV.SYS_RESOURCE_PATH.name: ENV.SYS_RESOURCE_PATH.val,
                    #'AUTODIST_RESOURCE_SPEC': self._resource_spec,
                    #'val': property(lambda x: x.value)
                }

                runner = Runner.remote(model, data_creator, train_step, env, self._resource_spec, strategy)
                self._workers.append(runner)

    @staticmethod
    def _get_default_cluster_spec(resource_spec: ResourceSpec):
        """Create list of workers from the resource spec with semi-arbitrarily chosen ports."""
        return {
            'worker': [f'{node}:{next(DEFAULT_PORT_RANGE)}'
                        for node in sorted(resource_spec.nodes, reverse=True)]
        }

    def _get_resource_info(self):
        cpu_list = []
        for node in ray.nodes():
            node_ip = node["NodeManagerAddress"]
            cpu_count = node["Resources"].get("CPU")
            if not cpu_count or not node["Alive"]:
               continue
            cpu_list.append((cpu_count, node_ip))

        chief_address = ray._private.services.get_node_ip_address()

        resource_info = {}
        resource_info["nodes"] = []
        for cpu_count, node_ip in cpu_list:
            node = {"address": node_ip, "cpus": [0]}
            if node_ip == chief_address:
                node["chief"] = True
            resource_info["nodes"].append(node)
        sorted(resource_info["nodes"], key=lambda x: x.get("chief", False))
        return resource_info

    def train(self):
        """Runs a training epoch."""
        ray.get([worker.step.remote() for worker in self._workers])

    def validate(self):
        pass

    def shutdown(self):
        for server, worker in zip(self._servers, self._workers):
            ray.kill(worker)
            ray.kill(server)

    def save(self):
        pass

    def restore(self):
        pass
