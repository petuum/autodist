import sys
import os
import time
from enum import Enum, auto

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training.server_lib import ClusterSpec, Server

from autodist import AutoDist
from autodist.strategy import PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax
from autodist.const import ENV, DEFAULT_PORT_RANGE, DEFAULT_WORKING_DIR, DEFAULT_GROUP_LEADER
from autodist.resource_spec import ResourceSpec

import ray

from ray.cluster_utils import Cluster

ray.init(address='auto')

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
                #log_device_placement=True
            )
        )
        s.join()


class TFRunner:
    def __init__(self, model, data_creator, train_step, env, rs):
        import os
        for k, v in env.items():
            if type(v) == bool:
                os.environ[k] = "True" if v else "False"
            else:
                os.environ[k] = v

        self.autodist = AutoDist(resource_spec=rs, strategy_builder=PS())
        self.g = v1.Graph()
        with self.g.as_default(), self.autodist.scope():
            self.fetches = train_step(Model(), *data_creator())
            self.session = self.autodist.create_distributed_session()

    def step(self):
        with self.g.as_default(), self.autodist.scope():
            l, t, b = self.session.run(self.fetches)
            print('node: {}, loss: {}\nb:{}'.format(self.autodist._cluster.get_local_address(), l, b))

    def _get_strategy_id(self):
        return self.autodist._


class TFTrainer:
    def __init__(self, model, data_creator, train_step):
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
                    ENV.AUTODIST_STRATEGY_ID.name: "20210224T233038M775422",
                    #ENV.AUTODIST_STRATEGY_ID.name: "",
                    ENV.AUTODIST_MIN_LOG_LEVEL.name: ENV.AUTODIST_MIN_LOG_LEVEL.val,
                    ENV.AUTODIST_IS_TESTING.name: ENV.AUTODIST_IS_TESTING.val,
                    ENV.AUTODIST_PATCH_TF.name: ENV.AUTODIST_PATCH_TF.val,
                    ENV.AUTODIST_INTERNAL_TF.name: ENV.AUTODIST_INTERNAL_TF.val,
                    ENV.SYS_DATA_PATH.name: ENV.SYS_DATA_PATH.val,
                    ENV.SYS_RESOURCE_PATH.name: ENV.SYS_RESOURCE_PATH.val,
                    #'AUTODIST_RESOURCE_SPEC': self._resource_spec,
                    #'val': property(lambda x: x.value)
                }

                runner = Runner.remote(model, data_creator, train_step, env, self._resource_spec)
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


############################################################################


EPOCHS = 10

def data_creator():
    TRUE_W = 3.0
    TRUE_b = 2.0
    NUM_EXAMPLES = 1000

    inputs = np.random.randn(NUM_EXAMPLES)
    noises = np.random.randn(NUM_EXAMPLES)
    outputs = inputs * TRUE_W + TRUE_b + noises

    class MyIterator:
        def initialize(self):
            return tf.zeros(1)
        def get_next(self):
            # a fake one
            return inputs
    return MyIterator().get_next(), outputs


class Model:
    def __init__(self):
        self.W = tf.Variable(5.0, name='W', dtype=tf.float64)
        self.b = tf.Variable(0.0, name='b', dtype=tf.float64)

    def __call__(self, x):
        return self.W * x + self.b


def train_step(model, inputs, outputs):
    def l(predicted_y, desired_y):
        return tf.reduce_mean(tf.square(predicted_y - desired_y))

    major_version, _, _ = tf.version.VERSION.split('.')
    if major_version == '1':
        optimizer = tf.train.GradientDescentOptimizer(0.01)
    else:
        optimizer = tf.optimizers.SGD(0.01)

    loss = l(model(inputs), outputs)
    vs = [model.W, model.b]

    gradients = tf.gradients(loss, vs)

    train_op = optimizer.apply_gradients(zip(gradients, vs))
    return loss, train_op, model.b


def main(_):
    trainer = TFTrainer(Model, data_creator, train_step)
    for epoch in range(EPOCHS):
        trainer.train()

    trainer.shutdown()

main(sys.argv)

