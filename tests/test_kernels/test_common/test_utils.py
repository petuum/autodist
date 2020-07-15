

import os, threading
import tensorflow as tf
from tensorflow.python import ops
from tensorflow.python import constant_op
from tensorflow.python.ops.gen_control_flow_ops import no_op

from autodist.kernel.common import utils
from autodist.kernel.common.utils import replica_prefix, strip_replica_prefix


def test_parse_name_scope():
    with ops.Graph().as_default():
        name_scope = 'name_scope/child_name_scope'
        a = constant_op.constant(5)
        new_name = ops.prepend_name_scope(a.name, name_scope)
        assert new_name == 'name_scope/child_name_scope/Const:0'
        assert name_scope == utils.parse_name_scope(new_name)
        assert '' == utils.parse_name_scope(a.name)


        with ops.control_dependencies([no_op(name='my_op')]):
            b = constant_op.constant(6)
        name_scope = 'name_scope'
        new_name = ops.prepend_name_scope(b.op.node_def.input[0], name_scope)
        assert new_name == '^name_scope/my_op'
        assert name_scope == utils.parse_name_scope(new_name)


def test_strip_replica_prefix():
    for name in ['my_op', '^my_op', 'my_tensor:0']:
        new_name = ops.prepend_name_scope(name, replica_prefix(12))
        assert strip_replica_prefix(new_name) == name

def test_server_starter():
    cluster = tf.train.ClusterSpec({
        'worker1': ['localhost:15001'],
        'worker2': ['localhost:15002']
    })

    def start(job_name, task_index):

        from autodist.utils.server_starter import gen_server
        server = gen_server(cluster, job_name=job_name, task_index=task_index, 
                     cpu_device_num=1)

        with tf.Graph().as_default():
            if job_name == 'worker1':
                with tf.device('/job:worker1/task:%d' % task_index):
                    queue = tf.compat.v1.FIFOQueue(cluster.num_tasks('worker2'), tf.int32, shared_name='queue%d' % task_index)
                with tf.compat.v1.Session(server.target) as sess:
                    for i in range(cluster.num_tasks('worker2')):
                        sess.run(queue.dequeue())

            elif job_name == 'worker2':
                queues = []
                for i in range(cluster.num_tasks('worker1')):
                    with tf.device('/job:worker1/task:%d' % i):
                        queues.append(tf.compat.v1.FIFOQueue(cluster.num_tasks('worker2'), tf.int32, shared_name='queue%d' % i))
                with tf.compat.v1.Session(server.target) as sess:
                    for i in range(cluster.num_tasks('worker1')):
                        sess.run(queues[i].enqueue(task_index))

    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    threads = [
        threading.Thread(target=start, args=('worker1', 0)),
        threading.Thread(target=start, args=('worker2', 0))
        ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

