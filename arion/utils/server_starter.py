"""TensorFlow Server Starter."""

import argparse
import json
import os
import subprocess

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training.server_lib import ClusterSpec, Server

from autodist.const import DEFAULT_WORKING_DIR, DEFAULT_GROUP_LEADER


def _clean_stale_server():
    cmd = "ps aux | awk '\\''! /awk/ && ! /{}/ /{}/ {{print \\$2}}'\\'' | xargs kill -9".format(
        os.getpid(),
        os.path.splitext(os.path.basename(__file__))
    )
    # The above `cmd` is escaped for a string-within-a-string, so we have to nest `bash -c`
    # There's probably a better way to do this
    local_cmd = 'bash -c \'bash -c "{}"\''.format(cmd)
    proc = subprocess.Popen(local_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()


def start_server(cluster_spec, job_name: str, task_index: int):
    """
    Start a TensorFlow server.

    Args:
        cluster_spec (dict): TensorFlow ClusterSpec dict
        job_name: TensorFlow job name
        task_index: TensorFlow task index
    """
    _clean_stale_server()

    # TODO: The following config should be less hard coded ad based on strategy
    experimental = config_pb2.ConfigProto.Experimental(
        collective_nccl=True,
        collective_group_leader=DEFAULT_GROUP_LEADER)
    s = Server(
        ClusterSpec(cluster_spec),
        job_name=job_name,
        task_index=task_index,
        config=config_pb2.ConfigProto(experimental=experimental)
    )
    s.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_name",
        help="tensorflow server job name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--task_index",
        help="tensorflow server task index",
        type=int,
        required=True
    )
    parser.add_argument(
        "--cluster_spec_filename",
        help="filename of tensorflow cluster_spec in json",
        type=str,
        default="cluster_spec.json",
    )

    FLAGS, unparsed = parser.parse_known_args()

    cluster_spec_path = os.path.join(DEFAULT_WORKING_DIR, FLAGS.cluster_spec_filename)
    cluster_spec_dict = json.load(open(cluster_spec_path))

    start_server(cluster_spec_dict, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
