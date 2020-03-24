"""TensorFlow Server Starter."""

import argparse
import json
import os
import subprocess

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training.server_lib import ClusterSpec, Server

from autodist.const import DEFAULT_WORKING_DIR, DEFAULT_GROUP_LEADER
from autodist.utils import logging


def _clean_stale_servers():
    # pylint: disable=anomalous-backslash-in-string
    cmd = """ps aux | awk "/{}/ && !/ssh/ && ! /{}/ && ! /{}/" | awk "{{print \$2}}" | xargs kill -9"""  # noqa: W605
    # Processes of | the local stale servers && excluding the current starter's pid && ppid | keep pids | kill them
    cmd = cmd.format(
        os.path.splitext(os.path.basename(__file__))[0],
        os.getpid(),
        os.getppid()
    )
    local_cmd = "bash -c '{}'".format(cmd)
    logging.debug('>>> {}'.format(local_cmd))
    try:
        output = subprocess.check_output(local_cmd, shell=True, stderr=subprocess.STDOUT)
        logging.debug('>>> {}'.format(output.decode('utf-8')))
    except subprocess.CalledProcessError as e:
        if e.returncode != 123:  # No stale process to kill
            raise


def start_server(cluster_spec, job_name: str, task_index: int):
    """
    Start a TensorFlow server.

    Args:
        cluster_spec (dict): TensorFlow ClusterSpec dict
        job_name: TensorFlow job name
        task_index: TensorFlow task index
    """
    _clean_stale_servers()

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
