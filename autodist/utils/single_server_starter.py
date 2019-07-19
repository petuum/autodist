"""TensorFlow Server Starter."""

import argparse
import json
import os

from tensorflow.python.training.server_lib import ClusterSpec, Server


def start_server(cluster_spec, job_name: str, task_index: int):
    """
    Start a TensorFlow server.

    Args:
        cluster_spec (dict): TensorFlow ClusterSpec dict
        job_name: TensorFlow job name
        task_index: TensorFlow task index
    """
    s = Server(
        ClusterSpec(cluster_spec),
        job_name=job_name,
        task_index=task_index
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

    cluster_spec_path = os.path.join(os.path.dirname(__file__), FLAGS.cluster_spec_filename)
    cluster_spec_dict = json.load(open(cluster_spec_path))

    start_server(cluster_spec_dict, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
