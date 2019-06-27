import os
import signal
import argparse
from multiprocessing import Process
import json

from .remote_lib import remote_exec, remote_pre_start_tf_server
from . import single_server_starter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", "-m",
    help='"local": all servers are local processes'
         '"remote": mode for multiple machines"',
    type=str,
    choices=['local', 'remote'],
    default='local'
)
parser.add_argument(
    "--cluster_spec", "-f",
    help="cluster spec json file path",
    type=str,
)
parser.add_argument(
    "--private_key", "-i",
    help="private key file path",
    type=str,
)
parser.add_argument(
    "--venv",
    help="command to source venv",
    type=str,
    default='source activate tensorflow_p36'  # aws cluster
)
FLAGS, _ = parser.parse_known_args()


REMOTE_USERNAME = 'ubuntu'  # aws cluster
REMOTE_WORKING_DIR = '/tmp/autodist'
REMOTE_PYTHON_VENV = FLAGS.venv
PRIVATE_KEY_FILEPATH = FLAGS.private_key
CLUSTER_SPEC_FILEPATH = FLAGS.cluster_spec  # 'cluster_spec.json'
SERVER_STARTER_FILENAME = os.path.basename(single_server_starter.__file__)


class TFCluster:

    def __init__(self):
        self.cluster_spec = json.load(open(CLUSTER_SPEC_FILEPATH))
        self.processes = []

    def start(self):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError


class LocalTFCluster(TFCluster):
    def __init__(self):
        super().__init__()

    def start(self):

        for k, v in self.cluster_spec.items():
            for i, a in enumerate(v):
                proc = Process(target=single_server_starter.start_server,
                               args=(self.cluster_spec, k, i), daemon=False)
                self.processes.append(proc)
                proc.start()

    def terminate(self):
        for proc in self.processes:
            proc.terminate()


class RemoteTFCluster(TFCluster):

    def __init__(self):
        super().__init__()
        self.subprocesses = []

    def start(self):

        for k, v in self.cluster_spec.items():
            for i, a in enumerate(v):
                d = dict(
                    job_name=k,
                    task_index=i,
                    address=a.split(':')[0]
                )

                remote_pre_start_tf_server(
                    REMOTE_WORKING_DIR,
                    SERVER_STARTER_FILENAME,
                    CLUSTER_SPEC_FILEPATH,
                    hostname=d.get('address'),
                    username=REMOTE_USERNAME,
                    key_file=PRIVATE_KEY_FILEPATH
                )

                file = os.path.join(REMOTE_WORKING_DIR, SERVER_STARTER_FILENAME)
                args = [
                    '--job_name=%s' % d.get('job_name'),
                    '--task_index=%d' % d.get('task_index')
                ]
                bash = ' '.join(['python', '-u', file] + args)
                proc = remote_exec(
                    bash,
                    hostname=d.get('address'),
                    username=REMOTE_USERNAME,
                    key_file=PRIVATE_KEY_FILEPATH,
                    env={
                        'TF_CPP_MIN_LOG_LEVEL': '0'
                    },
                    python_venv=REMOTE_PYTHON_VENV
                )

                p = Process(target=proc.wait, daemon=True)
                p.start()

                self.subprocesses.append(proc)
                self.processes.append(p)

    def terminate(self):
        for proc in self.subprocesses:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for p in self.processes:
            p.terminate()


if __name__ == '__main__':
    Cluster = RemoteTFCluster if FLAGS.mode == 'remote' else LocalTFCluster
    c = Cluster()
    c.start()

    def shutdown(recv_signal, frame):
        c.terminate()
        print('\n[AutoDist] TF cluster terminated.')
    signal.signal(signal.SIGINT, shutdown)

    signal.pause()
