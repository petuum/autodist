"""Network utils."""

import json
import os
import subprocess
import warnings

import paramiko

warnings.filterwarnings(action='ignore', module=paramiko.__name__)


class SSHConfig():
    """Contains any necessary SSH information (e.g. passwords, keyfiles, etc.)"""

    def __init__(self, info):
        """
        Initialize the object with a dictionary of SSH information.

        Args:
            info (dict): any SSH information needed for remote control.
        """
        self._username = info.get('username', '')
        self._port = info.get('port', 22)
        self._python_venv = info.get('python_venv', '')
        self._key_file = info.get('key_file', None)
        self._env = dict(
            TF_CPP_MIN_LOG_LEVEL=0,
            **info.get('env', {})
        )

    @property
    def username(self):
        """Username for SSH."""
        return self._username

    @property
    def port(self):
        """Port to SSH into."""
        return self._port

    @property
    def python_venv(self):
        """Remote Python Virtualenv to use when SSHing."""
        return self._python_venv

    @property
    def key_file(self):
        """Key file for SSH authentication."""
        return self._key_file

    @property
    def env(self):
        """Environment variables for SSH host."""
        return self._env


def is_local_address(address):
    """
    Determine whether an address is local.

    Args:
        address: ip

    Returns:
        Boolean
    """
    # TODO: use ipaddress for more rigorous checking
    if address.split(':')[0] == 'localhost':
        return True
    return False


def remote_exec(args,
                hostname,
                ssh_config):
    """
    Execute a bash script remotely.

    Args:
        args (list): bash commands
        hostname (str): host name or address
        username (str): host username
        port (int): host ssh port
        key_file (str): ssh key file path
        env (dict): environment variables to use for the bash script
        python_venv (str): command activating Python virtual environment
            (e.g. "source /home/user/venv/bin/activate")

    Returns:
        Process: process handle
    """
    cmd_list = []
    if ssh_config.python_venv:
        cmd_list.append('%s;' % ssh_config.python_venv)
    if ssh_config.env:
        cmd_list.extend(['%s=%s' % (k, v) for k, v in ssh_config.env.items()])
    full_cmd = ' '.join(cmd_list + args)

    remote_cmd = 'ssh -i %s -tt -p %d %s@%s \'bash -c "%s"\' </dev/null' % (
        ssh_config.key_file, ssh_config.port, ssh_config.username, hostname, full_cmd
    )

    print(colored('\n$ %s' % remote_cmd))

    proc = subprocess.Popen(remote_cmd, shell=True)
    return proc


def colored(msg):
    """Colorful print."""
    return '\033[31m' + msg + '\033[0m'


def remote_file_write(remote_path, data, hostname, ssh_config):
    """
    Write a remote file.

    Args:
        remote_path (str): remote file path
        data (str): data to be written
        hostname (str): host name or address
        username (str): host username
        port (int): host ssh port
        key_file (str): ssh keyfile
    """
    pkey = paramiko.RSAKey.from_private_key_file(
        os.path.expanduser(os.path.abspath(ssh_config.key_file))
    ) if ssh_config.key_file else None
    t = paramiko.Transport((hostname, ssh_config.port))
    t.connect(username=ssh_config.username, pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(t)
    with sftp.open(remote_path, 'w') as f:
        f.write(data)
    sftp.close()
    t.close()


def remote_copy(local_path, remote_path, hostname, ssh_config):
    """
    Copy a file to a remote directory.

    Args:
        local_path (str): local file path to be copied
        remote_path (str): remote directory path
        hostname (str): host name or address
        username (str): host username
        port (int): host ssh port
        key_file (str): ssh keyfile
    """
    pkey = paramiko.RSAKey.from_private_key_file(
        os.path.expanduser(os.path.abspath(ssh_config.key_file))
    ) if ssh_config.key_file else None

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy)
    client.connect(hostname=hostname, port=ssh_config.port, username=ssh_config.username, pkey=pkey)
    _ = client.exec_command('mkdir -p %s' % remote_path)
    client.close()

    t = paramiko.Transport((hostname, ssh_config.port))
    t.connect(username=ssh_config.username, pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(t)
    sftp.put(localpath=local_path, remotepath=os.path.join(remote_path, os.path.basename(local_path)))
    sftp.close()
    t.close()


def remote_pre_start_tf_server(working_dir, tf_server_starter_filepath, cluster_spec,
                               hostname, ssh_config):
    """
    Prepare to start a TensorFlow server remotely.

    Args:
        working_dir (str): remote working directory
        tf_server_starter_filepath (str): local starter file path
        cluster_spec (dict): TensorFlow ClusterSpec for servers
        hostname (str): host name or address
        username (str): host username
        key_file (str): ssh key file path
    """
    # TODO: handle all error messages to help user config the cluster
    remote_copy(
        local_path=tf_server_starter_filepath,
        remote_path=working_dir,
        hostname=hostname,
        ssh_config=ssh_config
    )
    remote_file_write(
        remote_path=os.path.join(working_dir, 'cluster_spec.json'),
        data=json.dumps(cluster_spec),
        hostname=hostname,
        ssh_config=ssh_config
    )
