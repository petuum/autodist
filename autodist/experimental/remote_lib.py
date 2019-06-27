import os
import subprocess
import warnings
import paramiko

warnings.filterwarnings(action='ignore', module=paramiko.__name__)


def remote_exec(bash_script,
                hostname,
                username,
                port=22,
                key_file=None,
                env=None,
                python_venv=None):
    full_cmd = ''
    if python_venv is not None:
        full_cmd += '%s;' % python_venv
    if env is not None:
        full_cmd += ' '.join(['%s=%s' % (k, v) for k, v in env.items()])
    full_cmd += ' ' + bash_script

    remote_cmd = 'ssh -i %s -tt -p %d %s@%s \'bash -c "%s"\' </dev/null' % (
        key_file, port, username, hostname, full_cmd
    )

    def colored(msg):
        return '\033[31m' + msg + '\033[0m'

    print(colored('\n$ %s' % remote_cmd))

    proc = subprocess.Popen(remote_cmd, shell=True)
    return proc


def remote_file_write(remote_path, data, hostname, username, port=22, key_file=None):
    pkey = paramiko.RSAKey.from_private_key_file(key_file) if key_file else None
    t = paramiko.Transport((hostname, port))
    t.connect(username=username, pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(t)
    with sftp.open(remote_path, 'w') as f:
        f.write(data)
    sftp.close()
    t.close()


def remote_copy(local_path, remote_path, hostname, username, port=22, key_file=None):
    pkey = paramiko.RSAKey.from_private_key_file(key_file) if key_file else None

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy)
    client.connect(hostname=hostname, port=port, username=username, pkey=pkey)
    stdin, stdout, stderr = client.exec_command('mkdir -p %s' % remote_path)
    client.close()

    t = paramiko.Transport((hostname, port))
    t.connect(username=username, pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(t)
    sftp.put(localpath=local_path, remotepath=os.path.join(remote_path, os.path.basename(local_path)))
    sftp.close()
    t.close()


def remote_pre_start_tf_server(working_dir, launcher_script_filename, cluster_spec_filepath,
                               hostname, username, key_file=None):
    remote_copy(
        local_path=os.path.join(os.path.dirname(__file__), launcher_script_filename),
        remote_path=working_dir,
        hostname=hostname,
        username=username,
        key_file=key_file
    )
    remote_file_write(
        remote_path=os.path.join(working_dir, os.path.basename(cluster_spec_filepath)),
        data=open(cluster_spec_filepath).read(),
        hostname=hostname,
        username=username,
        key_file=key_file
    )
