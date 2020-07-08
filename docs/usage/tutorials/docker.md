# Train with Docker 

To facilitate the installation process on GPU machines, the AutoDist team has published the reference Dockerfile so you can get started with AutoDist in minutes.

## Building

First clone the AutoDist repository.

```bash
git clone https://github.com/petuum/autodist.git
```

Once we cloned the repository successfully we can build the Docker image with the provided Dockerfile.

```bash
cd autodist
docker build -t autodist:latest -f docker/Dockerfile.gpu .
```

## Running on a single machine

Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the built container with GPU access. `<WORK_DIR>` is the directory where your python script is located. In this example, we are using `autodist/examples/` as our `<WORK_DIR>`.

```bash
docker run --gpus all -it -v <WORK_DIR>:/mnt autodist:latest
```

Inside the docker environment, you will be able to run the examples using the machine's GPUs. Remember to follow the "[Getting Started](getting-started.md)" tutorial to
properly set up your `<WORK_DIR>/resource_spec.yml`

```bash
python /mnt/linear_regression.py
```

## Multiple Machine Setup

In this section, we will describe a way to set up passwordless SSH between docker containers in different machines. If your machines are already set up, you can go directly to the [Running on multiple machines](#Running-on-multiple-machines) section.

This section describes a way to enable passwordless authentification between multiple docker environments in different machines. The idea is to create a shared private key across all containers and to allow this shared private key to access all machines.

Create the `SHARE_DIR` to hold the shared credentials

```bash
mkdir <SHARE_DIR> && mkdir <SHARE_DIR>/.ssh
```

Create the credentials in the created folder

```bash
ssh-keygen -f <SHARE_DIR>/.ssh/id_rsa
```

Copy the created public key into the `<SHARE_DIR>/.ssh/authorized_keys` file

```bash
cat <SHARE_DIR>/.ssh/id_rsa.pub | cat >> <SHARE_DIR>/.ssh/authorized_keys
```

Setup the `<SHARE_DIR>/.ssh` directory with the correct ownership and permissions for SSH. 

**Note:** you might need to use the `sudo` command if permission is denied.

```bash
chown -R root <SHARE_DIR>/.ssh
chmod 700 <SHARE_DIR>/.ssh
chmod 600 <SHARE_DIR>/.ssh/authorized_keys
```

Once you have set up the credential folder, you must make sure all your machines have access to this folder or a copy of this folder. You also need to make sure that the ownership and permission of the files do not change. You can use `rsync` or `scp` to share this credential folder across all machines. An example of the command to do this is:

```bash
sudo rsync -av --rsync-path "sudo rsync" <SHARE_DIR>/.ssh/ user@remote:<SHARE_DIR>/.ssh
```

## Running on multiple machines

Once you have set up the passwordless SSH, you need to configure the `<WORK_DIR>/resource_spec.yml` using the "[Getting Started](getting-started.md)" and "[Train on Multiple Nodes](multi-node.md)" with all worker machine's port set to the number `<PORT_NUM>`.

This is an example of `resource_spec.yml` file for multiple machine setup with `12345` as the `<PORT_NUM>`

```yaml
nodes:
  # multi-nodes docker experiment
  - address: 10.20.41.126
    gpus: [0,1]
    chief: true
  - address: 10.20.41.114
    gpus: [0,1]
    ssh_config: conf
  - address: 10.20.41.128
    gpus: [0,1]
    ssh_config: conf
ssh:
  conf:
    username: 'root'
    key_file: '/root/.ssh/id_rsa' # shared credential file
    port: 12345
```

**Note**: the `resource_spec.yml` file must be inside the `<WORK_DIR>` directory as it is the directory that will be mounted inside the docker image.

Before running the multi-machine training job, you must make sure all contents in `<WORK_DIR>` in all node machines are the same. In this example, we are using `autodist/examples/` as our `<WORK_DIR>`.

Then on every single autodist worker machine run

```bash
docker run --gpus all -it --privileged -v <SHARE_DIR>/.ssh:/root/.ssh -v <WORK_DIR>:/mnt --network=host autodist:latest bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
```

And on the autodist chief machine run

```bash
docker run -it --gpus all --network=host -v <SHARE_DIR>/.ssh:/root/.ssh:ro -v <WORK_DIR>:/mnt autodist:latest
```

And inside the autodist chief's docker environment run

```bash
python /mnt/linear_regression.py
```
