
# Train on Multiple Nodes


### Resource Specification

In tutorial [Getting Started](getting-started.md),
there is an example using the minimal resource specification in `yaml` format:

```yaml
nodes:
  - address: localhost
    gpus: [0,1]
```

However, we often need to train a large model with multiple nodes / virtual machines / pods etc.,
and at the same time with multiple accelerators.
Here is an example to define a resource specification for multi-node training:  


```yaml
nodes:
  - address: 172.31.30.187
    cpus: [0]
    gpus: [0,1,2,3]
    chief: true
  - address: 172.31.18.140
    gpus: [0,1,2,3]
    ssh_config: ssh_conf1
  - address: 172.31.18.65
    cpus: [0,1]
    gpus: [0,1]
    ssh_config: ssh_conf2

ssh:
  ssh_conf1:
    key_file: '/home/ubuntu/.ssh/autodist.pem'
    username: 'ubuntu'
    python_venv: 'source /home/ubuntu/venvs/autodist/bin/activate'
  ssh_conf2:
    key_file: '/home/ubuntu/.ssh/my_private_key'
    username: 'username'
    python_venv: 'source /home/username/anaconda3/etc/profile.d/conda.sh;conda activate autodist'
    shared_envs:
      LD_LIBRARY_PATH: '$LD_LIBRARY_PATH:/usr/local/cuda/lib64'
```

* `nodes`: the computational nodes spec defined for distributed training.
    * `address` *str*: the ipv4 or ipv6 address to access the nodes.
        Note that in a multiple-node scenario the resource specification is shared across the node,
        and the address will be the identifier for communications across nodes in the cluster,
        so one should avoid using `localhost` in such case to prevent ambiguity.
    * `cpus` *List\[int\]* (optional): If not specified, it means only the first CPU will be utilized.
        One can also specify multiple CPUs. However, it requires some careful treatment.
        For example, if a strategy uses the extra CPU for data-parallel computation together with GPU,
        it can slow down the performance. Even though in the future AutoDist will support
        auto-generated strategies to utilize the extra CPUs smartly, the multiple CPUs strategy is still experimental in the current releases.
    * `gpus` *List\[int\]* (optional): the CUDA device IDs of GPUs.
    * `chief` *bool* (optional): Optional for single-node training.
    However, it is required to specify exactly one of the multiple nodes to be the `chief`.
    The `chief` node should be the node you launch the program from (it will also participate in the training, so make sure it is a node in your cluster).

* `ssh`:  You also need to create `ssh_config`s that can be used by the `chief` node to ssh into all other nodes.
They must be named; this allows for having different ssh configs if, for example, different nodes have different private keys.
`ssh_conf1`, `ssh_conf2` can be any unique string to specify a group of configurations.
    * `key_file` *str*: key path on **`chief` node** to use to connect the remote node via ssh
    * `username` *str*: username to access the remote (non-chief) node with
    * `python_venv` *str* (optional): the command to activate the virtual environment on a remote (non-chief) node.
    If not provided, it will use the default system default `python`.
    * `shared_envs` *pair* (optional): the key-value environment variable pairs for a remote (non-chief) node to use

> In the current release, the multi-node coordination is based on SSH and Python virtual environments;
while other types of coordination are being actively developed.
For example to support for docker container instead of Python virtual environment, or
for K8 launching instead of SSH, etc.


### Environment Preparation

Below are the steps of setting up environments for multi-node training.
If you are familiar with Docker, you could skip the following and 
follow this straight-forward [instruction](docker.md) to directly launch with Docker.

Before running your program distributed with AutoDist on multiple nodes,
1. The model should be able to train on each node successfully with AutoDist, as in
[Getting Started](getting-started.md).
2. The working directory containing project files should share the same absolute path on all nodes.
For example, if the working dir is `/home/username/myproject` on the node where you launch the program,
it is also required that your project files are placed under the same absolute path
for all the other nodes. Symbolic link is also allowed.
3. Note that the field `python_venv` requires the command to source remote env, *without* sourcing `~/.bash_rc` unless specified; while `shared_envs` is for passing the environment vars to the remote. For conda environments, `conda.sh` must be sourced before conda environment can be activated.

Then one can launch the script from the chief machine. 
