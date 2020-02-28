
# Train on Multi Nodes


### Recap: Minimal Resource Specification
In tutorial <[Get Started](get-started.md)>, 
there is an example using the minimal resource specification in `yaml` format:

```yaml
nodes:
  - address: localhost
    gpus: [0,1]
```

### Full Format of Resource Specification
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
      python_venv: 'source /home/username/venvs/autodist/bin/activate'
```

* `nodes`: the computational nodes spec defined for distributed training.
    * `address` *str*: the ipv4 or ipv6 address to access the node. 
        Note that in multiple-node scenario the resource specification is shared across the node,
        and the address will be the identifier for communications across nodes in the cluster,
        so one should avoid using `localhost` in such case to prevent ambiguity.
    * `cpus` *List\[int\]* (optional): If not specified, it means only the first CPU will be utilized. 
        One can also specify multiple CPUs. However, it requires some careful treatment.
        For example, if a strategy use the extra CPU for data-parallel computation together with GPU,
        it can slow down the performance. Even though in the future AutoDist will support
        auto-generated strategy to utilize the extra CPUs smartly, the multiple CPUs is still
        experimental in the current releases.
    * `gpus` *List\[int\]* (optional): the CUDA device IDs of GPUs.
    * `chief` *bool* (optional): Optional for single-node training.
    However, it is required to specify exactly one of multiple nodes to be the `chief`. 
    The `chief` node should be the node you launch the program from (it will also participate in the training, so make sure it is a node in your cluster).

* `ssh`:  You also need to create `ssh_config`s that can be used by the `chief` node to ssh into all other nodes. 
They must be named; this allows for having different ssh configs if, for example, different nodes have different private keys.
`ssh_conf1`, `ssh_conf2` can be any unique string to specify a group of configurations.
    * `key_file` *str*: key path on `chief` node to use to connect the remote node via ssh
    * `username` *str*: username to access the remote node with
    * `python_venv` *str*: the command to activate the virtual environment on the remote node

### Future Support

In the current release, the multi-node coordination is based on SSH and Python virtual environements;
while other types of coordination is being actively developed. 
For example to support for docker container instead of Python virtual envrionement, or 
for K8 launching instead of SSH, etc.

