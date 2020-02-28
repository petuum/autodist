
# Multi-Node Resources

* To distribute across GPUs on multiple nodes:

```yaml
nodes:
  - address: 172.31.30.187
    gpus: [0,1,2,3]
    chief: true  # mark the current node where the program is launching
  - address: 172.31.18.140
    gpus: [0,1,2,3]
    ssh_config: ssh_conf1

ssh:
  ssh_conf1:
      username: 'ubuntu'
      key_file: '/home/ubuntu/.ssh/autodist.pem'
      python_venv: 'source /home/ubuntu/venvs/autodist/bin/activate'
```

Note that:

1) When using remote nodes, you must specify exactly one of them to be the `chief` node. The `chief` node should be the node you launch the program from (it will also participate in the training, so make sure it is a node in your cluster).
2) You also need to create `ssh_config`s that can be used by the `chief` node to ssh into all other nodes. They must be named; this allows for having different ssh configs if, for example, different nodes have different private keys.

# Switch Between Strategy Builders

# Various API supports
