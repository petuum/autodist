The experimental nodes launcher.

Prerequisite:
* TensorFlow is already installed in the env of all nodes.
* Only support in graph launching logic. Only one node `NODE 0` runs the session client.
* AutoDist is already installed in the env of the worker node `NODE 0`, where the main script runs.
* The open ssh private key to other nodes is accessible on `NODE 0` given a path. 
All other nodes are added in the `known_host` of `NODE 0`. 


On `NODE 0`, in the env:

* Launch TF cluster
```bash
python -m autodist.experimental.cluster_manager -m remote -i ~/.ssh/autodist.pem -f ./cluster_spec.json
```

* Run script
```bash
python examples/multiworker/linear_regression.py
```


