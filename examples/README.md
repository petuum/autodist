

For experiments,

* Run locally
```bash
python -m autodist.experimental.cluster_manager -f examples/cluster_spec.json # check if the hosts are localhost in json
python examples/multiworker/linear_regression.py
```

* Run remotely
```bash
python -m autodist.experimental.cluster_manager -f examples/cluster_spec.json -m remote -i ~/.ssh/autodist.pem
python examples/multiworker/linear_regression.py
```

For more configuration details, please check https://gitlab.int.petuum.com/internal/scalable-ml/autodist/blob/master/autodist/experimental/README.md