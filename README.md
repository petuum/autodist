# AutoDist: Automated Distributed Deep Learning

[![pipeline status](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/badges/master/pipeline.svg)](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/commits/master)
[![coverage report](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/badges/master/coverage.svg)](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/commits/master)

[Documentation](http://10.20.41.55:8080) |
[Examples](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/tree/master/examples) |
[Releases](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/tags)

**AutoDist** is Petuum's new scalable ML engine. 
AutoDist provides user-friendly interfaces to distribute local deep-learning model training across multiple processing units (for example, distributed GPU clusters) with scalability and minimal code changes. 


AutoDist currently supports [TensorFlow 2.0](https://www.tensorflow.org/beta/).


## Installation

#### Install from released binaries 

Download the latest autodist wheel file from [Petuum PyPI](http://pypi.int.petuum.com:8080/#/package/autodist).
```bash
pip install <path/to/wheel.whl>
```


#### Install from latest source and develop locally

```bash
git clone https://gitlab.int.petuum.com/internal/scalable-ml/autodist.git
cd autodist
pip install -e .
```

## Issue Report

AutoDist is still in the early stages of developement. We'd really appreciate any feedback! 
If you find any issues, please report them on JIRA under the `Symphony` project with `component=AutoDist`.   


## Reference & Acknowledgement

We learned and borrowed insights from a few open source projects:

- [tf.distribute.strategy](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/distribute)
- [Parallax](https://github.com/snuspl/parallax)