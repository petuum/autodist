# AutoDist:  Easy and Composable Distributed Deep Learning

[![pipeline status](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/badges/master/pipeline.svg)](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/commits/master)
[![coverage report](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/badges/master/coverage.svg)](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/commits/master)

[Documentation](http://10.20.41.55:8080) |
[Examples](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/tree/master/examples) |
[Releases](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/releases)

**AutoDist** is a distributed deep-learning training engine. 
AutoDist provides a user-friendly interface to distribute the training of a wide variety of deep learning models 
across many GPUs with scalability and minimal code change.

AutoDist has been tested with TensorFlow versions 1.15 through 2.1. 

## Introduction
Different from specialized distributed ML systems, AutoDist is created to speed up a broad range of DL models with excellent all-around performance.
AutoDist achieves this goal by:
- **Compilation**: AutoDist expresses the parallelization of DL models as a standardized compilation process, optimizing multiple dimensions of ML 
parallelization ranging from synchronization, model partitioning, placement to consistency. 
- **Composable architecture**: AutoDist designs a flexible backend that encapsulates various different ML parallelization techniques, and 
allows for composing distribution strategies that interpolates different distributed ML system architectures.     
- **Model and resource awareness**: Based on the compilation process, AutoDist analyzes the model and generates more optimal distribution strategies that 
adapt to both the ML properties and the cluster specification.

Besides all these advanced features, AutoDist is cautiously designed to isolate the sophistication of distributed systems 
from ML prototyping, and exposes a simple API that makes it easy to use and switch between different distributed ML techniques 
for all-level users.


## Installation

#### Install From Release Wheel 

```bash
pip install --extra-index-url http://pypi.int.petuum.com:8080/simple --trusted-host pypi.int.petuum.com autodist
```

#### Install From Latest Source

Before running AutoDist, we require a small compilation of our Protocol Buffers. 
To do so, you must first have [protoc installed](https://google.github.io/proto-lens/installing-protoc.html).

Then, you can run the following command :
```bash
git clone https://gitlab.int.petuum.com/internal/scalable-ml/autodist.git
cd autodist
PROTOC=`which protoc` python setup.py build  # compile our protobufs
pip install -e .  # install in development mode
```

To clean up any compiled files, run:
```bash
python setup.py clean --all
```

## Using AutoDist

It should be incredibly easy to modify existing TensorFlow code to use AutoDist.

```python
import tensorflow as tf
from autodist import AutoDist  # Import AutoDist

autodist = AutoDist(resource_spec_file="resource_spec.yml")  # Config AutoDist

with tf.Graph().as_default(), autodist.scope():  # Build under AutoDist
    # ... BUILD YOUR_MODEL ...
    sess = autodist.create_distributed_session()  # Run with AutoDist
    sess.run(YOUR_MODEL.fetches)
```

## References & Acknowledgements

We learned and borrowed insights from a few open source projects:

- [tf.distribute.strategy](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/distribute)
- [Horovod](https://github.com/horovod/horovod)
- [Parallax](https://github.com/snuspl/parallax)
