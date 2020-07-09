# Orchestra Integration

## What is Orchestra?

[Orchestra](https://petuum.com/platform/) is a machine learning platform built by Petuum Inc. which includes data management, job control, etc. AutoDist will be the distributed training backend of this platform.

## AutoDist in Orchestra

### Data Directory Path

In Orchestra, data will be managed automatically, and sometimes AutoDist users can not get the data path until the runtime. In this case, Orchestra will provide the data directory path through an environment variable `SYS_DATA_PATH` to all AutoDist worker nodes. Here is an example:

```
data_path = os.environ.get("SYS_DATA_PATH")
```

### Resource File Path

Orchestra is built on virtualization techniques, so there is no static network configuration for AutoDist. To match the current AutoDist coordination mechanism, Orchestra will generate a temporary resource spec used by AutoDist, which the file path is also given by an environment path `SYS_RESOURCE_PATH`.

```
resource_spec_file = os.environ.get("SYS_RESOURCE_PATH")
autodist = AutoDist(resource_spec_file, AllReduce(128))
```

## Example Code

This is an example AutoDist code that can be executed on Orchestra.

```
import os

import tensorflow as tf

from autodist import AutoDist
from autodist.strategy.all_reduce_strategy import AllReduce

# A customized loader provided by user
from data_loader import load_data


resource_spec_file = os.environ.get("SYS_RESOURCE_PATH")
autodist_data_path = os.environ.get("SYS_DATA_PATH")

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
EPOCHS = 10
inputs, noises = load_data(autodist_data_path)
outputs = inputs * TRUE_W + TRUE_b + noises


def train():
    """Train the model and save the serialized model and its weights."""
    autodist = AutoDist(resource_spec_file, AllReduce(128))

    with tf.Graph().as_default(), autodist.scope():
        x = tf.compat.v1.placeholder(shape=[NUM_EXAMPLES], dtype=tf.float64)
        W = tf.Variable(5.0, name='W', dtype=tf.float64)    # nopep8
        b = tf.Variable(0.0, name='b', dtype=tf.float64)

        def y():
            return W * x + b

        def loss_func(predicted_y, desired_y):
            return tf.reduce_mean(tf.square(predicted_y - desired_y))

        major_version, _, _ = tf.version.VERSION.split('.')
        if major_version == '1':
            optimizer = tf.train.GradientDescentOptimizer(0.01)
        else:
            optimizer = tf.optimizers.SGD(0.01)

        with tf.GradientTape():
            prediction = y()
            loss = loss_func(prediction, outputs)
            vs = [W, b]
            gradients = tf.gradients(loss, vs)
            train_op = optimizer.apply_gradients(zip(gradients, vs))

        fetches = [loss, train_op, b, prediction]
        feeds = [x]

        session = autodist.create_distributed_session()
        for _ in range(EPOCHS):
            train_loss, _, b, _ = session.run(fetches, feed_dict={feeds[0]: inputs})
            print('loss: {}'.format(train_loss)

if __name__ == "__main__":
    train()
```
