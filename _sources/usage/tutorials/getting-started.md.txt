
# Getting Started

Thanks for choosing to use AutoDist!

Before reading the following tutorial, it is highly recommended to get familiar with the [TensorFlow Quickstart Guide](https://www.tensorflow.org/tutorials/quickstart/advanced) first, 
and especially understand the difference between eager and graph mode. If you can run the [Quickstart](https://www.tensorflow.org/tutorials/quickstart/advanced) properly, you can use the same environment to follow this tutorial.

AutoDist currently supports `Python>=3.6` with `tensorflow>=1.15, <=2.1`. Install the downloaded wheel file by

```bash
pip install autodist
``` 

The following model is based on the [Quickstart](https://www.tensorflow.org/tutorials/quickstart/advanced). Note that it is a model written for one node and one GPU. If we want to train it on multiple GPUs with AutoDist, we could follow these next 3 steps:

### Step 1: Ensure Model Built under Graph Mode

AutoDist currently is only expected to work with TensorFlow graph mode, instead of eager.
It is natural for TensorFlow 1.x; while with TensorFlow 2.x, 
one needs to put their code under `tf.Graph().as_default()` to ensure the [graph mode](https://www.tensorflow.org/api_docs/python/tf/Graph).
For example,

```python
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g
```

Before using AutoDist, the model built without AutoDist should be able to train with graph mode successfully.
Then one can move to the following steps in the same environment. 

### Step 2: Prepare Resource Specification File

AutoDist needs to know what devices are available in order to distribute the computational graph on them. 
Currently the supported devices can be `cpus` or `gpus`.
Let's create a file called `resource_spec.yml`:

```yaml
nodes:
  - address: localhost
    gpus: [0,1]  # List the GPU (CUDA_DEVICE) ids
```

For other resource cases (e.g., multiple nodes), please refer to the [next tutorial](multi-node.md).

### Step 3: Add AutoDist APIs

Given TensorFlow code (either TF1.x or TF2.x) for training a model in graph mode, 
we can easily modify it to train in a distributed fashion. 
Based on the <code>[AutoDist](../../api/autodist.autodist)</code> interfaces, 
all we have to do is make the following 3 changes (marked by inline comments):

```python
import numpy as np
import tensorflow as tf

#########################################################################
# Change 1: Construct AutoDist with ResourceSpec.
#   Pass the absolute path of the file created in Step 1 above to AutoDist.
#
import os
from autodist import AutoDist
filepath = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')
autodist = AutoDist(resource_spec_file=filepath)
#########################################################################

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images[:512, :, :, None]
test_images = test_images[:512, :, :, None]
train_labels = train_labels[:512]
test_labels = test_labels[:512]
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

BATCH_SIZE = 64
EPOCHS = 1

##########################################################################
# Change 2: Build with Graph mode, and put it under AutoDist scope.
#   Note that, for both TensorFlow v1 and v2,
#   AutoDist currently is only expected to work with graph mode, not eager.
#
# [original line]
# >>> with tf.Graph().as_default():
#
with tf.Graph().as_default(), autodist.scope():
##########################################################################

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).repeat(EPOCHS).shuffle(len(train_images)//2).batch(BATCH_SIZE)

    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD()

    def train_step(inputs):
        x, y = inputs
        y_hat = model(x, training=True)
        loss = loss_fn(y, y_hat)
        all_vars = []
        for v in model.trainable_variables:
            all_vars.append(v)
        grads = tf.gradients(loss, all_vars)
        update = optimizer.apply_gradients(zip(grads, all_vars))

        return loss, update

    fetches = train_step(train_iterator)
    #####################################################################
    # Change 3: Create distributed session.
    #   Instead of using the original TensorFlow session for graph execution,
    #   let's use AutoDist's distributed session, in which a computational
    #   graph for distributed training is constructed.
    #
    # [original line]
    # >>> sess = tf.compat.v1.Session()
    #
    sess = autodist.create_distributed_session()
    #####################################################################
    for _ in range(min(10, len(train_images) // BATCH_SIZE * EPOCHS)):
        loss, _ = sess.run(fetches)
        print(f"train_loss: {loss}")
```

And just like that, our single-GPU code can run on multiple GPUs.

Thanks for reading, and have fun with AutoDist!
