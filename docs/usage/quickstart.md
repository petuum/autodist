# Get Started with AutoDist

AutoDist aims to help distribute local deep-learning training with minimal code change
and reduced attention on distributed semantics.

## The First Example

AutoDist developers are maximizing our effort to optimize the interface experience.
The changes needed to switch from simple local training to distributed training
using the current version of AutoDist are illustrated in the following example.

```python
import os
import tensorflow as tf

############################################################
# Change 1: Construct AutoDist with ResourceSpec
from autodist import AutoDist
resource_spec_file = os.path.join(os.path.dirname(__file__), 'resource_spec.yml')
d = AutoDist(resource_spec_file, 'PS')
#############################################################


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (_, _) = fashion_mnist.load_data()
train_images = train_images[:, :, :, None]
train_images = train_images / 255.0

BATCH_SIZE = 32
STEPS_PER_EPOCH = len(train_images) // BATCH_SIZE
EPOCHS = 1

#############################################################
# Change 2: Put Model under the Scope
with d.scope():
#############################################################

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).repeat(EPOCHS).shuffle(
        10000).batch(BATCH_SIZE)

    #############################################################
    # Change 3.1: Construct Graph-Mode Iterator
    # train_iterator = iter(train_dataset)  # original
    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    batch = train_iterator.get_next()
    #############################################################

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    #############################################################
    # Change 4: Mark the Training Step
    @d.function
    #############################################################
    def train_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            y_hat = model(x, training=True)
            loss = loss_fn(y, y_hat)
            grads = tape.gradient(loss, model.trainable_variables)
            #############################################################
            # Change 5: Return the Training Op
            # optimizer.apply_gradients(zip(grads, model.trainable_variables))  # original
            train_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            #############################################################
        return optimizer.iterations, loss, train_op

    for epoch in range(EPOCHS):
        for _ in range(STEPS_PER_EPOCH):
            #############################################################
            # Change 3.2: Use the Graph-Mode Iterator
            # batch = next(train_iterator)  # original
            #############################################################
            i, loss, _ = train_step(batch)
            print("step: {}, train_loss: {:5f}".format(int(i), loss))
```

#### Change 1: Construct AutoDist with ResourceSpec

This tells AutoDist the current specifications of resources (e.g. device topology),
so that AutoDist can optimize the distributed strategy based on it.

Here are some examples of the `.yml` file:
* To distribute across GPUs on one node:
```yaml
nodes:
  - address: localhost
    gpus: [0,1]
    chief: true
```
* To distribute across GPUs on multiple nodes:
```yaml
nodes:
  - address: 172.31.30.187
    gpus: [0,1,2,3]
    chief: true  # mark the current node where the program is launching
  - address: 172.31.18.140
    gpus: [0,1,2,3]

# When distributing across homogeneous nodes, 
# currently we needs to make sure in advance that 
# all nodes should already be able to SSH into each other.  
ssh:
  username: 'ubuntu'
  key_file: '/home/ubuntu/.ssh/autodist.pem'
  python_venv: 'source /home/ubuntu/venvs/autodist/bin/activate'
```


#### Change 2: Put the Model Under the Scope

It tells AutoDist what TensorFlow computations are essential in order to distribute.

Note that every TensorFlow op under the AutoDist scope will behave as if it is in graph mode 
(like Tensorflow 1.x), instead of eager mode (the default in Tensorflow 2.x).

#### Change 3: Dataset Iterator

* (Change 3.1) Construct the TensorFlow graph-mode dataset iterator and its getting-next handle op.
* (Change 3.2) Use the getting-next handle op directly for input.

#### Change 4: Mark the Training Step

This is the function that runs the model for training. 
Marking this function tells us what specifically to distribute.
 
#### Change 5: Return the Training Op

Under graph mode, it is required to pass the explicit training handle; while in eager, 
the training op is inferred with the support of implicit control dependencies.


## More Info

The changes above are all we need to distribute our first example.
 
* For more examples, you may refer to the 
[examples](https://gitlab.int.petuum.com/internal/scalable-ml/autodist/tree/master/examples) 
in our repository.
* For more detailed introduction, you may refer to the 
[tutorial](tutorial.md).
* AutoDist is still in the early stages of developement. We'd really appreciate any feedback! 
If you find any issues, please report them on JIRA under the `Symphony` project with `component=AutoDist`.   

Have fun with AutoDist!





