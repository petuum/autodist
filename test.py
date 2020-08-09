import tensorflow as tf
import autodist

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