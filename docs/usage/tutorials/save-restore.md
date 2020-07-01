# Save and Restore Training

Users need to use the AutoDist internal wrappers for saving. These wrappers will save the **original graph** instead of the transformed distributed graph executed during the training. In this case, users can further fine-tune the model without AutoDist or under other distributed settings.

There are 2 AutoDist saving APIs, which are very similar to the native TensorFlow saving interface. One is `saver` another is `SavedModelBuilder`.

## saver

AutoDist provides a `saver`, which is a wrapper of the original Tensorflow Saver. It has the exactly same interface as the one in Tensorflow. **Notice, this saver should be created before the `create_distributed_session` function.** Here is an example:

```
from autodist.checkpoint.saver import Saver as autodist_saver
...


# Build your model
model = get_your_model()

# create the AutoDist Saver
saver = autodist_saver()

# create the AutoDist session
sess = autodist.create_distributed_session()
for steps in steps_to_train:
    # some training steps
    ...

# Save the training result
saver.save(sess, checkpoint_name, global_step=step)
```

More detailed usage, including the model restore, can be found [Here](../../../tests/checkpoint/test_keras_saver.py).

## SavedModelBuilder

The `SavedModelBuilder` API will not only save the trainable variables, but also some other training metadata, such as the Tensorflow MetaGraph and training operations. Like the `saver`, `SavedModelBuilder` also has the same interface as the original one in the Tensorflow. However, instead of using the default saver, users needs use the AutoDist saver to initialize it. Here is an example:

```
# create the AutoDist Saver
saver = autodist_saver()

# create the AudoDist session
sess = autodist.create_distributed_session()
for steps in steps_to_train:
    # some training steps
    ...

builder = SavedModelBuilder(EXPORT_DIR)
builder.add_meta_graph_and_variables(
    sess=sess,
    tags=[TAG_NAME],
    saver=saver,
    signature_def_map=signature_map)
builder.save()
```

More detailed usage, including the model fine-tuning, can be found [Here](../../../tests/checkpoint/test_saved_model.py).
