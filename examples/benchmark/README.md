## Benchmark Examples using AutoDist
```
# You can run autodist on these examples with different supported parallel strategy through setting AUTODIST_STRATEGY from PS, PSLoadBalancing, PartitionedPS, AllReduce, Parallax.
export AUTODIST_STRATEGY=PS
```

#### ImageNet Models (e.g. VGG16, Densenet121, Resnet101, Inceptionv3)
The instruction for generating the tfrecord data for ImageNet can be found following [this link](https://github.com/tensorflow/models/tree/master/official/vision/image_classification#legacy-tfrecords).
```
# You can set cnn models from vgg16, resnet101, densenet121, inceptionv3
export CNN_MODEL=resnet101
python ${REAL_SCRIPT_PATH}/imagenet.py --data_dir=${REAL_DATA_PATH}/train --train_epochs=10 --cnn_model=$CNN_MODEL --autodist_strategy=$AUTODIST_STRATEGY
# ${REAL_SCRIPT_PATH} and ${REAL_DATA_PATH} are the real paths you place the code and dataset
```

#### Bidirectional Encoder Representations from Transformers (BERT)
The instruction for generating the training data and setting up the pre-trained model with the config file can be found following [this link](https://github.com/tensorflow/models/tree/master/official/nlp/bert).
```
python ${REAL_SCRIPT_PATH}/bert.py -input_files=${REAL_DATA_PATH}/sample_data_tfrecord/*.tfrecord --bert_config_file=${REAL_DATA_PATH}/uncased_L-24_H-1024_A-16/bert_config --num_train_epochs=1 --learning_rate=5e-5 --steps_per_loop=20 --autodist_strategy=$AUTODIST_STRATEGY
```
##### Running BERT on Ray backend
Autodist can be used with Ray with the help of the RaySGD API. To start a ray cluster, first start the head node
```
ray start --head --port 6379 --include-dashboard=false
```
and subsequently attach any other nodes to the head node. The training job can then be started by running
```
python bert_ray.py --input_files=data/*.tfrecord --bert_config_file=bert_config.json
```
where `data/` has all the pretraining data and `bert_config.json` is the configuration file. This will submit the job to the local Ray cluster (`address='auto'`). Use the `--address` argument if you are targeting a different cluster. The `data/` has to be present on all the nodes of the cluster at the same path. The example supports all other arguments from the base implementation like `--autodist_strategy`.

Few caveats: During execution on some platforms the TensorFlow servers might complain about too many open files. You can get rid of the errors by setting a higher open file handle limit with  `ulimit -n 1064` on all nodes before starting the Ray cluster. 
To use a custom CUDA path, export it before starting the Ray cluster processes.

#### Neural Collaborative Filtering (NCF) 
The instruction for generating the training data can be found following [this link](https://github.com/tensorflow/models/tree/master/official/recommendation).
```
python ${REAL_SCRIPT_PATH}/ncf.py --default_data_dir=${REAL_DATA_PATH}/movielens --autodist_strategy=$AUTODIST_STRATEGY
```

