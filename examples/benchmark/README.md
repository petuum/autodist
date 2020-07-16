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

#### Neural Collaborative Filtering (NCF) 
The instruction for generating the training data can be found following [this link](https://github.com/tensorflow/models/tree/master/official/recommendation).
```
python ${REAL_SCRIPT_PATH}/ncf.py --default_data_dir=${REAL_DATA_PATH}/movielens --autodist_strategy=$AUTODIST_STRATEGY
```

