
This folder contains helper information about how to access and use the public dataset released with the paper: 
[AutoSync: Learning to Synchronize for Data-Parallel Distributed Deep Learning](https://papers.nips.cc/paper/2020/hash/0a2298a72858d90d5c4b4fee954b6896-Abstract.html).


## Download Data 
Download the data from this [Google Drive Link](https://drive.google.com/file/d/18iEdIP5NncvxrIu1HaKCooRVYFcMoZH-/view?usp=sharing).

The dataset is organized as follows: the outermost folder corresponds to ML models. 
For an ML model, we provide a ``graph_item`` file which could be read via AutoDist API, 
and contains the serialized ``tf.Graph`` object of the original model (before distributed graph transformation). 
The data is then put in subfolders that correspond to the cluster setup (i.e., in-house cluster-A and AWS cluster-B) where we collected the data (see the paper for details). 
Each data sample then contains a <`resource_spec`, `runtime`, `strategy`> tuple, which are all uniquely named by an ``ID``. 
See the AutoSync paper for more details how to use the tuple to train strategy simulators.


Below is a visualization of the folder structure:

    Model-1/ (e.g., BERT-large)
        graph_item
        cluster-A/ (e.g., In-house cluster)
            resource_spec/
                <ID>
            runtime/
                <ID>
            strategies/
                <ID>  
        cluster-B/ (e.g., AWS cluster)
            resource_spec/
                <ID>
            runtime/
                <ID>.yml
            strategies/
                <ID>  
    Model-2 
        ......

    Model-3 
        ...... 


## Read an AutoDist Strategy

To read a distributed strategy file ``strategy_file`` in the dataset, use the autodist strategy APIs as follows

```python
from autodist.strategy import base
strategy = base.Strategy.deserialize(strategy_file)
```
where ``strategy_file`` is 


## Inspect the Runtime
Each runtime file contains the per-iteration runtime (in seconds) of the ``<model, strategy, resource_spec>`` we collected through many
trial runs, and an average runtime of all trials. To read a ``runtime_file``:
```python
import yaml

with open(runtime_file) as file:
    runtime_dict = yaml.load(file, Loader=yaml.FullLoader)

    # now you can inspect the runtime such as
    print(runtime_dict["runtime"])
    print(runtime_dict["average"])
```

## Read the Resource Spec
The resource spec contains the setup information of the cluster in YAML. To read a ``resource_spec_fle``:
```python
from autodist.resource_spec import ResourceSpec
rs = ResourceSpec(resource_spec_file)
```
See the class ``autodist.resource_spec`` for more information on how to inspect the cluster specification.

## Access the GraphItem Object
A graph item object is a serialized binary that contains the graph definiation of the original model. It is an intermediate representation (IR)
used in AutoDist to perform graph rewriting and distribution. See the class ``autodist.graph_item`` for the full implementation of this IR. 
To read the graph item object at path ``graph_item_path``:
```python
from autodist.graph_item import GraphItem
graph_item = GraphItem.deserialize(graph_item_path)
```

## How to cite
If you find the dataset useful for your development or research, please consider citing the following paper:
```
@article{zhang2020autosync,
title={AutoSync: Learning to Synchronize for Data-Parallel Distributed Deep Learning},
author={Zhang, Hao and Li, Yuan and Deng, Zhijie and Liang, Xiaodan and Carin, Lawrence and Xing, Eric},
journal={Advances in Neural Information Processing Systems},
volume={33},
year={2020}
}
```
