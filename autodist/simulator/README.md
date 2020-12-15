
The ``simulator`` folder implements predefined simulator in AutoSync proposed in: [AutoSync: Learning to Synchronize for Data-Parallel Distributed Deep Learning](https://papers.nips.cc/paper/2020/hash/0a2298a72858d90d5c4b4fee954b6896-Abstract.html).

## Download Data 
Download the data from https://drive.google.com/file/d/1CTtIVORxzF_wOmxrsusbAhNC3bwmxuD8/view?usp=sharing.

The data folder is organized by ML model categories. For a ML model, the simulation is conducted on two kinds of clusters (AWS and an in-house cluster). Each data sample comprises a <resource specification, runtime, strategy> pair. The resource specification file corresponds to all runtimes and strategies inside runtimes and strategies folders, respectively. The detailed data organization is:  

    Model-1/ (e.g., BERT-large)
        Cluster-1/ (e.g., AWS-4-g4)
            resource_spec.yml
            runtime/
                <ID>.yml
            strategies/
                <ID>  
        Cluster-2 (e.g., In-house-11-nodes)
            resource_spec.yml
            runtime/
                <ID>.yml
            strategies/
                <ID>  
    Model-2 
        ......

    Model-3 
        ...... 
 

## Train a predefined simulator
 
Inside ``autodist/simulator`` folder. 
 
Define configuration in ``config.py`` including the model to simulate and the data folders (samples) to use. 
 
Run: ``python absolute_dir/train_predefined_simulator_clean.py`` 


## Simulate (infer) a strategy 

Inside ``autodist/simulator`` folder. 

Define the strategy to simulate and checkpoint to load in ``simulate.py``. 

Run: ``python simulate.py``. 


## Read a strategy 

Use ``strategy = base.Strategy.deserialize(strategy_file)`` to read a strategy stored as ``strategy_file``. 

