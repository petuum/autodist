import glob

from tensorflow.python.eager import context

from models.predefined_simulator import PredefinedSimulator
from autodist.strategy import base
from autodist.resource_spec import ResourceSpec

from pathlib import Path

GRAPH_ITEM_DIR = f'{str(Path.home())}/graph_items'
SIMULATION_DATA_DIR = f'{str(Path.home())}/autosync_dataset_release'
CHECKPOINT_DIR =  f'{str(Path.home())}'


resource_spec_file = f'{SIMULATION_DATA_DIR}/cluster1/bert12l_aws4_from_bert3l_aws4_2/resource_spec.yml'
original_graph_item_path = f'{GRAPH_ITEM_DIR}/bert_original_graph_item_large'
checkpoint_path = f'{CHECKPOINT_DIR}/bert_predefined_checkpoints/ckpV1_bert_orca_100_0.67000_0.50000'
strategy_dir = f'{SIMULATION_DATA_DIR}/cluster1/bert12l_aws4_from_bert3l_aws4_2/strategies'
strategy_files = glob.glob(f'{strategy_dir}/*')
strategy_file = strategy_files[0]


with context.graph_mode():

    strategy = base.Strategy.deserialize(strategy_file)

    simulator = PredefinedSimulator(original_graph_item_path=original_graph_item_path)

    cost = simulator.simulate(strategy=strategy, resource_spec=ResourceSpec(resource_spec_file), checkpoint=checkpoint_path)

    print(f"strategy_file: {strategy_file}, cost: {cost}")


print('finished')
