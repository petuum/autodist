from arion.simulator.simulator import Simulator
from arion.strategy import base
from arion.graph_item import GraphItem

resource_spec_file = '/home/hao.zhang/project/pycharm/ncf-trial/official/recommendation/trial/trialrun_resource_specs/resource_spec_2.yml'
strategy_path = '/home/hao.zhang/oceanus_simulator/ncf_3/strategies/20200505T174311M650364'
original_graph_item_path = '/home/hao.zhang/oceanus_simulator/ncf/strategies/original_graph_item'

s = base.Strategy.deserialize(strategy_path)


simulator = Simulator(resource_file=resource_spec_file,
                      original_graph_item_path=original_graph_item_path)

ret = simulator.simulate(s)

print('finished')
