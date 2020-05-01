import argparse
import importlib

from autodist import AutoDist
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.strategy.parallax_strategy import Parallax
from autodist.strategy.partitioned_ps_strategy import PartitionedPS
from autodist.strategy.ps_lb_strategy import PSLoadBalancing
from autodist.strategy.ps_strategy import PS
from autodist.strategy.partitioned_all_reduce_strategy import PartitionedAR
from autodist.strategy.uneven_partition_ps_strategy import UnevenPartitionedPS
from autodist.strategy.random_axis_partition_all_reduce_strategy import RandomAxisPartitionAR

STRATEGIES_FOR_DISTRIBUTED_TESTS = {
    'PS': PS(sync=True),
    'PS_stale_3': PS(sync=True, staleness=3),
    'PartitionedPS': PartitionedPS(),
    'PartitionedPS_stale_3': PartitionedPS(staleness=3),
    'AllReduce': AllReduce(chunk_size=1),
    'AllReduce_2': AllReduce(chunk_size=2),
    'Parallax': Parallax(),
    'PSLoadBalancingProxy_stale_3': PSLoadBalancing(local_proxy_variable=True, staleness=3),
    'ParallaxProxy': Parallax(local_proxy_variable=True),
    'PartitionedAR': PartitionedAR(),
    'RandomAxisPartitionAR': RandomAxisPartitionAR(chunk_size=4),
    'UnevenPartitionedPS': UnevenPartitionedPS(local_proxy_variable=True)
}


def run_test(resource, strategy, case):
    print("\n>>>>>>>> Running Test: Case:{}, Strategy:{}, ResourceSpec:{} >>>>>>>>\n".format(case, strategy, resource))
    a = AutoDist(resource_spec_file=resource, strategy_builder=STRATEGIES_FOR_DISTRIBUTED_TESTS[strategy])
    c = importlib.import_module("cases." + case)
    c.main(a)
    print('<<<<<<<<<< Test Case Finished. <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single training script')
    parser.add_argument('--case', type=str, 
                        help='case name of the test case')
    parser.add_argument('--strategy', type=str, 
                        help='strategy name of the test case')
    parser.add_argument('--resource', type=str, 
                        help='resource path of the test case')
    args = parser.parse_args()
    run_test(args.resource, args.strategy, args.case)
