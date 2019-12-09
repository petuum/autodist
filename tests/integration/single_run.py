import argparse
import importlib

from autodist import AutoDist
from autodist.strategy.all_reduce_strategy import AllReduce
from autodist.strategy.parallax_strategy import Parallax
from autodist.strategy.partitioned_ps_strategy import PartitionedPS
from autodist.strategy.ps_lb_strategy import PSLoadBalancing
from autodist.strategy.ps_strategy import PS

STRATEGIES_FOR_DISTRIBUTED_TESTS = {
    'PS': PS(),
    'PSLoadBalancing': PSLoadBalancing(),
    'PartitionedPS': PartitionedPS(),
    'AllReduce': AllReduce(),
    'Parallax': Parallax(),
    'PSLoadBalancingProxy': PSLoadBalancing(local_proxy_variable=True),
    'ParallaxProxy': Parallax(local_proxy_variable=True)
}


def run_test(resource, strategy, case):
    a = AutoDist(resource_spec_file=resource, strategy_builder=STRATEGIES_FOR_DISTRIBUTED_TESTS[strategy])
    c = importlib.import_module("cases." + case)
    print("received args are:", case, strategy, resource)
    c.main(a)


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
