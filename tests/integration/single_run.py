import argparse
import importlib

from autodist import AutoDist

def run_test(resource, strategy, case):
    a = AutoDist(resource_spec_file=resource, strategy_name=strategy)
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
