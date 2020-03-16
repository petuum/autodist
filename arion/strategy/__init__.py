"""
Strategies and the corresponding builders.

For new users of AutoDist, one can choose one of the following built-in strategy builders:
"""
from .ps_strategy import PS
from .ps_lb_strategy import PSLoadBalancing
from .partitioned_ps_strategy import PartitionedPS
from .all_reduce_strategy import AllReduce
from .parallax_strategy import Parallax
