# Copyright 2020 Petuum. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An AutoStrategy using a trained linear simulator."""

from autodist.strategy.auto.base import AutoStrategyBase
from autodist.autosync.simulator.linear_simulator import LinearSimulator

class AutoStrategy(AutoStrategyBase):
    """
    AutoStrategy builder using a trained linear simulator

    It generates a suitable Strategy based on graph_item and resource_spec using a pretrained simulator weight.
    This implementation currenlty provides a linear simulator weight trained on > 9000 data points.
    """

    def __init__(self):
        space = {
            'synchronizer_types': ['PS', 'AR'],
            'maybe_partition': [True, False],
            'compressor': ['HorovodCompressor', 'NoneCompressor'],
            'local_replication': [True, False],
            'partitionable_axis': [],
        }
        heuristics = {
            'ps_load_balancer': 'sorted_christy',  # None, 'christy', 'greedy', 'LP'
            'merge_scheme': 'ordered_balanced',  # random, by_chunk, christy, ordered_balanced
            'num_group_bounds': [-1, 20],
            'num_partition_bounds': [2, 40],
            'enable_single_node_no_partition': False,
            'same_synchronizer_for_parts': False,
        }

        simulator = LinearSimulator()

        super(AutoStrategy, self).__init__(
            space=space,
            heuristics=heuristics,
            num_proposals=2000,
            simulator=simulator
        )

    def build(self, graph_item, resource_spec):
        candidates = self.propose_n(graph_item, resource_spec, self._num_proposals)

        # Assess all candidates and simply pick the highest-scored one
        features, scores = self._simulator.inference(candidates)
        best_index = scores.index(min(scores))
        return candidates[best_index]
