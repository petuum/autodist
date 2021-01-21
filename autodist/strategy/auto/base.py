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

"""A base class to implementating different auto strategies."""

from multiprocessing import Process, Queue

import numpy as np

from autodist.strategy.auto.strategy_sampler import RandomStrategySampler, \
    default_space, default_heuristics
from autodist.strategy.base import StrategyBuilder
from autodist.utils import logging


class AutoStrategyBase(StrategyBuilder):
    """AutoStrategy Base class."""

    def __init__(self,
                 space=None,
                 heuristics=None,
                 num_proposals=1000,
                 simulator=None,
                 train_simulator=False):
        # space and heuristics params
        self._space = space or default_space
        self._heuristics = heuristics or default_heuristics

        # params
        self._num_proposals = num_proposals
        self._sampler = RandomStrategySampler(self._space,
                                              self._heuristics)
        if train_simulator:
            raise NotImplementedError()
        self._simulator = simulator

    def build(self, graph_item, resource_spec):
        raise NotImplementedError()

    def propose_one(self, graph_item, resource_spec):
        """
        Sequentially generate `self._num_proposals` strategies.

        Args:
            graph_item:
            resource_spec:

        Returns:
            Strategy
        """
        proposal = self._sampler.build(graph_item, resource_spec)
        return proposal

    def propose_n(self,
                  graph_item,
                  resource_spec,
                  num_proposals,
                  num_threads=1):
        """
        Proposal `num_proposals` strategies using multi-threading.

        Args:
            graph_item:
            resource_spec:
            num_proposals:
            num_threads:

        Returns:
            List(Strategy)
        """
        if num_threads > 1:
            def sampler_worker(q, sampler, graph_item, resource_spec):
                np.random.seed()
                expr = sampler.build(graph_item, resource_spec)
                q.put(expr)

            proposals = []
            while len(proposals) < num_proposals:
                # create thread-safe objects before multi-threading
                samplers = [RandomStrategySampler(graph_item, resource_spec) for _ in range(num_threads)]
                graph_items = [graph_item for _ in range(num_threads)]
                resource_specs = [resource_spec for _ in range(num_threads)]
                q = Queue()
                threads = []
                try:
                    for sampler, gi, rs in zip(samplers, graph_items, resource_specs):
                        thread = Process(target=sampler_worker, args=(q,sampler, gi, rs))
                        thread.start()
                        threads.append(thread)
                    batch = [q.get() for _ in threads]
                    proposals.extend(batch)
                    for thread in threads:
                        thread.join()
                except:
                    logging.error('Error when proposing strategies with {} threads'.format(num_threads))
                    raise
        else:
            proposals = [self.propose_one(graph_item, resource_spec) for i in range(num_proposals)]
        return proposals
