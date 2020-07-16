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

"""PS StrategyBuilder."""

from autodist.strategy.base import Strategy, StrategyBuilder
from autodist.proto import strategy_pb2
from autodist.strategy.auto.strategy_sampler import RandomStrategySampler


class AutoStrategy(StrategyBuilder):
    """
    AutoStrategy Builder.

    It generates a suitable Strategy based on graph_item and resource_spec following the AutoSync framework.
    """

    def __init__(self):
        return

    def build(self, graph_item, resource_spec):
        # TODO: merge the code in search and propose here.
        return

    def search(self):
        # candidates, scores, features = self.propose(self.search_params['num_candidate_explore'])
        candidates, scores, features = self.batch_propose(self.search_params['num_candidate_explore'])
        n_pick = self.search_params['num_candidate_per_trial']

        # cast them to be np arrays
        if self.search_params['diversity_metric'] == 'embedding':
            picked_candidates = self.submodular_pick_by_embedding(np.array(scores),
                                                                  candidates,
                                                                  np.stack(features),
                                                                  n_pick,
                                                                  self.search_params['simulation_weight'],
                                                                  self.search_params['diversity_weight'])
        elif self.search_params['diversity_metric'] == 'expression':
            picked_candidates = self.submodular_pick_by_expression(np.array(scores),
                                                                   candidates,
                                                                   n_pick,
                                                                   self.search_params['simulation_weight'],
                                                                   self.search_params['diversity_weight'])
        else:
            raise ValueError('Unrecognized diversity metric...')
        if self.trial_run_fn:
            self.trial_run(picked_candidates, search_iteration=0)

    def propose(self, num_proposal, use_simulator=True):
        builder = RandomStrategy(self.space, self.heuristics)
        candidates = []
        features = []
        scores = []
        # np.random.seed(1)
        idx = 0

        while len(candidates) < num_proposal:
            logging.info('Sampling strategy {}'.format(idx))
            start_time = time.time()
            expr = builder.build(self._original_graph_item, self._resource_spec)
            elapsed = time.time() - start_time
            logging.info('Sampling strategy takes {}'.format(elapsed))
            builder.reset()
            idx += 1
            logging.info('Progress {}/{}'.format(len(candidates), num_proposal))
            if self.simulator and use_simulator:
                start_time = time.time()
                score, feature = self.simulator.simulate(expr, self._resource_spec)
                elapsed = time.time() - start_time
                logging.info('Inference strategy takes {}'.format(elapsed))
                if score > self.search_params['rejection_score']:
                    logging.info('strategy {} has score {} > {}, '
                                 'rejected..'.format(idx, score, self.search_params['rejection_score']))
                    continue
                else:
                    candidates.append(expr)
                    features.append(feature)
                    scores.append(score[0])
            else:
                candidates.append(expr)
                features.append([])
                scores.append(0)
        logging.info('rejection ratio: {}'.format(1 - num_proposal / float(idx)))
        return candidates, scores, features

    def batch_propose(self, num_proposal, batch_size=32, use_simulator=True):

        builders = [RandomStrategy(self.space, self.heuristics) for _ in range(batch_size)]
        graph_items = [self._original_graph_item for _ in range(batch_size)]
        rss = [ResourceSpec(self.resource_file) for _ in range(batch_size)]
        candidates = []
        features = []
        scores = []
        # np.random.seed(1)
        idx = 0

        while len(candidates) < num_proposal:
            logging.info('Sampling strategy {}'.format(idx))
            start_time = time.time()

            q = Queue()
            exprs = []
            prs = []
            for obj, arg1, arg2 in zip(builders, graph_items, rss):
                prs.append(Process(target=build_worker, args=(q, obj, arg1, arg2)))
                prs[-1].start()
            for pr in prs:
                expr = q.get() # will block
                exprs.append(expr)
            for pr in prs:
                pr.join()

            elapsed = time.time() - start_time
            logging.info('Sampling strategy takes {}'.format(elapsed))
            for builder in builders: builder.reset()
            logging.info('Progress {}/{}'.format(len(candidates), num_proposal))
            if self.simulator and use_simulator:
                start_time = time.time()
                batch_score, batch_feature = self.simulator.simulate(exprs, rss)
                elapsed = time.time() - start_time
                logging.info('Inference strategy takes {}'.format(elapsed))
                for ite, expr in enumerate(exprs):
                    # print(batch_score[ite], batch_feature[ite].shape)
                    if batch_score[ite] > self.search_params['rejection_score']:
                        logging.info('strategy {} has score {} > {}, '
                                     'rejected..'.format(idx+ite, batch_score[ite], self.search_params['rejection_score']))
                    else:
                        candidates.append(expr)
                        features.append(batch_feature[ite])
                        scores.append(batch_score[ite])
            else:
                for ite, expr in enumerate(exprs):
                    candidates.append(expr)
                    features.append([])
                    scores.append(0)
            idx += batch_size
        logging.info('rejection ratio: {}'.format(1 - num_proposal / float(idx)))
        return candidates[:num_proposal], scores[:num_proposal], features[:num_proposal]

    def submodular_pick_by_embedding(self,
                                     scores,
                                     candidates,
                                     candidate_features,
                                     n_pick,
                                     beta=1.0,
                                     alpha=1.0):
        n = len(scores)
        assert n == len(candidate_features)

        ret = []
        sim = np.dot(candidate_features, candidate_features.T)
        remain = list(range(len(scores)))

        for _ in range(n_pick):
            tmp_delta = -scores[remain] * beta
            if len(ret) > 0:
                tmp_delta -= alpha * (sim[remain, :][:, ret]).mean(1)
            max_x = tmp_delta.argmax()
            max_x = remain[max_x]

            ret.append(max_x)
            remain.remove(max_x)

        return [candidates[i] for i in ret]

    def submodular_pick_by_expression(self,
                                      scores,
                                      candidates,
                                      n_pick,
                                      beta=1.0,
                                      alpha=1.0):

        def remove_group_or_reduction_destination(strategy):
            tmp_strategy = copy.deepcopy(strategy)
            for node in tmp_strategy.node_config:
                if node.partitioner:
                    for part in node.part_config:
                        synchronizer = getattr(part, part.WhichOneof('synchronizer'))
                        if hasattr(synchronizer, 'reduction_destination'):
                            synchronizer.reduction_destination = ''
                        else:
                            synchronizer.group = 0
                else:
                    synchronizer = getattr(node, node.WhichOneof('synchronizer'))
                    if hasattr(synchronizer, 'reduction_destination'):
                        synchronizer.reduction_destination = ''
                    else:
                        synchronizer.group = 0
            return tmp_strategy

        def estimate_difference(strategy, node_config_set):
            score = 0
            for i, node in enumerate(strategy.node_config):
                if_seen = False
                for seen_node in node_config_set[i]:
                    if seen_node == node:
                        if_seen = True
                        break
                if not if_seen:
                    score += 1
            return score

        assert len(scores) == len(candidates)

        node_config_set = [list() for _ in candidates[0].node_config]
        remain = list(range(len(scores)))
        ret = []
        for _ in range(n_pick):
            max_x = -1
            max_delta = -1e9
            max_strategy_copy = None

            for x in remain:
                tmp_strategy = remove_group_or_reduction_destination(candidates[x])
                diff_score = estimate_difference(tmp_strategy, node_config_set)
                assert(diff_score <= len(tmp_strategy.node_config))
                # print('diff score {}..'.format(diff_score))
                tmp_delta = - scores[x] * beta + diff_score * alpha
                if tmp_delta > max_delta:
                    max_delta, max_x, max_strategy_copy = tmp_delta, x, tmp_strategy
                    max_diff_score = diff_score *alpha
                    max_simulation_score= -scores[x]

            print('Add one candidate with max score: {}, {}, {}'.format(max_simulation_score, max_diff_score, max_delta))
            ret.append(max_x)
            remain.remove(max_x)

            # update the node config set
            for i, node in enumerate(max_strategy_copy.node_config):
                if_seen = False
                for seen_node in node_config_set[i]:
                    if seen_node == node:
                        if_seen = True
                        break
                if not if_seen:
                    node_config_set[i].append(node)

        return [candidates[i] for i in ret]
