import json
import time
from multiprocessing import Process, Queue

import copy
import numpy as np
import os

from arion.const import DEFAULT_RANDOM_SEARCH_DIR
from arion.graph_item import GraphItem
from arion.resource_spec import ResourceSpec
from arion.strategy import RandomStrategy, AllReduce
from arion.utils import logging


def build_worker(queue, builder, gi, rs):
    np.random.seed()
    ret = builder.build(gi, rs)
    queue.put(ret)

def get_resource_specs(trial_resource_spec_dir):
    resource_specs = []
    if os.path.isdir(trial_resource_spec_dir):
        for file_name in os.listdir(trial_resource_spec_dir):
            file_path = os.path.join(trial_resource_spec_dir, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.yml'):
                resource_specs.append(file_path)
    elif os.path.isfile(trial_resource_spec_dir):
        resource_specs.append(trial_resource_spec_dir)
    else:
        raise ValueError("Cannot find valid files in {}".format(trial_resource_spec_dir))
    return resource_specs


def get_strategies(strategies_dir):
    strategies = []
    if os.path.isdir(strategies_dir):
        for file_name in os.listdir(strategies_dir):
            file_path = os.path.join(strategies_dir, file_name)
            if os.path.isfile(file_path) and file_path.split('/')[-1].startswith('2020'):
                strategies.append(file_path)
    elif os.path.isfile(strategies_dir):
        strategies.append(strategies_dir)
    else:
        raise ValueError("Cannot find valid files in {}".format(strategies_dir))
    return strategies


class RandomSearch:
    def __init__(self,
                 space,
                 heuristics,
                 search_params,
                 original_graph_item_path,
                 resource_file,
                 simulator=None,
                 trial_run_fn=None):

        self.space = space
        self.heuristics = heuristics
        self.search_params = search_params

        self.original_graph_item_path = original_graph_item_path
        self.resource_file = resource_file

        self.simulator = simulator
        self.trial_run_fn = trial_run_fn

        self._resource_spec = ResourceSpec(self.resource_file)
        self._original_graph_item = GraphItem.deserialize(original_graph_item_path)

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

    def trial_run(self,
                  candidate_strategies=None,
                  search_iteration=0):
        # serialize all candidates to folder
        target_dir = os.path.join(DEFAULT_RANDOM_SEARCH_DIR, str(search_iteration))
        os.makedirs(target_dir, exist_ok=False)
        self._serialize_candidate_strategies(candidate_strategies, target_dir)
        self._save_hyperparams(target_dir)

        # launch trial run
        self._launch_trial_run(target_dir)

    @staticmethod
    def _serialize_candidate_strategies(candidate_strategies, target_dir):
        for strategy in candidate_strategies:
            path = os.path.join(target_dir, strategy.id)
            strategy.serialize(path)

    def _launch_trial_run(self, strategies_dir):
        strategies = get_strategies(strategies_dir)

        # this will launch distributed processes and take very long
        self.trial_run_fn([self.resource_file], strategies)

    def _save_hyperparams(self, target_dir):
        # copy the constraint file as well
        space_file = os.path.join(target_dir, 'space.json')
        with open(space_file, 'w') as f:
            json.dump(self.space, f)
        heuristics_file = os.path.join(target_dir, 'heuristics.json')
        with open(heuristics_file, 'w') as f:
            json.dump(self.heuristics, f)
        search_params_file = os.path.join(target_dir, 'search_params.json')
        with open(search_params_file, 'w') as f:
            json.dump(self.search_params, f)

    def check_if_visited(self):
        raise NotImplementedError()

    def check_if_trial_run(self):
        raise NotImplementedError()

    # Don't use, only for debug.
    def _single_run(self):
        # builder = BalancedPartitionedPS()
        # builder = PartitionedAR(chunk_size=1)
        builder = AllReduce()
        expr = builder.build(self._original_graph_item, self._resource_spec)
        logging.info(expr)
        self.trial_run([expr], search_iteration=0)
