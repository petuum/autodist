# Copyright 2020 Petuum, Inc. All Rights Reserved.
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

"""Feed and Fetch Remapper."""
import contextlib
import numpy as np

from tensorflow.python.client.session import _REGISTERED_EXPANSIONS
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.core.protobuf import config_pb2

from autodist.kernel.common.variable_utils import get_read_var_tensor
from autodist.kernel.common.utils import replica_prefix
from autodist.utils import logging


class Remapper:
    """
    Remaps feeds and fetches for Sessions (and eager mode).

    Sets itself as the default remapper when in its scope
    and remaps graph elements to account for AutoDist's graph
    transformation.

    Feed Behavior:

    * We want to remap feeds to the corresponding element in the
      AutoDist-transformed graph. If the element has a polymorphic
      size (e.g. it has a `None` size for the batch dimension),
      we split the feed up across all replicas of the element.
      Otherwise, we duplicate the feed across all replicas of
      the element.

    Fetch Behavior:

    * We want to remap the potential new shards and replicas of
      a graph element into the original fetch.
    * Cases:

        * If original fetch exists (which is not affected by graph transformation), fetch the original.
        * Otherwise, for fetches that are train_ops, fetch them from all replicas;
        * for other fetches, only fetch it from the master replica.

            * For example, for partitioned vars, it corresponds to the concat one as_tensor on the first replica.
    """

    _default_remapper = None
    _default_registered_expansions = []

    def __init__(self, graph_transformer, graph_item):
        self._graph_transformer = graph_transformer
        self._graph_item = graph_item

    def _polymorphic_dim(self, o):
        if isinstance(o, ops.Tensor) and self._graph_transformer.num_local_replicas > 1 and \
                bool(o.shape) and not o.shape.is_fully_defined():
            return o.shape.as_list().index(None)  # first leftmost undefined shape index
        return None

    def _remap_element(self, ele_type, name):
        """Remap element based on type."""
        graph = self._graph_item.graph
        if ele_type is ResourceVariable:
            res = get_read_var_tensor(graph.get_tensor_by_name(name).op)
        else:  # Default element mapper, including the RefVariable case
            res = graph.as_graph_element(name, allow_tensor=True, allow_operation=True)
        return res

    def _remap_feed(self, feed, feed_val=None):
        """
        Remap the feeds to the right element in the transformed graph.

        For example, there are N copies of a placeholder for N replicas
          and we have to feed all of them with tensors.

        Args:
            feed: feed graph element or name
            feed_val: feed value

        Returns:
            List of (new_feed, new_feed_value) pairs
        """
        feed_name = feed if isinstance(feed, str) else feed.name
        try:
            transformed_feeds = [self._graph_item.graph.as_graph_element(feed_name)]
        except KeyError:
            transformed_feeds = [
                self._graph_item.graph.as_graph_element(
                    ops.prepend_name_scope(feed_name, replica_prefix(i))
                )
                for i in range(self._graph_transformer.num_local_replicas)
            ]

        num_replicated_feeds = self._graph_transformer.num_local_replicas
        feed = feed if not isinstance(feed, str) else transformed_feeds[0]

        def expand_feed_val(feed_val, feed=feed):
            """Given a original feed or replicated feed, expand the feed value."""
            # If we have replicated placeholders with undefined (polymorphic) shape, we split the feed_val across it;
            #  otherwise we feed all replicated placeholders the same feed_val
            polymorphic_dim = self._polymorphic_dim(feed)
            if polymorphic_dim:
                feed_vals = np.array_split(np.asarray(feed_val), num_replicated_feeds, axis=polymorphic_dim)
            else:
                feed_vals = [feed_val for _ in range(num_replicated_feeds)]
            return feed_vals

        if feed_val is not None:
            feed_vals = expand_feed_val(feed_val)
            transformed_feeds = list(zip(transformed_feeds, feed_vals))
        return transformed_feeds, expand_feed_val

    def _remap_fetch(self, fetch):
        """
        Remap the user-provided fetches to the right list of fetches after graph transformations.

        Cases:
            * If original fetch exists (which is not affected by graph transformation), fetch the original.
            * Otherwise, for fetches that are train_ops, fetch them on all replicas;
            * for other fetches, only fetch it on master replica.
                * For example, for partitioned vars, it corresponds to the concat one as_tensor on the first replica.
        """
        _remap_element = self._remap_element
        fetch_type = type(fetch)
        fetch_name = fetch if isinstance(fetch, str) else fetch.name
        contract_fn = lambda fetched_vals: fetched_vals[0]  # noqa: E731
        try:
            transformed_fetch = [_remap_element(fetch_type, fetch_name)]
        except KeyError:
            master_replica_name = ops.prepend_name_scope(fetch_name, replica_prefix(0))
            master_replica_fetch = _remap_element(fetch_type, master_replica_name)
            polymorphic_dim = self._polymorphic_dim(master_replica_fetch)

            def is_train_op(op):
                # In TF2: train_op as AssignAddVariableOp
                # In TF1 (being deprecated): no_op with a groups of stateful ops as control dependencies
                # TODO(unless deprecating): make the checking as strict as possible
                return isinstance(op, ops.Operation) and (op.op_def.is_stateful or op.op_def.name == 'NoOp')

            if is_train_op(master_replica_fetch):
                transformed_fetch = [
                    _remap_element(fetch_type, ops.prepend_name_scope(fetch_name, replica_prefix(i)))
                    for i in range(self._graph_transformer.num_local_replicas)
                ]
                ####################################################################
                # # For Debugging Local Replicas
                ####################################################################
                # transformed_fetch = [
                #     self._graph_item.graph.as_graph_element('AutoDist-Replica-0/emb/part_0_take_grad')
                # ]
                # transformed_fetch = [
                #     _remap_element(ops.Tensor, ops.prepend_name_scope(
                #         'Mean:0',
                #         replica_prefix(i)))
                #     for i in range(self._graph_transformer.num_local_replicas)
                # ]
                # transformed_fetch = [_remap_element(ops.Tensor,
                #     ops.prepend_name_scope(
                #         'sampled_softmax_loss/embedding_lookup:0',
                #         replica_prefix(1)
                #     )
                # )]
                ####################################################################
                logging.debug('Fetch mapped from {} to {}'.format(fetch, transformed_fetch))
            elif polymorphic_dim:
                transformed_fetch = [
                    _remap_element(fetch_type, ops.prepend_name_scope(fetch_name, replica_prefix(i)))
                    for i in range(self._graph_transformer.num_local_replicas)
                ]
                contract_fn = lambda fetch_vals: np.concatenate(fetch_vals, axis=polymorphic_dim)  # noqa: E731
            else:
                transformed_fetch = [master_replica_fetch]
        return transformed_fetch, contract_fn

    def remap_callable_options(self, callable_options):
        """
        Remap Callable Options.

        Args:
            callable_options: A `CallableOptions` protocol buffer message describing
              the computation that will be performed by the callable.

        Returns:
            A new CallableOptions
        """
        # Prepare callable options.
        new_callable_options = config_pb2.CallableOptions()
        callable_arg_fns = []
        # Handle external-data feed.
        for f in callable_options.feed:
            nf, fn = self._remap_feed(f)
            new_callable_options.feed.extend([o.name for o in nf])
            callable_arg_fns.append(fn)

        # Handle connection.
        if len(callable_options.tensor_connection) > 0:
            # TODO: Remapping Tensor Connections for New Callable Options
            # for c in callable_options.tensor_connection:
            #   connection = new_callable_options.tensor_connection.add()
            #   connection.from_tensor ~ remap(c.from_tensor)
            #   connection.to_tensor ~ remap(c.to_tensor)
            raise NotImplementedError('AutoDist will support feeding symbolic connections later.')

        # Handle fetches.
        for f in callable_options.fetch:
            nf, _ = self._remap_fetch(f)
            new_callable_options.fetch.extend([o.name for o in nf])
        # Handle updates.
        for f in callable_options.target:  # f in type str
            if f:
                nf, _ = self._remap_fetch(f)
                new_callable_options.target.extend([o.name for o in nf])
        # Handle run_options.
        new_callable_options.run_options.CopyFrom(callable_options.run_options)
        return new_callable_options, callable_arg_fns

    def _set_default(self):
        """Switch the current global default mapper to be the current one and register conversion functions."""
        assert Remapper._default_remapper is None
        _autodist_fetch_fn = self._remap_fetch
        _autodist_feed_fn = self._remap_feed
        for i, expansion in enumerate(_REGISTERED_EXPANSIONS):

            tensor_type, fetch_fn, feed_fn, feed_fn_for_partial_run = expansion

            # Register nested conversion functions while keeping the function types
            # tensor_type: type
            # fetch_fn: fetch -> (List[fetch], List[fetched_val] -> final_fetched_val)
            # feed_fn: (feed, feed_val) -> List[(feed, feed_val)]
            # feed_fn_for_partial_run: feed -> List[feed]

            def nested_fetch_fn(fetch, fetch_fn=fetch_fn):
                """
                Two-level nested fetch function.

                TensorFlow ElementMapper Fetch Expansion -> AutoDist Fetch Expansion ->
                AutoDist Fetch Values Contraction -> TensorFlow ElementMapper Fetch Values Contraction

                Note that `fetch_fn(local_var)=fetch_fn(instant_outer_var)` is to avoid cell-var-from-loop issue.
                """
                fetches, contract_fn = fetch_fn(fetch)
                final_fetches = []
                inner_contract_fns = {}
                for f in fetches:
                    ff, fn = _autodist_fetch_fn(f)
                    i, j = len(final_fetches), len(final_fetches) + len(ff)
                    final_fetches.extend(ff)
                    inner_contract_fns[(i, j)] = fn
                return final_fetches, lambda fetched_vals: contract_fn([
                    fn(fetched_vals[i:j]) for (i, j), fn in sorted(inner_contract_fns.items())
                ])

            def nested_feed_fn(feed, feed_val, feed_fn=feed_fn):
                feeds = feed_fn(feed, feed_val)
                final_feeds = []
                for fd, fv in feeds:
                    ff, _ = _autodist_feed_fn(fd, fv)
                    final_feeds.extend(ff)
                return final_feeds

            def nested_feed_fn_for_partial_run(feed, feed_fn_for_partial_run=feed_fn_for_partial_run):
                feeds = feed_fn_for_partial_run(feed)
                final_feeds = []
                for fd in feeds:
                    ff, _ = _autodist_feed_fn(fd)
                    final_feeds.extend(ff)
                return final_feeds

            # Backup the _REGISTERED_EXPANSTION[i]
            Remapper._default_registered_expansions.append(expansion)
            # Write the new _REGISTERED_EXPANSTION[i]
            _REGISTERED_EXPANSIONS[i] = (
                tensor_type,
                nested_fetch_fn,
                nested_feed_fn,
                nested_feed_fn_for_partial_run
            )

        Remapper._default_remapper = self

    def _is_default(self):
        """Whether the current remapper is the default one."""
        return Remapper._default_remapper == self

    @staticmethod
    def _clear_default():
        """Un-register conversion functions and clear default remapper."""
        _REGISTERED_EXPANSIONS[:] = Remapper._default_registered_expansions[:]
        Remapper._default_registered_expansions = []
        Remapper._default_remapper = None

    @contextlib.contextmanager
    def as_default(self):
        """Ensure the current one as the default remapper; otherwise switch to it."""
        if self._is_default():
            raise SyntaxError('Nested remapper context is invalid.')
        self._set_default()
        try:
            yield self
        finally:
            Remapper._clear_default()
