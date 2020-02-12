"""Feed and Fetch Remapper."""
import contextlib

from tensorflow.python.client.session import _REGISTERED_EXPANSIONS
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.python.ops.variables import RefVariable

from autodist.kernel.common.resource_variable_utils import get_read_var_tensor
from autodist.kernel.common.utils import replica_prefix


class Remapper:
    """Feed and Fetch Remapper."""

    _default_remapper = None
    _default_registered_expansions = []
    _remap_fn = {
        ops.Tensor: lambda graph, name: graph.as_graph_element(name),
        ops.Operation: lambda graph, name: graph.as_graph_element(name),
        RefVariable: lambda graph, name: graph.as_graph_element(name),
        ResourceVariable: lambda graph, name: get_read_var_tensor(graph.get_tensor_by_name(name).op)
    }

    def __init__(self, graph_transformer, graph_item):
        self._graph_transformer = graph_transformer
        self._graph_item = graph_item

    def _remap_feed(self, feed, feed_val=None):
        """
        Remap the feeds to the right element in the transformed graph.

        For example, there are N copies of a placeholder for N replicas
          and we have to feed all of them with tensors.

        Args:
            feed: feed graph element
            feed_val: feed value

        Returns:
            List of (new_feed, new_feed_value) pairs
        """
        try:
            transformed_feeds = [self._graph_item.graph.as_graph_element(feed.name)]
        except KeyError:
            # Temporary Workaround for SYM-9004
            transformed_feeds = [
                ops.prepend_name_scope(feed.name, replica_prefix(i))
                for i in range(self._graph_transformer.num_local_replicas)
            ]
        if feed_val is not None:
            # TODO: map placeholders on different replicas to different values
            transformed_feeds = [(f, feed_val) for f in transformed_feeds]
        return transformed_feeds

    def _remap_fetch(self, fetch):
        """
        Remap the user-provided fetches to the right list of fetches after graph transformations.

        Cases:
            * If original fetch exists (which is not affected by graph transformation), fetch the original.
            * Otherwise, for fetches that are train_ops, fetch them on all replicas;
            * for other fetches, only fetch it on master replica.
                * For example, for partitioned vars, it corresponds to the concat one as_tensor on the first replica.
        """
        graph = self._graph_item.graph
        _remap_fn = Remapper._remap_fn
        fetch_type = type(fetch)
        try:
            transformed_fetch = [_remap_fn[fetch_type](graph, fetch.name)]
        except KeyError:
            master_replica_name = ops.prepend_name_scope(fetch.name, replica_prefix(0))
            master_replica_fetch = _remap_fn[fetch_type](graph, master_replica_name)

            def is_train_op(op):
                # In TF2: train_op as AssignAddVariableOp
                # In TF1 (being deprecated): no_op with a groups of stateful ops as control dependencies
                # TODO(unless deprecating): make the checking as strict as possible
                return op.op_def.is_stateful or op.op_def.name == 'NoOp'

            if fetch_type is ops.Operation and is_train_op(master_replica_fetch):
                transformed_fetch = [
                    _remap_fn[fetch_type](graph, ops.prepend_name_scope(fetch.name, replica_prefix(i)))
                    for i in range(self._graph_transformer.num_local_replicas)
                ]

                ####################################################################
                # # For Debugging Local Replicas
                ####################################################################
                # transformed_fetch = [
                #     _remap_fn[ops.Tensor](graph, ops.prepend_name_scope(
                #         'Mean:0',
                #         replica_prefix(i)))
                #     for i in range(self._graph_transformer.num_local_replicas)
                # ]
                # transformed_fetch = [_remap_fn[ops.Tensor](graph,
                #     ops.prepend_name_scope(
                #         'sampled_softmax_loss/embedding_lookup:0',
                #         replica_prefix(1)
                #     )
                # )]
                ####################################################################
                print('###', transformed_fetch)
            else:
                transformed_fetch = [master_replica_fetch]
        return transformed_fetch, lambda fetched_vals: fetched_vals[0]

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
                    ff = _autodist_feed_fn(fd, fv)
                    final_feeds.extend(ff)
                return final_feeds

            def nested_feed_fn_for_partial_run(feed, feed_fn_for_partial_run=feed_fn_for_partial_run):
                feeds = feed_fn_for_partial_run(feed)
                final_feeds = []
                for fd in feeds:
                    ff = _autodist_feed_fn(fd)
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
        yield self
        Remapper._clear_default()
