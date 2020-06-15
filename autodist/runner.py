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

"""
Runner.

Contains functionality and classes needed for
running an AutoDist-transformed graph with TensorFlow.
"""
import hashlib
import os

from tensorflow.core.protobuf import config_pb2, rewriter_config_pb2
from tensorflow.python.client import timeline, session as tf_session

import autodist.autodist
import autodist.const
from autodist.const import ENV, MAX_INT32
from autodist.utils import logging


def get_default_session_config():
    """Create a default session config."""
    session_config = config_pb2.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True
    # session_config.log_device_placement = True

    # enable scoped_allocator for collective_ops
    rewrite_options = session_config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')
    return session_config


def get_default_run_options():
    """Create a default run option."""
    run_options = config_pb2.RunOptions()
    # Force every worker session to use different collective graph keys,
    # thus to make session run has different TensorFlow step_ids when fetching collective_op.
    # A different step_id avoids skipping re-initialization of a rendezvous object.
    # Shared step_id may cause racing for different sessions, across which some variable is shared.
    # For more information, please refer to TensorFlow worker.proto: `collective_graph_key`
    run_options.experimental.collective_graph_key = int(
        hashlib.md5(ENV.AUTODIST_WORKER.val.encode()).hexdigest(), 16
    ) % MAX_INT32
    # run_options.trace_level = config_pb2.RunOptions.FULL_TRACE
    return run_options


# TODO(Hao): could extend this to use tfprof (though I don't
#            see immediate benefit now)
def _log_timeline(run_metadata, name='timeline', step=0):
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    directory = os.path.join(autodist.const.DEFAULT_WORKING_DIR, "traces")
    os.makedirs(directory, exist_ok=True)
    # TODO(Hao): add a runner step count and use it here.
    p = os.path.join(directory, "{}_{}.json".format(name, step))
    with open(p, "w") as f:
        f.write(chrome_trace)
        logging.info('Traced timeline written to: %s' % p)


class WrappedSession(tf_session.Session):
    """
    Wraps :tf_main:`tf.Session <compat/v1/Session>`.

    This allows us to use our transformed `Graph` and our `Remapper`
    on our cluster.
    """

    def __init__(self, cluster, graph_item, remapper):
        """Initialize Wrapped Session."""
        self._cluster = cluster
        self._graph_item = graph_item
        self._remapper = remapper

        super(WrappedSession, self).__init__(
            target=self._cluster.get_local_session_target(),
            graph=self._graph_item.graph,
            config=get_default_session_config()
        )
        # TensorFlow default initializations
        super(WrappedSession, self).run(
            self._graph_item.get_ops_in_graph(self._graph_item.info.initializers)
        )

    def _make_callable_from_options(self, callable_options):
        self._extend_graph()
        return WrappedSession._Callable(self, callable_options)

    class _Callable(tf_session.Session._Callable):

        def __init__(self, session, callable_options):
            self._callable_options, self._callable_arg_fns = session._remapper.remap_callable_options(callable_options)
            super(WrappedSession._Callable, self).__init__(session, self._callable_options)

        def __call__(self, *args, **kwargs):
            autodist.autodist.get_default_autodist().is_built()
            args = [a for fn, arg in zip(self._callable_arg_fns, args) for a in fn(arg)]
            return super(WrappedSession._Callable, self).__call__(*args, **kwargs)

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """Wrapped Session.run."""
        autodist.autodist.get_default_autodist().is_built()
        _options = get_default_run_options()
        if options:
            _options.MergeFrom(options)  # options merges (while overwrites) into RUN_OPTIONS
        is_tracing = _options.trace_level > config_pb2.RunOptions.NO_TRACE
        if not run_metadata and is_tracing:
            run_metadata = config_pb2.RunMetadata()
        with self._remapper.as_default():
            res = super(WrappedSession, self).run(
                fetches, feed_dict=feed_dict, options=_options, run_metadata=run_metadata
            )
        if is_tracing:
            _log_timeline(run_metadata)
        return res
