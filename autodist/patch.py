# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# It includes the derived work based on:
# https://github.com/tensorflow/tensorflow
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Patches for TensorFlow."""

from collections import deque
from itertools import chain

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import keras
from tensorflow.python.client import session as session_module
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.python.training.optimizer import Optimizer as OptimizerV1

import autodist.autodist
from autodist.graph_item import wrap_optimizer_init, wrap_optimizer_apply_gradient
from autodist.runner import get_default_session_config
from autodist.utils import logging


class PatchTensorFlow:
    """
    Patches for TensorFlow.

    There are multiple "monkey patches":
    1) Patches the `value` attribute of a `ResourceVariable` to return
    the `graph_element` or `cached_value` instead of a `ReadVariableOp`.
    Not having to read the variable from memory again can provide
    performance increases.
    2) Patches optimizers to save their type and arguments so that we
    can re-create them when partitioning variables.
    3) Patches Keras' `Session` fetcher to return AutoDist's custom `Session`
    so that AutoDist is compatible with Keras.
    """

    @staticmethod
    def patch_var_reading():
        """It only works with tf.gradients but not tape.gradients."""

        def value(self):
            """A cached operation which reads the value of this variable."""
            if self._cached_value is not None:
                return self._cached_value
            with ops.colocate_with(None, ignore_existing=True):
                with ops.device(self._handle.device):
                    # return self._read_variable_op() # original line
                    return self._graph_element

        setattr(ResourceVariable, 'value', value)
        logging.warning('Resource variable is patched '
                        'to behave as ref (only on reading) to avoid multiple recv_tensor.')

    _DEFAULT_VAR_READING = ResourceVariable.value

    @staticmethod
    def unpatch_var_reading():
        """Revert the ReadVariable patch."""
        setattr(ResourceVariable, 'value', PatchTensorFlow._DEFAULT_VAR_READING)

    @staticmethod
    def patch_optimizers():
        """Patch all instances of OptimizerV2 for us to store optimizer and gradient information."""
        q = deque(chain(OptimizerV2.__subclasses__(), OptimizerV1.__subclasses__()))
        while q:
            subclass = q.popleft()
            q.extend(list(subclass.__subclasses__()))
            subclass.__init__ = wrap_optimizer_init(subclass.__init__)
            subclass.apply_gradients = wrap_optimizer_apply_gradient(subclass.apply_gradients)
            logging.debug('Optimizer type: %s has been patched' % subclass.__name__)

    # TODO: unpatch optimizers

    _DEFAULT_MODEL_COMPILE = training.Model.compile
    _DEFAULT_GET_SESSION = keras.backend._get_session
    _DEFAULT_GRAPH_EXECUTION_FUNCTION = keras.backend.GraphExecutionFunction

    @staticmethod
    def patch_keras():
        """Patch Keras way of getting session."""
        setattr(keras.backend, 'READY_FOR_AUTODIST', False)

        def _compile(self, *args, **kwargs):
            PatchTensorFlow._DEFAULT_MODEL_COMPILE(self, *args, **kwargs)
            setattr(keras.backend, 'READY_FOR_AUTODIST', True)

        training.Model.compile = _compile

        keras.backend._get_session = _KerasPatch.get_session

        keras.backend.GraphExecutionFunction = _KerasPatch.GraphExecutionFunction

    @staticmethod
    def unpatch_keras():
        """Revert the Keras patch."""
        training.Model.compile = PatchTensorFlow._DEFAULT_MODEL_COMPILE
        keras.backend._get_session = PatchTensorFlow._DEFAULT_GET_SESSION
        keras.backend.GraphExecutionFunction = PatchTensorFlow._DEFAULT_GRAPH_EXECUTION_FUNCTION


class _KerasPatch:
    @staticmethod
    def get_session(op_input_list=()):
        """Returns the session object for the current thread."""
        _SESSION = keras.backend._SESSION # noqa:N806
        default_session = ops.get_default_session()
        if default_session is not None:
            session = default_session
        else:
            if ops.inside_function():
                raise RuntimeError('Cannot get session inside Tensorflow graph function.')
            # If we don't have a session, or that session does not match the current
            # graph, create and cache a new session.
            if getattr(_SESSION, 'session', None) is None:
                if getattr(keras.backend, 'READY_FOR_AUTODIST', False):
                    _SESSION.session = autodist.autodist.get_default_autodist().create_distributed_session()
                else:
                    _SESSION.session = session_module.Session(config=get_default_session_config())
            session = _SESSION.session
        return session

    class GraphExecutionFunction(keras.backend.GraphExecutionFunction):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not kwargs.get('updates'):
                self.updates_op = None

        # pylint: disable=attribute-defined-outside-init
        def _make_callable(self, feed_arrays, feed_symbols, symbol_vals, session):
            """
            Generates a callable that runs the graph.

            Arguments:
              feed_arrays: List of input tensors to be fed Numpy arrays at runtime.
              feed_symbols: List of input tensors to be fed symbolic tensors at runtime.
              symbol_vals: List of symbolic tensors to be fed to `feed_symbols`.
              session: Session to use to generate the callable.

            Returns:
              Function that runs the graph according to the above options.
            """
            # Prepare callable options.
            callable_opts = config_pb2.CallableOptions()
            # Handle external-data feed.
            for x in feed_arrays:
                callable_opts.feed.append(x.name)
            if self.feed_dict:
                for key in sorted(self.feed_dict.keys()):
                    callable_opts.feed.append(key.name)
            # Handle symbolic feed.
            for x, y in zip(feed_symbols, symbol_vals):
                connection = callable_opts.tensor_connection.add()
                if x.dtype != y.dtype:
                    y = math_ops.cast(y, x.dtype)
                from_tensor = ops._as_graph_element(y)
                if from_tensor is None:
                    from_tensor = y
                connection.from_tensor = from_tensor.name  # Data tensor
                connection.to_tensor = x.name  # Placeholder
            # Handle fetches.
            for x in self.outputs + self.fetches:
                callable_opts.fetch.append(x.name)
            # Handle updates.
            if self.updates_op:
                callable_opts.target.append(self.updates_op.name)
            # Handle run_options.
            if self.run_options:
                callable_opts.run_options.CopyFrom(self.run_options)
            # Create callable.
            callable_fn = session._make_callable_from_options(callable_opts)
            # Cache parameters corresponding to the generated callable, so that
            # we can detect future mismatches and refresh the callable.
            self._callable_fn = callable_fn
            self._feed_arrays = feed_arrays
            self._feed_symbols = feed_symbols
            self._symbol_vals = symbol_vals
            self._fetches = list(self.fetches)
            self._session = session
