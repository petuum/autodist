"""Experimental Patch on TF."""

from collections import deque
from itertools import chain

from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.python.training.optimizer import Optimizer as OptimizerV1

from autodist.graph_item import wrap_optimizer_init, wrap_optimizer_apply_gradient
from autodist.utils import logging


class PatchTensorFlow:
    """Experimental Patch on TF."""

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
        """Revert the patch."""
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

    @staticmethod
    def patch_keras(autodist):
        """Patch Keras way of getting session."""
        def _get_session(op_input_list=()):
            """Returns the session object for the current thread."""
            _SESSION = keras.backend._SESSION
            default_session = ops.get_default_session()
            if default_session is not None:
                session = default_session
            else:
                if ops.inside_function():
                    raise RuntimeError('Cannot get session inside Tensorflow graph function.')
                # If we don't have a session, or that session does not match the current
                # graph, create and cache a new session.
                if getattr(_SESSION, 'session', None) is None:
                    _SESSION.session = autodist.create_distributed_session()
                session = _SESSION.session
            return session
        keras.backend._get_session = _get_session

    _DEFAULT_GET_SESSION = keras.backend._get_session

    @staticmethod
    def unpatch_keras():
        """Revert the patch."""
        keras.backend._get_session = PatchTensorFlow._DEFAULT_GET_SESSION
