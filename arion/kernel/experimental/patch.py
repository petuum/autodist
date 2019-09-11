"""Experimental Patch on TF."""
from tensorflow.python import ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

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
