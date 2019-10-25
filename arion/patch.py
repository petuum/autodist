"""Experimental Patch on TF."""
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from autodist.utils import logging
from autodist.graph_item import wrap_optimizer_init, wrap_optimizer_apply_gradient


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

    @staticmethod
    def patch_optimizers():
        """Patch all instances of OptimizerV2 for us to store optimizer and gradient information."""
        for subclass in optimizer_v2.OptimizerV2.__subclasses__():
            subclass.__init__ = wrap_optimizer_init(subclass.__init__)
            subclass.apply_gradients = wrap_optimizer_apply_gradient(subclass.apply_gradients)
