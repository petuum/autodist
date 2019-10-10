"""Experimental Patch on TF."""
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from autodist.utils import logging
from autodist.graph_item import wrap_gradients


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
    def init_gradient_handler():
        """Wrap the apis for gradients."""
        from tensorflow.python.ops import gradients_util
        original_api = gradients_util._GradientsHelper
        new_api = wrap_gradients(original_api)
        gradients_util._GradientsHelper = new_api

        from tensorflow.python.eager import backprop
        original_api = backprop.GradientTape.gradient
        new_api = wrap_gradients(original_api)
        backprop.GradientTape.gradient = new_api
        # TODO: tape
