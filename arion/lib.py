
####################################
# Communicator
# (Lib package may have multiple
#    modules as different sub-libs.)
####################################

from abc import abstractmethod
from tensorflow.python.ops import nccl_ops


class Config:
    """
    Default Config for Communicator.
    """
    pass


class Synchronizer:
    """
    ONLY deals with the shared states (variables) of the graph
    """

    def __init__(self, config=Config()):
        self._config = config

    @abstractmethod
    def apply(self, graph, target=None):
        raise NotImplementedError


class PS(Synchronizer):

    def apply(self, graph, target=None):
        # apply ps to the graph ops and vars
        # Maybe use CrossDeviceOps

        return graph


class AR(Synchronizer):

    def apply(self, graph, target=None):
        nccl_ops.all_sum()

        return graph
