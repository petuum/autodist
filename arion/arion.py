from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_contextlib

from .strategy import Strategy
from .runner import Runner


class AutoDist:
    """
    User Interface.

    Notes:
        * Don't initialize session here.
        * Independent of distributed logic.
    """

    def __init__(self, resource_spec, strategy_name=None):
        self._graph = ops.Graph()
        self._resource_spec = resource_spec

    @tf_contextlib.contextmanager
    def scope(self):
        with context.graph_mode(), self._graph.as_default():
            yield

    def _run(self, fetches):
        """
        Core Logic
        """
        s = Strategy.create(self._graph, self._resource_spec, strategy_name=None)
        Runner(s).run(fetches)

    # def run(self, fetches, feed_dict):
    #     """
    #     AutoDist().run(fetches, feed_dict)
    #     """
    #     return self._run(fetches, feed_dict=fetches)

    # def run(self, step_fn, *args, **kwargs):
    #     """
    #     AutoDist().run(step_fn, *args, **kwargs)
    #     """
    #     fetches = step_fn(*args, **kwargs)
    #     return self._run(fetches)

    def function(self, fn):
        """
        TODO(peng.wu)
        @AutoDist().function
        def step_fn(*args, **kwargs):
            ...

        step_fn(*args, **kwargs)
        """
        def wrapper(*args, **kwargs):
            # get the single-node graph
            # (?) tf.funtion helps you identify the fetches
            # self._run(fetches)
            pass
        return wrapper














