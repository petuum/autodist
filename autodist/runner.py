from tensorflow.python.client import session


class Runner:

    def __init__(self, strategy):
        self._strategy = strategy

    def run(self, graph, fetches):
        """
        Be able to support switch between master process and worker process in order to support between graph
        """

        option = 1 or 2
        # Not necessarily support switch by if condition
        if option == 1:  # if master
            # launch processes
            pass

        if option == 2:  # if worker
            g = self.strategy.get_distributed_graph(graph)
            with g.as_default():
                session.Session(g).run(fetches)
