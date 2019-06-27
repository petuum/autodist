

class Runner:

    def __init__(self, strategy):
        pass

    def run(self, graph, fetches):
        """
        Be able to support switch between master process and worker process in order to support between graph
        """
        # Not necessarily support switch by if condition
        if option == 1: # if master
            # launch processes

        if option == 2: # if worker
            g = strategy.get_distributed_graph(graph)
            with g.as_default():
                Session(g).run(fetches)

