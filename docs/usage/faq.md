# FAQ

1. **Does AutoDist support heterogeneous clusters?**
    > Yes, it's as straight-forward as the setup on a homogeneous cluster.
    > You can refer to [Train on Multiple Nodes](tutorials/multi-node.md).

2. **What device types does AutoDist support?**
    > Currently AutoDist only supports strategies on `CPU` and `GPU` to be configured in
    > the [resource specification]. But we are still actively improving this.

3. **Why doesn't AutoDist support Eager mode?**
    > AutoDist's design is based on transforming / compiling the computation graph.
    > Although eager mode can still utilize `tf.function` for graph execution,
    > it requires more effort to deal with variable states under
    > eager mode to fit in the AutoDist stack, so we de-prioritized it.
    > We might support it in the future, though!

4. **Will AutoDist support PyTorch?**
    > The current architecture of AutoDist is based on graph transformations.
    > At this time, PyTorch does not offer a good way to get a static computational graph directly
    > (except TorchScript, which is still in an early stage),
    > so we thus do not have plan to integrate the stack with PyTorch in the near future.

5. **Will there be Kubernetes integration?**
    > AutoDist is integrated with Kubernetes internally in a Petuum closed-source product.
    > Even so, one can still start with our [Running on Docker](tutorials/docker.md)
    > instructions for containerization and orchestration.

6. **Does AutoDist support model parallelism?**
    > Not yet, but with the ability of composing a strategy, AutoDist is able to
    > support defining the configuration of how to partition an operation on non-batch dimension
    > as part of the distributed [strategy](proto_docgen.md),
    > together with proper graph-transformation [kernels](../api/autodist.kernel.graph_transformer).

