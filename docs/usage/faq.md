# FAQ

1. **Does AutoDist support heterogeneous clusters?**
    > Yes, it's as straight-forward as the setup on homogeneous cluster. 
    > You could refer to [Train on Multiple Nodes](tutorials/multi-node.md)
    > or [Train with Docker](tutorials/docker.md) to get started.

2. **What device types does AutoDist support?**
    > Currently AutoDist only supports strategies on `CPU` and `GPU` to be configured in
    > the [resource specification]. But AutoDist is still actively improving this.

3. **Why not support Eager mode?**
    > AutoDist design is based on transforming / compiling computation graph.
    > Although eager mode can still utilize `tf.function` for graph execution,
    > it still requires more efforts to deal with variable states under
    > eager mode to fit in the AutoDist stack, so we de-prioritized it 
    > but maybe support it in the future.

4. **Will there be Kubernetes integration?**
    > AutoDist is integrated with Kubernetes internally in a Petuum closed-source product Orchestra.

5. **Does AutoDist support model parallelism?**
    > Not yet, but with the ability of composing a strategy, AutoDist is able to 
    > support defining the configuration of how to partition an operation on non-batch dimension 
    > as part of the distributed [strategy](proto_docgen.md), 
    > together with proper graph-transformation [kernels](../api/autodist.kernel.graph_transformer).  

6. **Will AutoDist support PyTorch?**
    > Since the architecture of AutoDist is based on graph transforming, while PyTorch does not offer
    > static computational graph directly except TorchScript which is still in an early stage, 
    > AutoDist thus does not have plan to integrate the stack with PyTorch in the near future.
