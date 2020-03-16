# FAQ (WIP)

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

4. **Compare AutoDist with distributed TensorFlow or Horovod?**

5. **Will there be Kubernetes integration?**

6. **Will AutoDist support PyTorch?**

7. **Does AutoDist support model parallelism?**

8. **Does AutoDist speed up single-node execution?** 
