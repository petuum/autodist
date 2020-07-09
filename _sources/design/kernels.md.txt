# Graph Transformation Kernels

AutoDist backend offers a library of primitive graph rewriting [kernels](../api/autodist.kernel.graph_transformer). 
Each sub-expression in the [strategy](../usage/proto_docgen.md) is mapped to a kernel type with specific configurations. 
Each type of kernel defines how to rewrite the graph from its current state to the next state, 
which manifests the corresponding semantics in the sub-expression. 
By designing each local kernel at the right level of abstraction, 
they can be flexibly composed to alter the graph based on a strategy.
 
* `Partitioner` reads node-level partitioning configurations from a strategy for each variable. 
It splits the variable across given axes into multiple smaller variables, 
as well as its gradients and subgraphs corresponded to its state-updating operations. 
However, it does not split the operations that consume the original variable – 
which will instead consume the value re-concatenated from all partitions. 
Without allowing recursive partition, each of the new smaller variables has its own node config 
(generated at strategy generation time), 
will be added into a variable as an independent variable identity in the following transformation steps. 
* `Replicator` reads graph-level configuration from a strategy. 
It replicates the original graph onto target devices. 
Unless overridden by other graphtransformation kernels or by developers, 
the replicated operations or variables have their placement same with the target replication destination in a strategy. 
* `Synchronizer` reads node-level configurations for each of the original and partitioned variable in a variable, 
where `Compressor` as its component rather than graph-transformation kernel is responsible for gradient encoding and 
decoding therein. Depending on the synchronizer type, it rewrites the graph: 
    1. either to share a variable on a destination device across replicas ((Reduce, Broadcast) synchronizer) 
with specified staleness in a strategy, 
    2. or to synchronize states of replicated variables via collective communication (AllReduce synchronizer) following specified device structures in a strategy. 
Moreover, the kernel implementations are dispatched to handle either dense or sparse cases.

Besides existing kernels, the system design allows convenient extensions to emerging synchronization optimizations, 
by allocating new dimensions in the representation and introducing corresponded graph-rewriting kernels in the backend.