# Customize a Strategy Builder

## Closer Look on Strategy 

In AutoDist, a strategy is a representation 
to instruct AutoDist to transform a single-node computational graph to
be a distributed one. 

For more technical details,
one can refer to the [definition page](../proto_docgen.md) 
of strategy Protocol Buffer message.

Below is an intuitive example of a strategy for a computational graph
with two variables `W:0` and `b:0`.  Each variable is assigned
with a corresponding configured synchronizer with in the 
<code>[node_config](../proto_docgen.html#autodist.proto.Strategy.Node)</code> section; 
while the <code>[graph_config](../proto_docgen.html#autodist.proto.Strategy.GraphConfig)</code>
in the below example is configured to instructs the data-parallel `replicas` of the whole graph.

```
node_config {
  var_name: "W:0"
  PSSynchronizer {
    reduction_destinations: "localhost:CPU:0"
    sync: true
  }
}
node_config {
  var_name: "b:0"
  AllReduceSynchronizer {
    chunk_size: 128
  }
}
graph_config {
  replicas: 
    [
        "10.21.1.24:GPU:0",
        "10.21.1.24:GPU:1",
        "10.21.1.25:GPU:0",
        "10.21.1.25:GPU:1"
    ]
}
```

## Build a Strategy

With the understanding of the strategy representation above,
you can create your own customized strategy builder 
just like the [built-in ones](choose-strategy.md).
The customized strategy builder needs to follow the 
<code>[StrategyBuilder](../../api/api/autodist.strategy.base.html#autodist.strategy.base.StrategyBuilder)</code> abstraction,
with a required interface `build`,
which takes a <code>[GraphItem](../../api/autodist.graph_item)</code> a
<code>[ResourceSpec](../../api/autodist.resource_spec)</code> and returns a
<code>[Strategy](../../api/api/autodist.strategy.base.html#autodist.strategy.base.Strategy)</code>.

```python
def build(self, graph_item: GraphItem, resource_spec: ResourceSpec) -> Strategy:
```

* Create a strategy representation wrapper object.
```python
from autodist.strategy.base import Strategy
strategy = Strategy()
```
* Set configurations for the whole graph. For example you can utilize the `resource_spec` properties to list 
all GPU devices for your data parallelism.
```python
strategy.graph_config.replicas.extend([k for k, v in resource_spec.gpu_devices])
```
* Before configuring nodes, for example, you can utilize the `graph_item` methods to list 
all variables that you want to configure; while utilize the `resource_spec` properties to
prepare for where to put the variable states.
```python
variables = graph_item.get_trainable_variables()
reduction_devices = [k for k, _ in resource_spec.cpu_devices][0:1]
```
* Set configurations for variable nodes. Besides the below example,
there are various choices of configurations listed here 
<code>[strategy_pb2.Strategy.Node](../proto_docgen.html#autodist.proto.Strategy.Node)</code> section.  
```python
from autodist.proto import strategy_pb2
node_config = []
for var in variables:
    node = strategy_pb2.Strategy.Node()
    node.var_name = var.name
    node.PSSynchronizer.reduction_destinations.extend(reduction_devices)
    node.PSSynchronizer.local_replication = False
    node.PSSynchronizer.sync = True
    node.PSSynchronizer.staleness = 2
    node_config.append(node)

strategy.node_config.extend(node_config)
```

Congratulations! You have successfully created your first strategy builder. AutoDist is flexible for developers
to create different types of strategies based on the configuration spaces, and also possible for auto-learning a strategy.
For more developments on strategies, you could refer to other [built-in strategy builders](../../api/autodist.strategy)
or our development reference to invent your own [kernels](../../api/autodist.kernel).
