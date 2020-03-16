
# Choose Strategy Builders

As in previous tutorials, the only required argument from 

<code>[AutoDist(resource_spec_file, strategy_builder=None)](../../api/autodist.autodist)</code>

is `resource_spec_file`, the path of a yaml file describing resource specifications.
In the meanwhile, AutoDist also offers flexibility for provide built-in or customized `strategy_builder`
as described below. If `strategy_builder=None`, AutoDist will automatically choose the default strategy builder for you.

One might ask there: what is a strategy builder? 
Intuitively speaking, a *strategy* in AutoDist is a representation on how to distribute the computational graph,
while a *strategy builder* creates that representation. 
For more technical details, please refer to 
[Architecture](../../design/architecture) or 
[Strategy Protocol](../proto_docgen.md) or 
[Developer API Reference](../../api/autodist.strategy.base).

New users of AutoDist can start from choosing one of 
the built-in strategy builders from `autodist.strategy`. 
To know more about different built-in strategy builders, and 
their configurations or pros & cons respectively, please see 
the reference page for [built-in strategy builders](../../api/autodist.strategy).

```python
from autodist.strategy import PSLoadBalancing
from autodist import AutoDist
autodist = AutoDist(
    resource_spec_file='resource_spec.yml', 
    strategy_builder=PSLoadBalancing(sync=True)
)
```

For advanced users, one can [customize a strategy](customize-strategy.md) besides the built-ins.

