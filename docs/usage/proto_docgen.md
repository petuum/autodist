# Strategy as ProtoBuf Message
<a name="top"></a>

AutoDist uses Protocol Buffer to standardize strategy representation and its configurations.


- [autodist/proto/strategy.proto](#autodist/proto/strategy.proto)
    - [Strategy](#autodist.proto.Strategy)
    - [Strategy.GraphConfig](#autodist.proto.Strategy.GraphConfig)
    - [Strategy.Node](#autodist.proto.Strategy.Node)
  
- [autodist/proto/synchronizers.proto](#autodist/proto/synchronizers.proto)
    - [AllReduceSynchronizer](#autodist.proto.AllReduceSynchronizer)
    - [PSSynchronizer](#autodist.proto.PSSynchronizer)
  
    - [AllReduceSynchronizer.Compressor](#autodist.proto.AllReduceSynchronizer.Compressor)
    - [AllReduceSynchronizer.Spec](#autodist.proto.AllReduceSynchronizer.Spec)
  
- [Scalar Value Types](#scalar-value-types)



<a name="autodist/proto/strategy.proto"></a>
<p align="right"><a href="#top">Top</a></p>

## autodist/proto/strategy.proto
AutoDist distributed strategy messages.

Represents how to distribute a TensorFlow computational graph.


<a name="autodist.proto.Strategy"></a>

### Strategy
Represents the strategy the AutoDist backend will implement.


| Field | Type | Description |
| ----- | ---- | ----------- |
| id | [string](#string) | unique strategy identifier |
| path | [string](#string) | optional serialized strategy message temp path |
| node_config | [Strategy.Node](#autodist.proto.Strategy.Node) | configuration of some individual nodes of the computational graph |
| graph_config | [Strategy.GraphConfig](#autodist.proto.Strategy.GraphConfig) | configuration of the computational graph as a whole |






<a name="autodist.proto.Strategy.GraphConfig"></a>

### Strategy.GraphConfig
Represents the configuration of the graph as a whole.

Based on the list of replicas, the AutoDist backend does
a combination of in-graph and between-graph distribution.


| Field | Type | Description |
| ----- | ---- | ----------- |
| replicas | [string](#string) | the number of batch-splitting/data-parallel replicas |






<a name="autodist.proto.Strategy.Node"></a>

### Strategy.Node
Represents the configuration of an individual node in the graph.

Right now, these nodes are just variables in the graph, so the only
information they contain is how to synchronize the variable's gradients.

In the future, for node partitioning, these could be any node in the graph.
In that case, they would also have more logic for partitioning the op.


| Field | Type | Description |
| ----- | ---- | ----------- |
| var_name | [string](#string) | variable name |
| PSSynchronizer | [PSSynchronizer](#autodist.proto.PSSynchronizer) | One of a synchronizer to choose |
| AllReduceSynchronizer | [AllReduceSynchronizer](#autodist.proto.AllReduceSynchronizer) | One of a synchronizer to choose |





 <!-- end messages -->

 <!-- end enums -->

 <!-- end HasExtensions -->

 <!-- end services -->



<a name="autodist/proto/synchronizers.proto"></a>
<p align="right"><a href="#top">Top</a></p>

## autodist/proto/synchronizers.proto
AutoDist synchronization messages.


<a name="autodist.proto.AllReduceSynchronizer"></a>

### AllReduceSynchronizer
Synchronization using AllReduce.


| Field | Type | Description |
| ----- | ---- | ----------- |
| spec | [AllReduceSynchronizer.Spec](#autodist.proto.AllReduceSynchronizer.Spec) | Specification for collective communication |
| compressor | [AllReduceSynchronizer.Compressor](#autodist.proto.AllReduceSynchronizer.Compressor) | One of the compressors to choose |
| chunk_size | [int32](#int32) | Size threshold for batching all-reduce communications |






<a name="autodist.proto.PSSynchronizer"></a>

### PSSynchronizer
Synchronization using a Parameter Server.


| Field | Type | Description |
| ----- | ---- | ----------- |
| reduction_destinations | [string](#string) | Parameter Servers to use |
| local_replication | [bool](#bool) | Whether to create local proxies of each PS variable |
| sync | [bool](#bool) | Whether to sync gradients across between-graph replications |
| staleness | [int32](#int32) | Staleness |





 <!-- end messages -->


<a name="autodist.proto.AllReduceSynchronizer.Compressor"></a>

### AllReduceSynchronizer.Compressor
Which gradient compression method to use

| Name | Number | Description |
| ---- | ------ | ----------- |
| NoneCompressor | 0 | No compression |
| HorovodCompressor | 1 | Horovod's Compression |
| HorovodCompressorEF | 2 | Horovod's Compression but with Error Feedback. |
| PowerSGDCompressor | 3 | PowerSGD compression algorithm (arxiv.org/abs/1905.13727) |



<a name="autodist.proto.AllReduceSynchronizer.Spec"></a>

### AllReduceSynchronizer.Spec
Which communication method to use

| Name | Number | Description |
| ---- | ------ | ----------- |
| AUTO | 0 | Runtime's automatic choices |
| NCCL | 1 | Use ncclAllReduce for all-reduce, and ring algorithms for all-gather |
| RING | 2 | TensorFlow's ring algorithms for all-reduce and all-gather |


 <!-- end enums -->

 <!-- end HasExtensions -->

 <!-- end services -->



## Scalar Value Types

| .proto Type | Notes | C++ | Python |
| ----------- | ----- | --- | ------ |
| <a name="double" /> double |  | double | float |
| <a name="float" /> float |  | float | float |
| <a name="int32" /> int32 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint32 instead. | int32 | int |
| <a name="int64" /> int64 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint64 instead. | int64 | int/long |
| <a name="uint32" /> uint32 | Uses variable-length encoding. | uint32 | int/long |
| <a name="uint64" /> uint64 | Uses variable-length encoding. | uint64 | int/long |
| <a name="sint32" /> sint32 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int32s. | int32 | int |
| <a name="sint64" /> sint64 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int64s. | int64 | int/long |
| <a name="fixed32" /> fixed32 | Always four bytes. More efficient than uint32 if values are often greater than 2^28. | uint32 | int |
| <a name="fixed64" /> fixed64 | Always eight bytes. More efficient than uint64 if values are often greater than 2^56. | uint64 | int/long |
| <a name="sfixed32" /> sfixed32 | Always four bytes. | int32 | int |
| <a name="sfixed64" /> sfixed64 | Always eight bytes. | int64 | int/long |
| <a name="bool" /> bool |  | bool | boolean |
| <a name="string" /> string | A string must always contain UTF-8 encoded or 7-bit ASCII text. | string | str/unicode |
| <a name="bytes" /> bytes | May contain any arbitrary sequence of bytes. | string | str |
