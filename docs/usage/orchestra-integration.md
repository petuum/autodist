# Orchestra Integration

## What is Orchestra?

[Orchestra](https://petuum.com/platform/) is a machine learning platform built by Petuum Inc. which includes resource management, job control, etc. AutoDist will be the distributed training backend of this platform.

## AutoDist in Orchestra

### Data Path and Resource Path

In Orchestra, data path and resource file path will be passed to the AutoDist code during the runtime as environment variables `SYS_DATA_PATH` and `SYS_RESOURCE_PATH`. 
For AutoDist code developer under Orchestra, they can set data path as:

```
data_path = os.environ.get("SYS_DATA_PATH")
resource_file = os.environ.get("SYS_RESOURCE_PATH")
```

