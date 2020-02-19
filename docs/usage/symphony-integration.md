# AutoDist in Symphony

## Symphony

Symphony is a machine learning platform which includes resource management, job control, etc. AutoDist will be the distributed training backend of this platform.

## AutoDist in Symphony

### Data Path and Resource Path

In Symphony, data path and resource file path will be passed to the AutoDist code during the runtime as environment variables `SYS_DATA_PATH` and `SYS_RESOURCE_PATH`. 
For AutoDist code developer under Symphony, they can set data path as:

```
data_path = os.environ.get("SYS_DATA_PATH")
resource_file = os.environ.get("SYS_RESOURCE_PATH")
```

