# AutoDist in Symphony

## Symphony

Symphony is a machine learning platform which includes resource management, job control, etc. AutoDist will be the distributed training backend of this platform.

## AutoDist in Symphony

### Data Path

In Symphony, data path will be passed to the AutoDist code during the runtime as a environment variable `SYS_DATA_PATH`. 
For AutoDist code developer under Symphony, they can set data path as:

```
data_path = os.environ.get("SYS_DATA_PATH")
```

### (TODO): resource spec path

This file will also be generated from Symphony.
