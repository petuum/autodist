"""The information of TensorFlow operations for AutoDist."""

# TODO: update all the mappings for update inference.
DENSE_VAR_UPDATE_OP_TYPES = {
    # map from ops to the index of the update in op.inputs
    "ResourceApplyGradientDescent": (2, 0),
    "ResourceApplyAdam": (9, 0),
    "AssignAddVariableOp": (1, 0),
    "ApplyGradientDescent": 2,
    "ApplyProximalGradientDescent": 4,
    "ApplyAdadelta": 6,
    "ApplyAdagrad": 3,
    "ApplyProximalAdagrad": 5,
    "ApplyAdagradDA": 3,
    "ApplyFtrl": 3,
    "ApplyMomentum": 3,
    "ApplyAdam": 9,
    "ApplyRMSProp": 7,
    "ApplyCenteredRMSProp": 8,
    "AssignAdd": 1,
    "AssignSub": 1
}

# For sparse operations:
# First: indices
# Second: updates
# third: resources
SPARSE_VAR_UPDATE_OP_TYPES = {
    "ResourceScatterUpdate": (1, 2, 0),
    "ResourceScatterAdd": (1, 2, 0),
    "ResourceScatterSub": (1, 2, 0),
    "ResourceScatterMul": (1, 2, 0),
    "ResourceScatterDiv": (1, 2, 0),
    "ScatterUpdate": (1, 2),
    "ScatterAdd": (1, 2),
    "ScatterSub": (1, 2),
    "ScatterMul": (1, 2),
    "ScatterDiv": (1, 2),
    "SparseApplyAdagrad": (4, 3)
}

UNSTAGE_OP_TYPES = ["Unstage"]

STAGE_OP_TYPES = ["Stage"]

QUEUE_OP_TYPES = [
    "RandomShuffleQueue", "RandomShuffleQueueV2",
    "FIFOQueue", "FIFOQueueV2",
    "PaddingFIFOQueue", "PaddingFIFOQueueV2",
    "PriorityQueue", "PriorityQueueV2"
]

DEQUEUE_OP_TYPES = [
    "ReaderRead", "ReaderReadV2",
    "ReaderReadUpTo", "ReaderReadUpToV2",
    "ReaderRestoreState", "ReaderRestoreStateV2",
    "QueueDequeueMany", "QueueDequeueManyV2",
    "QueueDequeue", "QueueDequeueV2",
    "QueueDequeueUpTo", "QueueDequeueUpToV2"
]

ITERATOR_OP_TYPES = [
    "Iterator", "IteratorV2", "OneShotIterator"
]

CPU_ONLY_TYPES = ["OneShotIterator"]

MUTABLE_STATE_OPS = {
    "Variable",
    "VariableV2",
    "AutoReloadVariable",
    "MutableHashTable",
    "MutableHashTableV2",
    "MutableHashTableOfTensors",
    "MutableHashTableOfTensorsV2",
    "MutableDenseHashTable",
    "MutableDenseHashTableV2",
    "VarHandleOp",
    "BoostedTreesEnsembleResourceHandleOp"
}
