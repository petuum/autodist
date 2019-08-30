"""The information of TensorFlow operations for AutoDist."""

DENSE_VAR_UPDATE_OP_TYPES = {
    # map from ops to the index of the update in op.inputs
    "ResourceApplyAdaMax": (8, 0),
    "ResourceApplyAdadelta": (6, 0),
    "ResourceApplyAdagrad": (3, 0),
    "ResourceApplyAdagradDA": (3, 0),
    "ResourceApplyAdam": (9, 0),
    "ResourceApplyAdamWithAmsgrad": (10, 0),
    "ResourceApplyAddSign": (6, 0),
    "ResourceApplyCenteredRMSProp": (8, 0),
    "ResourceApplyFtrl": (3, 0),
    "ResourceApplyFtrlV2": (3, 0),
    "ResourceApplyGradientDescent": (2, 0),
    "ResourceApplyKerasMomentum": (3, 0),
    "ResourceApplyMomentum": (3, 0),
    "ResourceApplyPowerSign": (6, 0),
    "ResourceApplyProximalAdagrad": (5, 0),
    "ResourceApplyProximalGradientDescent": (4, 0),
    "ResourceApplyRMSProp": (7, 0),
    "AssignAddVariableOp": (1, 0),
    "AssignSubVariableOp": (1, 0),
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
    "ResourceScatterMax": (1, 2, 0),
    "ResourceScatterMin": (1, 2, 0),
    "ResourceScatterNdAdd": (1, 2, 0),
    "ResourceScatterNdSub": (1, 2, 0),
    "ResourceScatterNdUpdate": (1, 2, 0),
    "ResourceSparseApplyAdadelta": (7, 6, 0),
    "ResourceSparseApplyAdagrad": (4, 3, 0),
    "ResourceSparseApplyAdagradDA": (4, 3, 0),
    "ResourceSparseApplyCenteredRMSProp": (9, 8, 0),
    "ResourceSparseApplyFtrl": (4, 3, 0),
    "ResourceSparseApplyFtrlV2": (4, 3, 0),
    "ResourceSparseApplyKerasMomentum": (4, 3, 0),
    "ResourceSparseApplyMomentum": (4, 3, 0),
    "ResourceSparseApplyProximalAdagrad": (6, 5, 0),
    "ResourceSparseApplyProximalGradientDescent": (5, 4, 0),
    "ResourceSparseApplyRMSProp": (8, 7, 0),
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
    "Iterator",
    "IteratorV2",
    "OneShotIterator",
    "AnonymousIterator",
    "AnonymousIteratorV2",
    "MultiDeviceIterator",
    "IteratorFromStringHandle",
    "IteratorFromStringHandleV2",
    "MultiDeviceIteratorFromStringHandle",
]

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
