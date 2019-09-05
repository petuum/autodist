"""The information of TensorFlow operations for AutoDist."""

# The 0-th index of an update op always points to the target resource
UPDATE_OP_VAR_POS = 0

DENSE_VAR_UPDATE_OP_TYPES = {
    # map from ops to the index of the update in op.inputs
    "ResourceApplyAdaMax": (8,),
    "ResourceApplyAdadelta": (6,),
    "ResourceApplyAdagrad": (3,),
    "ResourceApplyAdagradDA": (3,),
    "ResourceApplyAdam": (9,),
    "ResourceApplyAdamWithAmsgrad": (10,),
    "ResourceApplyAddSign": (6,),
    "ResourceApplyCenteredRMSProp": (8,),
    "ResourceApplyFtrl": (3,),
    "ResourceApplyFtrlV2": (3,),
    "ResourceApplyGradientDescent": (2,),
    "ResourceApplyKerasMomentum": (3,),
    "ResourceApplyMomentum": (3,),
    "ResourceApplyPowerSign": (6,),
    "ResourceApplyProximalAdagrad": (5,),
    "ResourceApplyProximalGradientDescent": (4,),
    "ResourceApplyRMSProp": (7,),
    "AssignAddVariableOp": (1,),
    "AssignSubVariableOp": (1,),
}

# For sparse operations:
# First: indices
# Second: updates
SPARSE_VAR_UPDATE_OP_TYPES = {
    "ResourceScatterUpdate": (1, 2,),
    "ResourceScatterAdd": (1, 2,),
    "ResourceScatterSub": (1, 2,),
    "ResourceScatterMul": (1, 2,),
    "ResourceScatterDiv": (1, 2,),
    "ResourceScatterMax": (1, 2,),
    "ResourceScatterMin": (1, 2,),
    "ResourceScatterNdAdd": (1, 2,),
    "ResourceScatterNdSub": (1, 2,),
    "ResourceScatterNdUpdate": (1, 2,),
    "ResourceSparseApplyAdadelta": (7, 6,),
    "ResourceSparseApplyAdagrad": (4, 3,),
    "ResourceSparseApplyAdagradDA": (4, 3,),
    "ResourceSparseApplyCenteredRMSProp": (9, 8,),
    "ResourceSparseApplyFtrl": (4, 3,),
    "ResourceSparseApplyFtrlV2": (4, 3,),
    "ResourceSparseApplyKerasMomentum": (4, 3,),
    "ResourceSparseApplyMomentum": (4, 3,),
    "ResourceSparseApplyProximalAdagrad": (6, 5,),
    "ResourceSparseApplyProximalGradientDescent": (5, 4,),
    "ResourceSparseApplyRMSProp": (8, 7,),
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
