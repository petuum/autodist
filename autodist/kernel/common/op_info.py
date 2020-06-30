# Copyright 2020 Petuum. All Rights Reserved.
#
# It includes the derived work based on:
# https://github.com/snuspl/parallax
# Copyright (C) 2018 Seoul National University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The information of TensorFlow operations for AutoDist."""

# The 0-th index of an update op always points to the target resource
UPDATE_OP_VAR_POS = 0

DENSE_VAR_UPDATE_OP_TYPES = {
    # map from ops to the index of the update in op.inputs
    "ResourceApplyAdaMax": (8,),
    "ResourceApplyAdadelta": (6,),
    "ResourceApplyAdagrad": (3,),
    "ResourceApplyAdagradV2": (3,),
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
    "AssignAdd": (1,),
    "AssignSub": (1,),
    # TODO: Verify Assign as update op
    "AssignVariableOp": (1,),
    "Assign": (1,),
    #
    "ApplyAdaMax": (8,),
    "ApplyAdadelta": (6,),
    "ApplyAdagrad": (3,),
    "ApplyAdagradDA": (3,),
    "ApplyAdagradV2": (3,),
    "ApplyAdam": (9,),
    "ApplyAddSign": (6,),
    "ApplyCenteredRMSProp": (8,),
    "ApplyFtrl": (3,),
    "ApplyFtrlV2": (3,),
    "ApplyGradientDescent": (2,),
    "ApplyMomentum": (3,),
    "ApplyPowerSign": (6,),
    "ApplyProximalAdagrad": (5,),
    "ApplyProximalGradientDescent": (4,),
    "ApplyRMSProp": (7,),
}

# For sparse operations:
# First: indices
# Second: updates
SPARSE_VAR_UPDATE_OP_TYPES = {
    "ResourceScatterAdd": (1, 2,),
    "ResourceScatterSub": (1, 2,),
    "ResourceScatterMul": (1, 2,),
    "ResourceScatterDiv": (1, 2,),
    "ResourceScatterMax": (1, 2,),
    "ResourceScatterMin": (1, 2,),
    "ResourceScatterNdAdd": (1, 2,),
    "ResourceScatterNdSub": (1, 2,),
    "ResourceScatterNdUpdate": (1, 2,),
    "ResourceScatterUpdate": (1, 2,),
    "ResourceSparseApplyAdadelta": (7, 6,),
    "ResourceSparseApplyAdagrad": (4, 3,),
    "ResourceSparseApplyAdagradV2": (4, 3,),
    "ResourceSparseApplyAdagradDA": (4, 3,),
    "ResourceSparseApplyCenteredRMSProp": (9, 8,),
    "ResourceSparseApplyFtrl": (4, 3,),
    "ResourceSparseApplyFtrlV2": (4, 3,),
    "ResourceSparseApplyKerasMomentum": (4, 3,),
    "ResourceSparseApplyMomentum": (4, 3,),
    "ResourceSparseApplyProximalAdagrad": (6, 5,),
    "ResourceSparseApplyProximalGradientDescent": (5, 4,),
    "ResourceSparseApplyRMSProp": (8, 7,),
    "ScatterAdd": (1, 2,),
    "ScatterSub": (1, 2,),
    "ScatterMul": (1, 2,),
    "ScatterDiv": (1, 2,),
    "ScatterMax": (1, 2,),
    "ScatterMin": (1, 2,),
    "ScatterNdAdd": (1, 2,),
    "ScatterNdSub": (1, 2,),
    "ScatterNdUpdate": (1, 2,),
    "ScatterUpdate": (1, 2,),
    "SparseApplyAdadelta": (7, 6,),
    "SparseApplyAdagrad": (4, 3,),
    "SparseApplyAdagradV2": (4, 3,),
    "SparseApplyAdagradDA": (4, 3,),
    "SparseApplyCenteredRMSProp": (9, 8,),
    "SparseApplyFtrl": (4, 3,),
    "SparseApplyFtrlV2": (4, 3,),
    "SparseApplyMomentum": (4, 3,),
    "SparseApplyProximalAdagrad": (6, 5,),
    "SparseApplyProximalGradientDescent": (5, 4,),
    "SparseApplyRMSProp": (8, 7,),
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

CONTROL_FLOW_OPS = {
    "Abort",
    "ControlTrigger",
    "Enter",
    "Exit",
    "LoopCond",
    "Merge",
    "NextIteration",
    "NoOp",
    "RefEnter",
    "RefExit",
    "RefMerge",
    "RefNextIteration",
    "RefSelect",
    "RefSwitch",
    "Switch",
}

MUTABLE_STATE_OP_DIRECT_CONSUMER_OPS = {
    # VarHandleOp
    'VarIsInitializedOp',
    'AssignVariableOp',
    'ReadVariableOp',
    'ResourceGather',
    # Variable & VariableV2
    'Assign',
    'Identity',
    # TODO(trevin): strict checking for partitioner deletions
}
