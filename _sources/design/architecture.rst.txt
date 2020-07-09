Architecture
=============

We recommend that you read AutoDist's :doc:`rationale` before reading this doc.
This document broadly describes the architecture of AutoDist and also goes into
some details about the implementation of various features.

Overview
--------

AutoDist was designed with two goals in mind:

1. Have a flexible backend capable of distributing graphs according to an arbitrary :code:`Strategy`
2. Have an incredibly easy to use interface

This means that the code is broadly separated into two parts: Strategy Generation and Graph Transformation.


General Workflow
----------------

.. image:: images/autodist-arch.png
  :align: center
  :scale: 40
  :alt: Architecture Diagram

The general workflow of AutoDist is described in the image.
Users provide a `TensorFlow Graph <https://www.tensorflow.org/api_docs/python/tf/Graph>`_ (:code:`tf.Graph`)
and a :doc:`resource specification<../../usage/tutorials/multi-node>` (:code:`ResourceSpec`).
From this, a :code:`StrategyBuilder` analyzes both the :code:`tf.Graph` and the :code:`ResourceSpec` and generates a :code:`Strategy`,
a Protobuf representation of how to distribute the graph.

This :code:`Strategy` is then passed to the :code:`GraphTransformer`, the "backend" of AutoDist that is responsible
for distributing the user's graph according to the given strategy. This :code:`GraphTransformer` will alter the
original :code:`tf.Graph` on a per-variable basis, adding the necessary TensorFlow
`operations <https://www.tensorflow.org/api_docs/python/tf/Operation>`_ (ops)
as defined in the :code:`Strategy`.

After the transformed graph has been built, it is sent back to TensorFlow for execution.

Some things to note:

- Currently, the strategy generation happens on one node. This strategy is then sent out to every node,
  and graph transformation happens on every node (it is much easier to send a small strategy file over the
  network than it is to send an entire transformed graph).
- When given multiple nodes with multiple devices (e.g. GPUs) each, we do in-graph replication within each worker
  and between-graph between workers. That means that each worker will have one :code:`tf.Graph` that contains
  the replicated graph for each GPU on that worker, but will not contain graphs for devices outside that worker.
- Currently, AutoDist only supports data-parallel distribution; that is, distribution along the batch dimension. There
  are plans to support model parallelism by providing the ability to partition ops, but that has not been implemented
  yet.

StrategyBuilders
----------------

Each :code:`StrategyBuilder` describes a method for synchronizing each trainable
variable in the graph. There are a few different :code:`StrategyBuilders` listed
:doc:`here<../../api/autodist.strategy>`, with each doing different things.
For more details, you could refer to ":doc:`Choose Strategy Builders<../../usage/tutorials/choose-strategy>`" or
":doc:`Customize Strategy Builders"<../../usage/tutorials/customize-strategy>`"

Essentially, Strategy Builders are just choosing a sample from the strategy space defined by the :code:`Strategy`
protobuf. Theoretically, every possible strategy representable in a :code:`Strategy` object should be able to be
distributed, with just a few limitations: we currently cannot partition variables that are part of a control flow,
and all-reduce does not work if there is only one machine with one GPU.


GraphTransformer
----------------

The :code:`GraphTransformer` currently has 3 phases:

1. Variable Partitioning,
2. Op Replication,
3. and Synchronization

The :code:`Partitioner` shards each variable (according to the Strategy) along the first dimension into equisized
partitions. The :code:`Replicator` does in-graph replication of the graph, duplicating it according to the
number of devices in that worker (e.g., a worker with 2 GPUs will have two identical subgraphs,
:code:`AutoDist-Replica-1/...` and :code:`AutoDist-Replica-2/...`). Lastly, the synchronizers handle adding the ops for
both in-graph and between-graph synchronization, again, according to the strategy.

For more information, please refer to :doc:`Graph Transformation Kernels<kernels>`

Networking
----------

The networking side of this is generally handled by the :code:`Cluster` and :code:`Coordinator` objects. They are
responsible for setting up SSH connections from the chief to each worker, copying any necessary files, and starting
a :code:`tf.Server` on each worker so they are ready to run the distributed graph.
