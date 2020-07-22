# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# It includes the derived work based on:
# https://github.com/tensorflow/tensorflow
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""AutoDist Saver."""
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as tf_saver

import autodist.autodist
from autodist.graph_item import get_default_graph_item


class Saver(tf_saver.Saver):
    """A wrapper upon tf.compat.v1.train.Saver."""

    def __init__(self,
                 var_list=None,
                 reshape=False,
                 sharded=False,
                 max_to_keep=5,
                 keep_checkpoint_every_n_hours=10000.0,
                 name=None,
                 restore_sequentially=False,
                 saver_def=None,
                 builder=None,
                 defer_build=False,
                 allow_empty=False,
                 write_version=saver_pb2.SaverDef.V2,
                 pad_step_number=False,
                 save_relative_paths=False,
                 filename=None):
        # pylint: disable=too-many-arguments, too-many-locals
        """
        Saver for AutoDist.

        This saver saves the variables that maps to the *original*, *user-declared*, *single-node* graph,
        instead of the transformed graph. Hence, the saved variables can be loaded either by the original
        user code (for resuming single-node training or inference), or by the AutoDist-distributed code (to
        resume distributed training). Differently, AutoDist saver saves the meta_graph_def that maps to the
        *AutoDist transformed* graph (TODO).

        AutoDist saver implements this by writing/loading the contents that maps to the master_replica
        (default to replica_index=0) of the transformed graph.

        Refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/saver.py
        for a detailed explanation of the signature.
        """
        # TODO(Hao): support constructing saver after AutoDist graph transformation
        _autodist = autodist.autodist.get_default_autodist()
        if _autodist.is_built():
            raise ValueError('Saver must be used before create_distributed_session().')

        # A saver will append relevant save/restore ops to all variables in var_list, i.e. one saver
        # maps to all variables, and encloses them under a "saver" scope.
        super(Saver, self).__init__(var_list=var_list,
                                    reshape=reshape,
                                    sharded=sharded,
                                    max_to_keep=max_to_keep,
                                    keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
                                    name=name,
                                    restore_sequentially=restore_sequentially,
                                    saver_def=saver_def,
                                    builder=builder,
                                    defer_build=defer_build,
                                    allow_empty=allow_empty,
                                    write_version=write_version,
                                    pad_step_number=pad_step_number,
                                    save_relative_paths=save_relative_paths,
                                    filename=filename)

        # Note: tensorflow does not add user-declared saver to collections, so have to track it in info.
        item = get_default_graph_item()
        new_saver_def = saver_pb2.SaverDef()
        new_saver_def.CopyFrom(self.to_proto())
        item.info.update_savers([new_saver_def], replace=False)

    def save(self,
             sess,
             save_path,
             global_step=None,
             latest_filename=None,
             meta_graph_suffix="meta",
             write_meta_graph=True,
             write_state=True,
             strip_default_attrs=False,
             save_debug_info=False):
        # pylint: disable=too-many-arguments, too-many-locals
        """
        Save the checkpoint to "save_path" using the saver managed by AutoDist.

        Follows the same signature with tf.saver save().
        """
        # sess is a WrappedSession instead of a tensorflow session.
        if not sess:
            raise ValueError('Saver must be used before create_distributed_session().')
        with sess._graph_item.graph.as_default():
            if not sess._graph_item.info.savers:
                raise ValueError("No saver captured by AutoDist.")
            super(Saver, self).save(sess,
                                    save_path,
                                    global_step=global_step,
                                    latest_filename=latest_filename,
                                    meta_graph_suffix=meta_graph_suffix,
                                    write_meta_graph=write_meta_graph,
                                    write_state=write_state,
                                    strip_default_attrs=strip_default_attrs,
                                    save_debug_info=save_debug_info)

    def restore(self, sess, save_path):
        """
        Restore the checkpoint from "save_path" using the saver managed by AutoDist.

        Follows the same signature with tf.saver restore().
        """
        # sess is a wrapped session instead of tensorflow session.
        with sess._graph_item.graph.as_default():
            if not sess._graph_item.info.savers:
                raise ValueError("No saver captured by AutoDist.")
            super(Saver, self).restore(sess, save_path)
