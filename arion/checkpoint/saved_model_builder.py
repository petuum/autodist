"""AutoDist SavedModelBuilder."""
from tensorflow.python.saved_model.builder_impl import SavedModelBuilder as tf_SavedModelBuilder
from tensorflow.python.saved_model import utils_impl as saved_model_utils


class SavedModelBuilder(tf_SavedModelBuilder):
    """Wrapper of SavedModelBuilder."""

    def __init__(self, export_dir):
        super(SavedModelBuilder, self).__init__(export_dir=export_dir)

    def add_meta_graph_and_variables(self,      # pylint: disable=too-many-arguments
                                     sess,
                                     tags,
                                     signature_def_map=None,
                                     assets_collection=None,
                                     legacy_init_op=None,
                                     clear_devices=False,
                                     main_op=None,
                                     strip_default_attrs=False,
                                     saver=None,
                                     train_op=None):
        """Save graph variables and metagraph."""
        # users must provide an autodist saver to use saved_model
        # We must use the autodist.saver to save variables, but the saver has to be
        # created before the autodist_distributed_session. In this case, we can't create
        # an autodist saver automatically for users.
        assert saver is not None, "An autodist saver must be provided!"

        if self._has_saved_variables:       # pylint: disable=access-member-before-definition
            raise AssertionError("Graph state including variables and assets has "
                                 "already been saved. Please invoke "
                                 "`add_meta_graph()` instead.")
        signature_def_map = signature_def_map or {}
        self._validate_signature_def_map(signature_def_map)
        main_op = main_op or legacy_init_op
        self._add_collections(assets_collection, main_op, train_op)
        saved_model_utils.get_or_create_variables_dir(self._export_dir)
        variables_path = saved_model_utils.get_variables_path(self._export_dir)

        saver.save(sess, variables_path, write_meta_graph=False, write_state=False)
        meta_graph_def = saver.export_meta_graph(
            clear_devices=clear_devices, strip_default_attrs=strip_default_attrs)

        self._tag_and_add_meta_graph(meta_graph_def, tags, signature_def_map)
        self._has_saved_variables = True    # pylint: disable=attribute-defined-outside-init
