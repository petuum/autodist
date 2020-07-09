.. The documentation master file, created by
   sphinx-quickstart on Wed Oct  3 21:24:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AutoDist!
==================================


AutoDist is a distributed deep-learning training engine.
AutoDist provides a user-friendly interface to distribute the training of a wide variety of deep learning models
across many GPUs with scalability and minimal code change.

.. toctree::
   :titlesonly:

   Overview<README.md>
   Performance<usage/performance.md>

.. toctree::
   :titlesonly:
   :caption: Tutorials

   usage/tutorials/getting-started.md
   usage/tutorials/multi-node.md
   usage/tutorials/save-restore.md
   usage/tutorials/installation.md
   usage/tutorials/docker.md
   usage/tutorials/choose-strategy.md
   usage/tutorials/customize-strategy.md
   usage/faq.md


.. toctree::
   :titlesonly:
   :caption: Developer Reference
   :maxdepth: 1

   design/rationale.md
   design/architecture.rst
   design/kernels.md
   usage/proto_docgen.md
   Developer API References<api/autodist.rst>
