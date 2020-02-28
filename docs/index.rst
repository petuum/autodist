.. The documentation master file, created by
   sphinx-quickstart on Wed Oct  3 21:24:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AutoDist!
==================================


AutoDist is a scalable ML engine.
AutoDist provides a user-friendly interface to distribute
TensorFlow model training across multiple processing units
(for example, distributed GPU clusters) with high scalability
and minimal code change.


.. toctree::
   :titlesonly:
   :caption: Tutorials

   usage/tutorials/get-started.md
   usage/tutorials/multi-node.md
   usage/tutorials/strategy-builders.md
   usage/tutorials/ux.md
   usage/tutorials/save-restore.md


.. toctree::
   :titlesonly:
   :caption: Dev Reference
   :maxdepth: 1

   Developer API Documentation<api/autodist.rst>
   usage/symphony-integration.md
   design/architecture.rst

Useful Links
------------

- `Source Code`_
- `Development Notes on Confluence`_
- `Performance Benchmarks`_

.. _`Source Code`: https://gitlab.int.petuum.com/internal/scalable-ml/autodist
.. _`Development Notes on Confluence`: https://petuum.atlassian.net/wiki/spaces/SYM/pages/166363204/AutoDist%3A+Goal+and+a+Proposal+of+Architecture?atlOrigin=eyJpIjoiOWU1Y2Q4YzNmMDg2NDkyZTk0Njg1ZTYwNmM3YWI1MDciLCJwIjoiYyJ9
.. _`Performance Benchmarks`: https://petuum.atlassian.net/wiki/spaces/SYM/pages/253100101/AutoDist+Performance


Recent Changes
--------------
.. git_changelog::
   :revisions: 3
