OpenDelta's documentation!
=====================================

[OpenDelta](https://github.com/thunlp/OpenDelta/) is a **Plug-and-play** Library of the parameter-efficient fine-tuning ([delta-tuning](WhatisDelta)) technology for pre-trained models.


## Essential Advantages:

- <span style="color:rgb(81, 217, 245);font-weight:bold">Clean:</span> No need to edit the backbone PTM’s codes.
- <span style="color:orange;font-weight:bold">Simple:</span> Migrating from full-model tuning to delta-tuning needs as little as 3 lines of codes.
- <span style="color:green;font-weight:bold">Sustainable:</span> Most evolution in external library doesn’t require a new OpenDelta.
- <span style="color:red;font-weight:bold">Extendable:</span> Various PTMs can share the same delta-tuning codes.
- <span style="color:purple;font-weight:bold">Flexible:</span> Able to apply delta-tuning to (almost) any position of the PTMs.

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   notes/overview.md
   notes/installation.md
   notes/quickstart.md
   notes/custom.md
   

.. toctree::
   :maxdepth: 1
   :caption: Advanced Usage
   
   notes/autodelta.md
   notes/deltacenter.md
   notes/composition.md
   notes/pluginunplug.md
   notes/withbmtrain.md
   notes/withaccelerate.md
   notes/examples.md

.. toctree::
   :maxdepth: 1
   :caption: Utilities
   
   notes/inspect.md

.. toctree::
   :maxdepth: 1
   :caption: Mechanisms

   notes/keyfeature.md
   notes/namebasedaddr.md
   notes/unifyname.md

.. toctree::
   :maxdepth: 1
   :caption: Information

   notes/citation.md
   notes/update.md
   notes/faq.md

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   modules/base
   modules/deltas
   modules/auto_delta
   modules/utils


Indices and tables
==================

* :ref:`genindex`

```