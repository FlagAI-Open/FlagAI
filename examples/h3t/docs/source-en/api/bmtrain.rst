====================
bmtrain
====================


Initialization 
==========================================

.. autofunction:: bmtrain.init_distributed

Distributed Parameters and Modules
==========================================

.. autoclass:: bmtrain.DistributedParameter
    :members:
    :show-inheritance:

.. autoclass:: bmtrain.ParameterInitializer
    :members:
    :show-inheritance:

.. autoclass:: bmtrain.DistributedModule
    :members:
    :show-inheritance:

.. autoclass:: bmtrain.CheckpointBlock
    :members:
    :show-inheritance:

.. autoclass:: bmtrain.TransformerBlockList
    :members:
    :show-inheritance:

Methods for Parameters
==========================================

.. autofunction:: bmtrain.init_parameters

.. autofunction:: bmtrain.grouped_parameters

.. autofunction:: bmtrain.save

.. autofunction:: bmtrain.load

Utilities
==========================================

.. autofunction:: bmtrain.rank

.. autofunction:: bmtrain.world_size

.. autofunction:: bmtrain.print_rank

.. autofunction:: bmtrain.synchronize

.. autofunction:: bmtrain.sum_loss

.. autofunction:: bmtrain.optim_step
