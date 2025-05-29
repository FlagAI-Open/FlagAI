import torch
from .block_layer import CheckpointBlock, TransformerBlockList
from .layer import DistributedModule, DistributedParameter

def make_distributed(model : torch.nn.Module):
    for kw in list(model._parameters.keys()):
        if model._parameters[kw] is not None:
            if not isinstance(model._parameters[kw], DistributedParameter):
                model._parameters[kw] = DistributedParameter(model._parameters[kw], requires_grad=model._parameters[kw].requires_grad)
    
    for kw in list(model._buffers.keys()):
        if model._buffers[kw] is not None:
            model._buffers[kw] = model._buffers[kw].cuda()
    
    for kw in list(model._modules.keys()):
        if isinstance(model, torch.nn.ModuleList):
            if not isinstance(model._modules[kw], CheckpointBlock):
                model._modules[kw] = CheckpointBlock(model_wrapper_dispatch(model._modules[kw]))
        else:
            model._modules[kw] = model_wrapper_dispatch(model._modules[kw])

    model.__class__ = type("bmtrain.Distributed" + model.__class__.__name__, (model.__class__, DistributedModule), {})
    return model

def model_wrapper_dispatch(model : torch.nn.Module):
    if isinstance(model, TransformerBlockList):
        return model
    elif isinstance(model, DistributedModule):
        return model
    elif isinstance(model, CheckpointBlock):
        return model
    else:
        return make_distributed(model)

def BMTrainModelWrapper(model : torch.nn.Module) -> torch.nn.Module:
    """
    Automatically wrap a model in a BMTrain model.
    Replaces all parameters with DistributedParameter, all modules with DistributedModule, and modules in ModuleList with CheckpointBlock.
    """
    return model_wrapper_dispatch(model)